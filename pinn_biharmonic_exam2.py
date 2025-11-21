import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from time import time
from matplotlib.ticker import MaxNLocator

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(0)
np.random.seed(0)


# Problem definition 
def u_exact(x, y):
    return (x**2) * (y**2) * (1 - x)**2 * (1 - y)**2

def laplacian_u_exact(x, y):
    term1 = y**2 * (1-y)**2 * (6*x**2 - 6*x + 1)
    term2 = x**2 * (1-x)**2 * (6*y**2 - 6*y + 1)
    return 2*(term1 + term2)

def f_source(x, y):
    return (0*x + 0*y)

def g1(x, y):
    return u_exact(x, y)

def g2(x, y):
    return laplacian_u_exact(x, y)


# Autograd helpers 
def laplacian(u, x, y):
    u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_y = torch.autograd.grad(u, y, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
    u_yy = torch.autograd.grad(u_y, y, grad_outputs=torch.ones_like(u_y), create_graph=True)[0]
    return u_xx + u_yy

def biharmonic(u, x, y):
    lap_u = laplacian(u, x, y)
    return laplacian(lap_u, x, y)


# 2. Neural Network Architecture 
class PINN_Biharmonic(nn.Module):
    def __init__(self, layers=(128,128,128,128)):
        super().__init__()
        net = []
        in_dim = 2
        for h in layers:
            net.append(nn.Linear(in_dim, h))
            net.append(nn.Tanh())
            in_dim = h
        net.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*net)
        self.apply(self.init_weights)

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, x, y):
        return self.net(torch.cat([x,y], dim=1))


# 4. Training Data (Interior + Boundary)
N_int = 20000
N_bc  = 10000

# interior pts
x_int = np.random.rand(N_int,1).astype(np.float32)
y_int = np.random.rand(N_int,1).astype(np.float32)

# boundary pts
s = np.linspace(0,1,N_bc//4, dtype=np.float32).reshape(-1,1)
x_b = np.vstack([s, s, np.zeros_like(s), np.ones_like(s)]).astype(np.float32)
y_b = np.vstack([np.zeros_like(s), np.ones_like(s), s, s]).astype(np.float32)

# tensors
x_int_t = torch.tensor(x_int, device=device, requires_grad=True)
y_int_t = torch.tensor(y_int, device=device, requires_grad=True)

x_bc_t = torch.tensor(x_b, device=device, requires_grad=True)
y_bc_t = torch.tensor(y_b, device=device, requires_grad=True)

# exact u on boundary
u_bc = torch.tensor(u_exact(x_b, y_b), dtype=torch.float32, device=device)
lap_bc = torch.tensor(laplacian_u_exact(x_b, y_b), dtype=torch.float32, device=device)


# Compute analytic BIHARMONIC of exact u(x,y) at interior

# A(x) = x^2(1-x)^2 = x^2 - 2x^3 + x^4
x = x_int_t
y = y_int_t

A     = x**2 * (1-x)**2
A1    = 2*x - 6*x**2 + 4*x**3
A2    = 2 - 12*x + 12*x**2
A3    = -12 + 24*x
A4    = 24*torch.ones_like(x)

B     = y**2 * (1-y)**2
B1    = 2*y - 6*y**2 + 4*y**3
B2    = 2 - 12*y + 12*y**2
B3    = -12 + 24*y
B4    = 24*torch.ones_like(y)

# Δ²u = A'''' B + 2 A'' B'' + A B''''
f_int = A4*B + 2*A2*B2 + A*B4
f_int = f_int.detach()   


model = PINN_Biharmonic(layers=(128,128,128,128)).to(device)

ADAM_EPOCHS = 5000       
PRINT_EVERY_ADAM = 100

optimizer = optim.Adam(model.parameters(), lr=1e-2)   
mse = nn.MSELoss()

LBFGS_MAX_ITER = 2000    
PRINT_EVERY = 100        

optimizer_lbfgs = torch.optim.LBFGS(
    model.parameters(),
    lr=1.0,
    max_iter=LBFGS_MAX_ITER,
    max_eval=LBFGS_MAX_ITER,
    history_size=50,
    tolerance_grad=1e-7,
    tolerance_change=1e-9,
    line_search_fn='strong_wolfe'
)

λ_int = 1.0        
λ_bc = 1.0         

loss_history_adam = []
lbfgs_losses = []
best_loss = 1e30
best_epoch = -1


# 6. Training Loop

start_time = time()
print("\nTraining Started (ADAM)...\n")

for epoch in range(1, ADAM_EPOCHS+1):
    optimizer.zero_grad()

    # Interior PDE: Δ²u = f
    xt = x_int_t.clone().requires_grad_(True)
    yt = y_int_t.clone().requires_grad_(True)
    u_pred = model(xt, yt)

    # Laplacian then biharmonic via autograd
    u_x  = torch.autograd.grad(u_pred, xt, grad_outputs=torch.ones_like(u_pred), create_graph=True)[0]
    u_y  = torch.autograd.grad(u_pred, yt, grad_outputs=torch.ones_like(u_pred), create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x, xt, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
    u_yy = torch.autograd.grad(u_y, yt, grad_outputs=torch.ones_like(u_y), create_graph=True)[0]
    lap_u = u_xx + u_yy

    lap_u_x  = torch.autograd.grad(lap_u, xt, grad_outputs=torch.ones_like(lap_u), create_graph=True)[0]
    lap_u_y  = torch.autograd.grad(lap_u, yt, grad_outputs=torch.ones_like(lap_u), create_graph=True)[0]
    laplap_u = (
        torch.autograd.grad(lap_u_x, xt, grad_outputs=torch.ones_like(lap_u_x), create_graph=True)[0] +
        torch.autograd.grad(lap_u_y, yt, grad_outputs=torch.ones_like(lap_u_y), create_graph=True)[0]
    )

    loss_pde = mse(laplap_u, f_int)

    # Boundary loss
    xb = x_bc_t.clone().requires_grad_(True)
    yb = y_bc_t.clone().requires_grad_(True)
    u_pred_b = model(xb, yb)

    ux_b  = torch.autograd.grad(u_pred_b, xb, grad_outputs=torch.ones_like(u_pred_b), create_graph=True)[0]
    uy_b  = torch.autograd.grad(u_pred_b, yb, grad_outputs=torch.ones_like(u_pred_b), create_graph=True)[0]
    uxx_b = torch.autograd.grad(ux_b, xb, grad_outputs=torch.ones_like(ux_b), create_graph=True)[0]
    uyy_b = torch.autograd.grad(uy_b, yb, grad_outputs=torch.ones_like(uy_b), create_graph=True)[0]
    lap_u_pred_b = uxx_b + uyy_b

    loss_bc = mse(u_pred_b, u_bc) + mse(lap_u_pred_b, lap_bc)

    loss = λ_int * loss_pde + λ_bc * loss_bc

    # backward + step
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

    loss_val = loss.item()
    loss_history_adam.append(loss_val)

    if loss_val < best_loss:
        best_loss = loss_val
        best_epoch = epoch

    if epoch % PRINT_EVERY_ADAM == 0 or epoch == 1:
        print(f"Epoch {epoch:6d} | Loss={loss_val:.3e} | PDE={loss_pde.item():.3e} | BC={loss_bc.item():.3e} | Best={best_loss:.3e} @ {best_epoch}")

end_time_adam = time()
print(f"\nADAM phase finished in {end_time_adam - start_time:.2f} s. Final Adam loss = {loss_history_adam[-1]:.6e}")
print(f"Best Loss during Adam = {best_loss:.6e} at epoch {best_epoch}")


# L-BFGS training (phase 2) 
print("\nSwitching to L-BFGS optimization...")

lbfgs_iter = 0
best_loss_lbfgs = float('inf')

def lbfgs_closure():
    global lbfgs_iter, best_loss_lbfgs
    optimizer_lbfgs.zero_grad()

    # interior residual
    u_pred = model(x_int_t, y_int_t)
    Δ2u_pred = biharmonic(u_pred, x_int_t, y_int_t)
    loss_int = mse(Δ2u_pred, f_int)

    # boundary residual
    u_bc_pred = model(x_bc_t, y_bc_t)
    Δu_bc_pred = laplacian(u_bc_pred, x_bc_t, y_bc_t)
    loss_bc = mse(u_bc_pred, u_bc) + mse(Δu_bc_pred, lap_bc)

    loss_total = λ_int * loss_int + λ_bc * loss_bc

    # backward
    loss_total.backward()

    current_loss = loss_total.item()
    lbfgs_losses.append(current_loss)

    if current_loss < best_loss_lbfgs:
        best_loss_lbfgs = current_loss

    if lbfgs_iter % PRINT_EVERY == 0:
        print(f"L-BFGS Iter {lbfgs_iter:6d} | Loss = {current_loss:.6e} | Best_LBFGS = {best_loss_lbfgs:.6e}")

    lbfgs_iter += 1
    return loss_total

# run L-BFGS 
optimizer_lbfgs.step(lbfgs_closure)

if len(lbfgs_losses) > 0:
    final_loss_lbfgs = lbfgs_losses[-1]
else:
    u_pred = model(x_int_t, y_int_t)
    Δ2u_pred = biharmonic(u_pred, x_int_t, y_int_t)
    loss_int = mse(Δ2u_pred, f_int)
    u_bc_pred = model(x_bc_t, y_bc_t)
    Δu_bc_pred = laplacian(u_bc_pred, x_bc_t, y_bc_t)
    loss_bc = mse(u_bc_pred, u_bc) + mse(Δu_bc_pred, lap_bc)
    final_loss_lbfgs = (λ_int * loss_int + λ_bc * loss_bc).item()

end_time_lbfgs = time()
print(f"\nL-BFGS finished.")
print(f"Final L-BFGS loss   = {final_loss_lbfgs:.6e}")
print(f"Best L-BFGS loss    = {best_loss_lbfgs:.6e}")
print(f"Total training time = {end_time_lbfgs - start_time:.2f} s")

model.eval()
nx = ny = 100
xg, yg = np.meshgrid(np.linspace(0,1,nx), np.linspace(0,1,ny))

xt = torch.tensor(xg.reshape(-1,1), device=device, dtype=torch.float32, requires_grad=True)
yt = torch.tensor(yg.reshape(-1,1), device=device, dtype=torch.float32, requires_grad=True)

# --- MODEL PREDICTION ---
u_pred = model(xt, yt)
u_pred_np = u_pred.detach().cpu().numpy().reshape(nx, ny)

# exact
u_ex_np = u_exact(xg, yg)

# ---- L2 ----
L2_num = np.linalg.norm(u_pred_np - u_ex_np)
L2_den = np.linalg.norm(u_ex_np) + 1e-14
L2_rel = L2_num / L2_den

# ---- First derivatives ----
u_x = torch.autograd.grad(u_pred, xt, grad_outputs=torch.ones_like(u_pred), create_graph=True)[0]
u_y = torch.autograd.grad(u_pred, yt, grad_outputs=torch.ones_like(u_pred), create_graph=True)[0]
u_x_np = u_x.detach().cpu().numpy().reshape(nx,ny)
u_y_np = u_y.detach().cpu().numpy().reshape(nx,ny)

# EXACT derivatives
A1_x = (2*xg - 6*xg**2 + 4*xg**3)
B_y  = (yg**2 * (1-yg)**2)
u_x_ex = A1_x * B_y

A_x     = (xg**2 * (1-xg)**2)
B1_y    = (2*yg - 6*yg**2 + 4*yg**3)
u_y_ex  = A_x * B1_y

u_x_ex = u_x_ex.reshape(nx,ny)
u_y_ex = u_y_ex.reshape(nx,ny)

G_num = np.sqrt(np.linalg.norm(u_x_np - u_x_ex)**2 + np.linalg.norm(u_y_np - u_y_ex)**2)
G_den = np.sqrt(np.linalg.norm(u_x_ex)**2 + np.linalg.norm(u_y_ex)**2) + 1e-14
H1_rel = np.sqrt((L2_num/L2_den)**2 + (G_num/G_den)**2)

# ---- Second derivatives ----
u_xx = torch.autograd.grad(u_x, xt, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
u_yy = torch.autograd.grad(u_y, yt, grad_outputs=torch.ones_like(u_y), create_graph=True)[0]

u_xx_np = u_xx.detach().cpu().numpy().reshape(nx,ny)
u_yy_np = u_yy.detach().cpu().numpy().reshape(nx,ny)

A_xx = (2 - 12*xg + 12*xg**2)
B_y  = (yg**2 * (1-yg)**2)
u_xx_ex = A_xx * B_y

B_yy = (2 - 12*yg + 12*yg**2)
u_yy_ex = A_x * B_yy

u_xx_ex = u_xx_ex.reshape(nx,ny)
u_yy_ex = u_yy_ex.reshape(nx,ny)

H2_num = np.sqrt(np.linalg.norm(u_xx_np - u_xx_ex)**2 + np.linalg.norm(u_yy_np - u_yy_ex)**2)
H2_den = np.sqrt(np.linalg.norm(u_xx_ex)**2 + np.linalg.norm(u_yy_ex)**2) + 1e-14

H2_rel = np.sqrt((L2_num/L2_den)**2 + (G_num/G_den)**2 + (H2_num/H2_den)**2)

print(f"Relative L2 error  = {L2_rel:.3e}")
print(f"Relative H1 error  = {H1_rel:.3e}")
print(f"Relative H2 error  = {H2_rel:.3e}")


# 8. Visualization & Report Summary
from matplotlib.ticker import MaxNLocator

adam_iters = np.arange(len(loss_history_adam))
lbfgs_iters = np.arange(len(lbfgs_losses)) + len(loss_history_adam)

plt.figure(figsize=(8,4))
plt.semilogy(adam_iters, loss_history_adam, label="Adam", linewidth=2)
if len(lbfgs_losses) > 0:
    plt.semilogy(lbfgs_iters, lbfgs_losses, label="L-BFGS", linewidth=2)
plt.axvline(x=len(loss_history_adam), color='black', linestyle='--', linewidth=1)
plt.text(len(loss_history_adam) + 5, plt.ylim()[0]*10, "L-BFGS start", fontsize=10)
plt.xlabel("Iteration (Adam epochs then L-BFGS iters)", fontsize=12)
plt.ylabel("Loss (log scale)", fontsize=12)
plt.title("Training Loss History (Adam → L-BFGS)", fontsize=14)
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()

fig, axes = plt.subplots(1, 3, figsize=(16, 4))
im0 = axes[0].imshow(u_ex_np, origin='lower', extent=[0,1,0,1], cmap='viridis'); axes[0].set_title("Exact")
plt.colorbar(im0, ax=axes[0])
im1 = axes[1].imshow(u_pred_np, origin='lower', extent=[0,1,0,1], cmap='viridis'); axes[1].set_title("Predicted")
plt.colorbar(im1, ax=axes[1])
im2 = axes[2].imshow(np.abs(u_pred_np - u_ex_np), origin='lower', extent=[0,1,0,1], cmap='inferno'); axes[2].set_title("Abs Error")
plt.colorbar(im2, ax=axes[2])
plt.tight_layout()
plt.show()


# 9. Final summary print
print("="*60)
print(" BIHARMONIC PINN — FINAL RESULTS")
print("="*60)
print(f"Best loss during ADAM      : {best_loss:.3e} @ epoch {best_epoch}")
print(f"Final L-BFGS loss         : {final_loss_lbfgs:.6e}")
print(f"Best L-BFGS loss          : {best_loss_lbfgs:.6e}")
print(f"Total training time (s)   : {end_time_lbfgs - start_time:.2f}")
print("-"*60)
print(f"L2 relative error         : {L2_rel:.3e}")
print(f"H1 relative error         : {H1_rel:.3e}")
print(f"H2 relative error         : {H2_rel:.3e}")
print("="*60)