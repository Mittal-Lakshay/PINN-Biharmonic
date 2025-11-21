import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from time import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(0)
np.random.seed(0)

# 1. Problem definition 
def u_exact(x, y):
    return 0.5 / np.pi**2 * np.sin(np.pi * x) * np.sin(np.pi * y)

def f_source(x, y):
    # Δ²u for u = (1 / (2π²)) sin(πx) sin(πy)
    return 2 * np.pi**2 * np.sin(np.pi * x) * np.sin(np.pi * y)

def g1(x, y):
    return u_exact(x, y)
def g2(x, y):
    return -2 * np.sin(np.pi * x) * np.sin(np.pi * y)


# 2. Neural Network 
class PINN_Biharmonic(nn.Module):
    def __init__(self, layers=(128,128,128,128), activation=nn.Tanh()):
        super().__init__()
        net = []
        in_dim = 2
        for h in layers:
            net.append(nn.Linear(in_dim, h))
            net.append(activation)
            in_dim = h
        net.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*net)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            nn.init.uniform_(m.bias, -0.1, 0.1)

    def forward(self, x, y):
        inp = torch.cat([x, y], dim=1)
        return self.net(inp)


# 3. Helper AD functions 
def laplacian(u, x, y):
    grads = torch.autograd.grad(u, [x, y], grad_outputs=torch.ones_like(u), create_graph=True)
    u_x, u_y = grads
    u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
    u_yy = torch.autograd.grad(u_y, y, grad_outputs=torch.ones_like(u_y), create_graph=True)[0]
    return u_xx + u_yy

def biharmonic(u, x, y):
    Δu = laplacian(u, x, y)
    return laplacian(Δu, x, y)


# 4. Training data (unchanged)
N_int = 10000
N_bc = 8000

x_int = np.random.rand(N_int, 1).astype(np.float32)
y_int = np.random.rand(N_int, 1).astype(np.float32)

s = np.linspace(0, 1, N_bc//4, dtype=np.float32).reshape(-1,1)
x_bottom = s; y_bottom = np.zeros_like(s)
x_top = s;    y_top = np.ones_like(s)
y_left = s;   x_left = np.zeros_like(s)
y_right = s;  x_right = np.ones_like(s)

x_bc = np.vstack([x_bottom, x_top, x_left, x_right]).astype(np.float32)
y_bc = np.vstack([y_bottom, y_top, y_left, y_right]).astype(np.float32)

# tensors
x_int_t = torch.tensor(x_int, dtype=torch.float32, requires_grad=True).to(device)
y_int_t = torch.tensor(y_int, dtype=torch.float32, requires_grad=True).to(device)
x_bc_t  = torch.tensor(x_bc, dtype=torch.float32, requires_grad=True).to(device)
y_bc_t  = torch.tensor(y_bc, dtype=torch.float32, requires_grad=True).to(device)

f_int   = torch.tensor(f_source(x_int, y_int), dtype=torch.float32).to(device)
u_bc    = torch.tensor(g1(x_bc, y_bc), dtype=torch.float32).to(device)
Δu_bc   = torch.tensor(g2(x_bc, y_bc), dtype=torch.float32).to(device)


# 5. Model & optimizers
model = PINN_Biharmonic(layers=(128,128,128,128)).to(device)

ADAM_EPOCHS = 5000                
print_every = 100                  

optimizer_adam = optim.Adam(model.parameters(), lr=1e-3)

LBFGS_MAX_ITER = 2000             
PRINT_EVERY = 100                 

optimizer_lbfgs = optim.LBFGS(
    model.parameters(),
    lr=1.0,
    max_iter=LBFGS_MAX_ITER,
    max_eval=LBFGS_MAX_ITER,
    history_size=50,
    tolerance_grad=1e-7,
    tolerance_change=1e-9,
    line_search_fn='strong_wolfe'
)

mse = nn.MSELoss()
λ_int, λ_bc = 1.0, 2.0


# 6. ADAM training (phase 1)
loss_history_adam = []
best_loss_adam = float('inf')

start_time = time()
print("\nStarting ADAM training...")

for epoch in range(1, ADAM_EPOCHS + 1):
    optimizer_adam.zero_grad()

    # interior residual
    u_pred = model(x_int_t, y_int_t)
    Δ2u_pred = biharmonic(u_pred, x_int_t, y_int_t)
    loss_int = mse(Δ2u_pred, f_int)

    # boundary residuals
    u_bc_pred = model(x_bc_t, y_bc_t)
    Δu_bc_pred = laplacian(u_bc_pred, x_bc_t, y_bc_t)
    loss_bc = mse(u_bc_pred, u_bc) + mse(Δu_bc_pred, Δu_bc)

    loss = λ_int * loss_int + λ_bc * loss_bc

    loss.backward()
    optimizer_adam.step()

    loss_val = loss.item()
    loss_history_adam.append(loss_val)

    if loss_val < best_loss_adam:
        best_loss_adam = loss_val

    if epoch % print_every == 0 or epoch == 1:
        print(f"Adam Epoch {epoch:5d} | Loss = {loss_val:.6e} | Best_Adam = {best_loss_adam:.6e}")

end_time_adam = time()
print(f"\nADAM phase finished in {end_time_adam - start_time:.2f} s. Final Adam loss = {loss_history_adam[-1]:.6e}")


# 7. L-BFGS training (phase 2) 
print("\nSwitching to L-BFGS optimization...")

lbfgs_losses = []
lbfgs_iter = 0
best_loss_lbfgs = float('inf')   # <-- NEW

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
    loss_bc = mse(u_bc_pred, u_bc) + mse(Δu_bc_pred, Δu_bc)

    # total loss
    loss_total = λ_int * loss_int + λ_bc * loss_bc

    # backward pass
    loss_total.backward()

    # update best loss
    current_loss = loss_total.item()
    if current_loss < best_loss_lbfgs:
        best_loss_lbfgs = current_loss

    # tracking + printing
    lbfgs_losses.append(current_loss)

    if lbfgs_iter % PRINT_EVERY == 0:
        print(f"L-BFGS Iter {lbfgs_iter:6d} | Loss = {current_loss:.6e} | Best = {best_loss_lbfgs:.6e}")

    lbfgs_iter += 1
    return loss_total


# RUN L-BFGS
optimizer_lbfgs.step(lbfgs_closure)

if len(lbfgs_losses) > 0:
    final_loss_lbfgs = lbfgs_losses[-1]
else:
    print("Warning: L-BFGS made zero closure calls. Computing final loss manually.")
    with torch.no_grad():
        u_pred = model(x_int_t, y_int_t)
        Δ2u_pred = biharmonic(u_pred, x_int_t, y_int_t)
        loss_int = mse(Δ2u_pred, f_int)
        u_bc_pred = model(x_bc_t, y_bc_t)
        Δu_bc_pred = laplacian(u_bc_pred, x_bc_t, y_bc_t)
        loss_bc = mse(u_bc_pred, u_bc) + mse(Δu_bc_pred, Δu_bc)
        final_loss_lbfgs = (λ_int * loss_int + λ_bc * loss_bc).item()

end_time_lbfgs = time()

print(f"\nL-BFGS finished.")
print(f"Final L-BFGS loss   = {final_loss_lbfgs:.6e}")
print(f"Best L-BFGS loss    = {best_loss_lbfgs:.6e}")
print(f"Total training time = {end_time_lbfgs - start_time:.2f} s")


# 8. Error computation (L2, H1, H2) 
model.eval()
nx, ny = 200, 200
xg, yg = np.meshgrid(np.linspace(0,1,nx), np.linspace(0,1,ny))

xt = torch.tensor(xg.reshape(-1,1), dtype=torch.float32, requires_grad=True).to(device)
yt = torch.tensor(yg.reshape(-1,1), dtype=torch.float32, requires_grad=True).to(device)

u_pred = model(xt, yt)

# first derivatives
u_x = torch.autograd.grad(u_pred, xt, grad_outputs=torch.ones_like(u_pred), create_graph=True)[0]
u_y = torch.autograd.grad(u_pred, yt, grad_outputs=torch.ones_like(u_pred), create_graph=True)[0]

# second derivatives
u_xx = torch.autograd.grad(u_x, xt, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
u_yy = torch.autograd.grad(u_y, yt, grad_outputs=torch.ones_like(u_y), create_graph=True)[0]
u_xy = torch.autograd.grad(u_x, yt, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]

# reshape to grid
u_pred_test = u_pred.detach().cpu().numpy().reshape(nx, ny)
u_ex_test   = u_exact(xg, yg)

u_x_np  = u_x.detach().cpu().numpy().reshape(nx, ny)
u_y_np  = u_y.detach().cpu().numpy().reshape(nx, ny)
u_xx_np = u_xx.detach().cpu().numpy().reshape(nx, ny)
u_yy_np = u_yy.detach().cpu().numpy().reshape(nx, ny)
u_xy_np = u_xy.detach().cpu().numpy().reshape(nx, ny)

# analytic derivatives (consistent with u_exact)
u_x_ex  = np.pi * np.cos(np.pi * xg) * np.sin(np.pi * yg) / (2*np.pi**2)
u_y_ex  = np.pi * np.sin(np.pi * xg) * np.cos(np.pi * yg) / (2*np.pi**2)
u_xx_ex = -np.sin(np.pi * xg) * np.sin(np.pi * yg) / 2
u_yy_ex = -np.sin(np.pi * xg) * np.sin(np.pi * yg) / 2
u_xy_ex = np.pi**2 * np.cos(np.pi * xg) * np.cos(np.pi * yg) / (2*np.pi**2)

# relative (corrected) errors
L2_err = np.linalg.norm(u_pred_test - u_ex_test) / np.linalg.norm(u_ex_test)

H1_err = np.sqrt(
    L2_err**2 +
    (np.linalg.norm(u_x_np - u_x_ex)/np.linalg.norm(u_x_ex))**2 +
    (np.linalg.norm(u_y_np - u_y_ex)/np.linalg.norm(u_y_ex))**2
)

H2_err = np.sqrt(
    H1_err**2 +
    (np.linalg.norm(u_xx_np - u_xx_ex)/np.linalg.norm(u_xx_ex))**2 +
    (np.linalg.norm(u_yy_np - u_yy_ex)/np.linalg.norm(u_yy_ex))**2 +
    (np.linalg.norm(u_xy_np - u_xy_ex)/np.linalg.norm(u_xy_ex))**2
)

print("\nCorrected Relative Errors:")
print(f"  L2  = {L2_err:.3e}")
print(f"  H1  = {H1_err:.3e}")
print(f"  H2  = {H2_err:.3e}")


# 9. Visualization
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
im0 = axes[0].imshow(u_ex_test, extent=[0,1,0,1], origin='lower', cmap='viridis')
axes[0].set_title("Exact Solution $u_{exact}(x,y)$", fontsize=13)
plt.colorbar(im0, ax=axes[0])

im1 = axes[1].imshow(u_pred_test, extent=[0,1,0,1], origin='lower', cmap='viridis')
axes[1].set_title("PINN Prediction $u_\\theta(x,y)$", fontsize=13)
plt.colorbar(im1, ax=axes[1])

im2 = axes[2].imshow(np.abs(u_pred_test - u_ex_test), extent=[0,1,0,1], origin='lower', cmap='inferno')
axes[2].set_title("Absolute Error $|u_\\theta - u_{exact}|$", fontsize=13)
plt.colorbar(im2, ax=axes[2])

plt.tight_layout()
plt.show()


# 10. Final summary print
print("="*60)
print("BIHARMONIC PINN RESULTS SUMMARY")
print("="*60)
print(f"Neural Network Architecture : [2 -> 128 -> 128 -> 128 -> 128 -> 1]")
print(f"Activation Function         : Tanh()")
print(f"Adam epochs (phase 1)       : {ADAM_EPOCHS}")
print(f"L-BFGS max_iter (phase 2)   : {LBFGS_MAX_ITER}")
print(f"λ_int, λ_bc                 : {λ_int}, {λ_bc}")
print(f"Total Time (s)              : {end_time_lbfgs - start_time:.2f}")
print("-"*60)
print(f"Final L-BFGS Loss           : {final_loss_lbfgs:.6e}")
print(f"Best Adam Loss (kept)       : {best_loss_adam:.6e}")
print(f"Relative L2 Error           : {L2_err:.3e}")
print(f"Relative H1 Error           : {H1_err:.3e}")
print(f"Relative H2 Error           : {H2_err:.3e}")
print("="*60)