import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from time import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(0)
np.random.seed(0)

# ----------------------------
# Exact solution and derivatives (Example 3.2)
# ----------------------------
def u_exact(x, y):
    return (x**2) * (y**2) * (1 - x)**2 * (1 - y)**2

# --------------------------------------------------
# Laplacian of exact u
# --------------------------------------------------
def laplacian_u_exact(x, y):
    term1 = y**2 * (1-y)**2 * (6*x**2 - 6*x + 1)
    term2 = x**2 * (1-x)**2 * (6*y**2 - 6*y + 1)
    return 2*(term1 + term2)

# --------------------------------------------------
# Biharmonic source:  f = Δ² u
# (We compute via autodiff later, but this is fine)
# --------------------------------------------------
def f_source(x, y):
    # You may set f = 0 and let PINN compute Δ²u automatically,
    # but we keep analytic form for stability.
    return (
        0*x +  # placeholder to match shape
        0*y
    )  # You will compute Δ²u using PINN autodiff internally

# Boundary data
def g1(x, y):   # u = exact
    return u_exact(x, y)

def g2(x, y):   # Δu = exact Laplacian
    return laplacian_u_exact(x, y)

# ============================================================
# 3. Helper: Laplacian and Biharmonic Operator via Autograd
# ============================================================

def laplacian(u, x, y):
    u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_y = torch.autograd.grad(u, y, grad_outputs=torch.ones_like(u), create_graph=True)[0]

    u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
    u_yy = torch.autograd.grad(u_y, y, grad_outputs=torch.ones_like(u_y), create_graph=True)[0]

    return u_xx + u_yy

def biharmonic(u, x, y):
    lap_u = laplacian(u, x, y)
    return laplacian(lap_u, x, y)

# ===========================================
# 2. Neural Network Architecture (with uniform weight & bias init)
# ===========================================

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
    
# ===========================================
# 4. Training Data (Interior + Boundary)
# ===========================================

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

# ========================================================
# Compute analytic BIHARMONIC of exact u(x,y) at interior
# ========================================================

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
f_int = f_int.detach()   # target, no grad

# ===========================================
# 5. Model, Optimizer, Loss Function (Improved)
# ===========================================

model = PINN_Biharmonic(layers=(128,128,128,128)).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
mse = nn.MSELoss()

λ_int = 1.0        # PDE weight
λ_bc = 2.0         # boundary weight
n_epochs = 10000
print_every = 500

loss_history = []
best_loss = 1e30
best_epoch = -1
start_time = time()

# ===========================================
# 6. Training Loop
# ===========================================

print("\nTraining Started...\n")

for epoch in range(1, n_epochs+1):

    optimizer.zero_grad()

    # --------------------------------------------------------
    #  Interior PDE: Δ²u = f
    # --------------------------------------------------------
    xt = x_int_t.clone().requires_grad_(True)
    yt = y_int_t.clone().requires_grad_(True)

    u_pred = model(xt, yt)

    # --- Laplacian ---
    u_x  = torch.autograd.grad(u_pred, xt, grad_outputs=torch.ones_like(u_pred), create_graph=True)[0]
    u_y  = torch.autograd.grad(u_pred, yt, grad_outputs=torch.ones_like(u_pred), create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x, xt, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
    u_yy = torch.autograd.grad(u_y, yt, grad_outputs=torch.ones_like(u_y), create_graph=True)[0]
    lap_u = u_xx + u_yy

    # --- Biharmonic Δ(Δu) ---
    lap_u_x  = torch.autograd.grad(lap_u, xt, grad_outputs=torch.ones_like(lap_u), create_graph=True)[0]
    lap_u_y  = torch.autograd.grad(lap_u, yt, grad_outputs=torch.ones_like(lap_u), create_graph=True)[0]
    laplap_u = (
        torch.autograd.grad(lap_u_x, xt, grad_outputs=torch.ones_like(lap_u_x), create_graph=True)[0] +
        torch.autograd.grad(lap_u_y, yt, grad_outputs=torch.ones_like(lap_u_y), create_graph=True)[0]
    )

    loss_pde = mse(laplap_u, f_int)


    # --------------------------------------------------------
    # Boundary Condition: simply supported
    #   u  = u_bc
    #   Δu = lap_u_bc
    # --------------------------------------------------------
    xb = x_bc_t.clone().requires_grad_(True)
    yb = y_bc_t.clone().requires_grad_(True)

    u_pred_b = model(xb, yb)

    # Laplacian for BC
    ux_b  = torch.autograd.grad(u_pred_b, xb, grad_outputs=torch.ones_like(u_pred_b), create_graph=True)[0]
    uy_b  = torch.autograd.grad(u_pred_b, yb, grad_outputs=torch.ones_like(u_pred_b), create_graph=True)[0]
    uxx_b = torch.autograd.grad(ux_b, xb, grad_outputs=torch.ones_like(ux_b), create_graph=True)[0]
    uyy_b = torch.autograd.grad(uy_b, yb, grad_outputs=torch.ones_like(uy_b), create_graph=True)[0]
    lap_u_pred_b = uxx_b + uyy_b

    loss_bc = mse(u_pred_b, u_bc) + mse(lap_u_pred_b, lap_bc)

    # --------------------------------------------------------
    # TOTAL LOSS
    # --------------------------------------------------------
    loss = λ_int * loss_pde + λ_bc * loss_bc
    loss.backward()

    # gradient clipping (stable)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

    loss_history.append(loss.item())

    if loss.item() < best_loss:
        best_loss = loss.item()
        best_epoch = epoch

    if epoch % print_every == 0 or epoch == 1:
        lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch:5d} | Loss={loss.item():.3e} | "
            f"PDE={loss_pde.item():.3e} | BC={loss_bc.item():.3e} | "
            f"BestEpoch={best_epoch} | BestLoss={best_loss:.3e}")



end_time = time()

print("\nTraining completed.")
print(f"Best Loss = {best_loss:.3e} at epoch {best_epoch}")
print(f"Total Training Time: {end_time - start_time:.2f} seconds")

# ===========================================
# 7. Error Computation (L2, H1, H2)  — FIXED
# ===========================================
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

print("\n====================== ERRORS ======================")
print(f"Relative L2 error  = {L2_rel:.3e}")
print(f"Relative H1 error  = {H1_rel:.3e}")
print(f"Relative H2 error  = {H2_rel:.3e}")
print("====================================================")

u_ex_test = u_ex_np
u_pred_test = u_pred_np

# ===========================================
# 8. Visualization & Report Summary
# ===========================================
from matplotlib.ticker import MaxNLocator

plt.figure(figsize=(6,4))
plt.semilogy(loss_history)
plt.title("Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss (log)")
plt.grid()
plt.show()

fig, ax = plt.subplots(1,3,figsize=(16,4))
im = ax[0].imshow(u_ex_np,origin='lower'); ax[0].set_title("Exact"); plt.colorbar(im,ax=ax[0])
im = ax[1].imshow(u_pred_np,origin='lower'); ax[1].set_title("Predicted"); plt.colorbar(im,ax=ax[1])
im = ax[2].imshow(np.abs(u_pred_np-u_ex_np),origin='lower',cmap='inferno'); ax[2].set_title("Abs Error"); plt.colorbar(im,ax=ax[2])
plt.show()

print("===========================================")
print(" BIHARMONIC PINN — FINAL RESULTS")
print("===========================================")
print(f"Best loss           : {best_loss:.3e}")
print(f"L2 relative error   : {L2_rel:.3e}")
print(f"H1 relative error   : {H1_rel:.3e}")
print(f"H2 relative error   : {H2_rel:.3e}")
print("===========================================")
