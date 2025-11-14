import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from time import time

# ===========================================
# 1. Define Problem (P3: Simply Supported BCs)
# ===========================================

# Domain: Ω = (0, 1)^2
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(0)
np.random.seed(0)

# Exact solution
def u_exact(x, y):
    return 0.5 / np.pi**2 * np.sin(np.pi * x) * np.sin(np.pi * y)
# source 
def f_source(x, y):
    # f = Δ²u for u = (1 / (2π²)) sin(πx) sin(πy)
    # Δu = -2π² * (1 / (2π²)) sin(πx) sin(πy) = -sin(πx) sin(πy)
    # Δ²u = Δ(Δu) = Δ(-sin(πx) sin(πy)) = 2π² sin(πx) sin(πy)
    return 2 * np.pi**2 * np.sin(np.pi * x) * np.sin(np.pi * y)

# Boundary data
def g1(x, y):  # u = exact
    return u_exact(x, y)
def g2(x, y):  # Δu = exact Laplacian
    return -2 * np.sin(np.pi * x) * np.sin(np.pi * y)

# ===========================================
# 2. Neural Network Architecture (with uniform weight & bias init)
# ===========================================

class PINN_Biharmonic(nn.Module):
    def __init__(self, layers=(50,50,50), activation=nn.Tanh()):
        super().__init__()
        net = []
        in_dim = 2

        for h in layers:
            net.append(nn.Linear(in_dim, h))
            net.append(activation)
            in_dim = h

        net.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*net)

        # ---- Custom Initialization ----
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # Uniform Xavier initialization for weights
            nn.init.xavier_uniform_(m.weight)
            # Small random bias
            nn.init.uniform_(m.bias, -0.1, 0.1)

    def forward(self, x, y):
        inp = torch.cat([x, y], dim=1)
        return self.net(inp)

# ===========================================
# 3. Helper functions (Automatic Differentiation)
# ===========================================

def laplacian(u, x, y):
    grads = torch.autograd.grad(u, [x, y], grad_outputs=torch.ones_like(u), create_graph=True)
    u_x, u_y = grads
    u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
    u_yy = torch.autograd.grad(u_y, y, grad_outputs=torch.ones_like(u_y), create_graph=True)[0]
    return u_xx + u_yy

def biharmonic(u, x, y):
    Δu = laplacian(u, x, y)
    return laplacian(Δu, x, y)

# ===========================================
# 4. Training Data (Interior + Boundary)
# ===========================================

N_int = 20000
N_bc = 10000

x_int = np.random.rand(N_int, 1)
y_int = np.random.rand(N_int, 1)

# Proper boundary sampling
x_bottom = np.linspace(0, 1, N_bc//4).reshape(-1, 1)
y_bottom = np.zeros_like(x_bottom)

x_top = np.linspace(0, 1, N_bc//4).reshape(-1, 1)
y_top = np.ones_like(x_top)

y_left = np.linspace(0, 1, N_bc//4).reshape(-1, 1)
x_left = np.zeros_like(y_left)

y_right = np.linspace(0, 1, N_bc//4).reshape(-1, 1)
x_right = np.ones_like(y_right)

x_bc = np.vstack([x_bottom, x_top, x_left, x_right])
y_bc = np.vstack([y_bottom, y_top, y_left, y_right])

# Convert to tensors
x_int_t = torch.tensor(x_int, dtype=torch.float32, requires_grad=True).to(device)
y_int_t = torch.tensor(y_int, dtype=torch.float32, requires_grad=True).to(device)
x_bc_t = torch.tensor(x_bc, dtype=torch.float32, requires_grad=True).to(device)
y_bc_t = torch.tensor(y_bc, dtype=torch.float32, requires_grad=True).to(device)

f_int = torch.tensor(f_source(x_int, y_int), dtype=torch.float32).to(device)
u_bc = torch.tensor(g1(x_bc, y_bc), dtype=torch.float32).to(device)
Δu_bc = torch.tensor(g2(x_bc, y_bc), dtype=torch.float32).to(device)

# ===========================================
# 5. Model, Optimizer, Loss Function (Improved)
# ===========================================

# Larger and deeper network for better representation of 4th-order PDEs
model = PINN_Biharmonic(layers=(128, 128, 128, 128)).to(device)

# Adam optimizer (with decaying learning rate for stable convergence)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# Mean Squared Error loss
mse = nn.MSELoss()

# Weighting factors for PDE and boundary losses
λ_int, λ_bc = 1 ,1   # ↑ Slightly higher boundary weight improves H1/H2 accuracy

# Total epochs (you can go 10000–15000 for better convergence)
n_epochs = 15000
print_every = 500        # print progress more frequently to monitor convergence
loss_history = []

# Record start time for performance measurement
start_time = time()

# ===========================================
# 6. Training Loop
# ===========================================

best_loss = float('inf')
best_epoch = 0

for epoch in range(1, n_epochs + 1):
    optimizer.zero_grad()

    # Interior residual
    u_pred = model(x_int_t, y_int_t)
    Δ2u_pred = biharmonic(u_pred, x_int_t, y_int_t)
    loss_int = mse(Δ2u_pred, f_int)

    # Boundary residuals (u and Δu)
    u_bc_pred = model(x_bc_t, y_bc_t)
    Δu_bc_pred = laplacian(u_bc_pred, x_bc_t, y_bc_t)
    loss_bc = mse(u_bc_pred, u_bc) + mse(Δu_bc_pred, Δu_bc)

    # Total loss
    loss = λ_int * loss_int + λ_bc * loss_bc
    loss.backward()
    optimizer.step()

    # Store current loss
    loss_history.append(loss.item())

    # Track minimum (best) loss
    if loss.item() < best_loss:
        best_loss = loss.item()
        best_epoch = epoch

    if epoch % print_every == 0:
        print(f"Epoch {epoch:5d}: Loss={loss.item():.4e} | L_int={loss_int.item():.4e} | L_bc={loss_bc.item():.4e} | Best={best_loss:.4e} @ {best_epoch}")

end_time = time()

# ===========================================
# 7. Error Computation (L2, H1, H2)
# ===========================================

model.eval()
nx, ny = 100, 100
xg, yg = np.meshgrid(np.linspace(0,1,nx), np.linspace(0,1,ny))

xt = torch.tensor(xg.reshape(-1,1), dtype=torch.float32, requires_grad=True).to(device)
yt = torch.tensor(yg.reshape(-1,1), dtype=torch.float32, requires_grad=True).to(device)

# Forward pass
u_pred = model(xt, yt)

# --- Compute derivatives using autograd ---
# First-order partials
u_x = torch.autograd.grad(u_pred, xt, grad_outputs=torch.ones_like(u_pred), create_graph=True)[0]
u_y = torch.autograd.grad(u_pred, yt, grad_outputs=torch.ones_like(u_pred), create_graph=True)[0]

# Second-order partials
u_xx = torch.autograd.grad(u_x, xt, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
u_yy = torch.autograd.grad(u_y, yt, grad_outputs=torch.ones_like(u_y), create_graph=True)[0]

# Optional cross term if needed
u_xy = torch.autograd.grad(u_x, yt, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]

# Convert all tensors to NumPy arrays (reshaped to grid)
u_pred_test = u_pred.detach().cpu().numpy().reshape(nx, ny)
u_ex_test = u_exact(xg, yg)

u_x_np  = u_x.detach().cpu().numpy().reshape(nx, ny)
u_y_np  = u_y.detach().cpu().numpy().reshape(nx, ny)
u_xx_np = u_xx.detach().cpu().numpy().reshape(nx, ny)
u_yy_np = u_yy.detach().cpu().numpy().reshape(nx, ny)
u_xy_np = u_xy.detach().cpu().numpy().reshape(nx, ny)

# --- Exact derivatives (from your exact u) ---
# IMPORTANT: Keep consistent with your u_exact definition
u_x_ex  = np.pi * np.cos(np.pi * xg) * np.sin(np.pi * yg) / (2*np.pi**2)
u_y_ex  = np.pi * np.sin(np.pi * xg) * np.cos(np.pi * yg) / (2*np.pi**2)
u_xx_ex = -np.pi**2 * np.sin(np.pi * xg) * np.sin(np.pi * yg) / (2*np.pi**2)
u_yy_ex = -np.pi**2 * np.sin(np.pi * xg) * np.sin(np.pi * yg) / (2*np.pi**2)
u_xy_ex = np.pi**2 * np.cos(np.pi * xg) * np.cos(np.pi * yg) / (2*np.pi**2)

# --- Relative errors ---
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

# --- Print results ---
print(f"\nRelative Errors:")
print(f"  L2  = {L2_err:.3e}")
print(f"  H1  = {H1_err:.3e}")
print(f"  H2  = {H2_err:.3e}")

# ===========================================
# 8. Visualization & Report Summary
# ===========================================
from matplotlib.ticker import MaxNLocator

# ---- 1. Plot Loss History ----
plt.figure(figsize=(6,4))
plt.semilogy(loss_history, color='purple', linewidth=2)
plt.xlabel("Epoch", fontsize=12)
plt.ylabel("Loss (log scale)", fontsize=12)
plt.title("Training Loss History", fontsize=14)
plt.grid(True, alpha=0.3)
plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
plt.tight_layout()
plt.show()

# ---- 2. Plot Exact, Predicted, and Error Fields ----
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

# ---- 3. Summary and Performance Metrics ----
print("="*60)
print("📘 BIHARMONIC PINN RESULTS SUMMARY")
print("="*60)
print(f"Neural Network Architecture : [2 -> 100 -> 100 -> 100 -> 1]")
print(f"Activation Function          : Tanh()")
print(f"Training Epochs              : {n_epochs}")
print(f"Optimizer                    : Adam (lr=1e-3)")
print(f"λ_int, λ_bc                  : {λ_int}, {λ_bc}")
print(f"Computation Time             : {end_time - start_time:.2f} seconds")
print("-"*60)
print(f"Best Loss Achieved = {best_loss:.4e} at Epoch {best_epoch}")
print(f"Relative L2 Error            : {L2_err:.3e}")
print(f"Relative H1 Error            : {H1_err:.3e}")
print(f"Relative H2 Error            : {H2_err:.3e}")
print("="*60)
