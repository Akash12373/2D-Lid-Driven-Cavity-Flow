# Navier–Stokes Physics-Informed Neural Network: 2D Lid-Driven Cavity Flow

A Physics-Informed Neural Network (PINN) implemented from scratch in PyTorch to solve the steady, incompressible 2D Navier–Stokes equations for classic lid-driven cavity flow ($Re = 100$)[cite: 1].

## 📂 Project Structure
- `cavity.ipynb`: Complete workflow containing automatic differentiation gradients, PDE residual loss computation, boundary sampling, model training, and velocity/pressure field post-processing[cite: 1].

## 🧠 Physics-Informed Formulation
Rather than relying purely on simulation data, the network takes spatial coordinates $(x, y)$ as inputs and directly outputs predicted flow variables $[u, v, p]^T$:
- **Continuity Equation:** Enforces mass conservation $\nabla \cdot \mathbf{u} = 0$:
  $$r_c = u_x + v_y = 0$$
[cite: 1]
- **Navier–Stokes (Momentum) Equations:** Steady 2D momentum conservation for $Re = 100$[cite: 1]:
  $$r_u = u u_x + v u_y + p_x - \frac{1}{Re}(u_{xx} + u_{yy}) = 0$$
[cite: 1]
  $$r_v = u v_x + v v_y + p_y - \frac{1}{Re}(v_{xx} + v_{yy}) = 0$$
[cite: 1]
- **Autograd Gradients:** Uses PyTorch's `torch.autograd.grad` with `create_graph=True` to calculate exact first- and second-order spatial derivatives without grid discretization[cite: 1].
- **Boundary Conditions & Gauge Fixing:** Enforces no-slip wall boundaries ($u=v=0$ on bottom, left, and right), lid velocity ($u=1, v=0$ on the top boundary), and pins reference pressure ($p=0$ at $(0.5, 0.5)$) to resolve the pressure gauge degree of freedom[cite: 1].

## 📊 Evaluation & Benchmarking
- **Reference Solution Comparison:** Compares predicted PINN velocity and pressure fields against established numerical benchmark data (e.g., Ghia et al., 1982) to validate centerline velocity profiles.
- **PDE Residual Analysis:** Evaluates the spatial distribution of the PDE residuals ($r_c, r_u, r_v$) across the domain to identify regions of high physical violation, particularly around the pressure singularities at the top lid corners[cite: 1].

## 📈 Training Sensitivity & Computational Cost
- **Loss Imbalance Study:** Analyzes the training sensitivity by monitoring the competing gradients between the boundary condition loss and the PDE residual loss. Investigates dynamic or static loss weighting strategies ($\lambda_{\text{BC}}, \lambda_p$) to prevent pathological convergence[cite: 1].
- **Computational Cost Profiling:** Measures the overhead introduced by building the higher-order computational graph (`create_graph=True`) during the forward pass[cite: 1]. Studies the impact of dynamic interior collocation resampling on memory transfer latency and overall training time[cite: 1].

## 🛠️ Usage
Clone the repository and open the notebook:
```bash
pip install torch matplotlib pandas
jupyter notebook cavity.ipynb
