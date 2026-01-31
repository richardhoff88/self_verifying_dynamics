import numpy as np
from math import sin, cos, exp
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from numpy.linalg import norm


alpha = 0.5
beta = 1.0
gamma = 1.0
x_hat = np.array([0.0, 0.0])
Z_max = 2.0  # max value of z on the domain

# Base dynamics
def f(x):
    x1, x2 = x
    return np.array([-sin(x1) - x1, -sin(x2) - x2])

def Df(x):
    x1, x2 = x
    return np.array([[-cos(x1) - 1, 0], [0, -cos(x2) - 1]])


grid_points = 100
x1_vals = np.linspace(-1, 1, grid_points)
x2_vals = np.linspace(-1, 1, grid_points)

M1 = 0  # sup ||Df(x)||
M2 = 0  # sup |f^T Df f|
R = 0   # sup ||x - x_hat||

for x1 in x1_vals:
    for x2 in x2_vals:
        x = np.array([x1, x2])
        df = Df(x)
        fx = f(x)
        M1 = max(M1, norm(df, 2))
        M2 = max(M2, abs(fx @ df @ fx))
        R = max(R, norm(x - x_hat, 2))

# Compute Lipschitz constant
L = M1 + alpha * Z_max + 2 * beta * M2 + alpha * R + gamma
print(f"Computed Lipschitz constant L ≈ {L:.4f}")


# Nominal initial condition
x0 = np.array([0.5, -0.2])
z0 = 0.5
y0 = [x0[0], x0[1], z0]

delta = 0.01
x0_perturbed = np.array([x0[0] + delta, x0[1]])
z0_perturbed = z0
y0_perturbed = [x0_perturbed[0], x0_perturbed[1], z0_perturbed]

def dynamics(t, state, alpha_val=alpha, beta_val=beta, gamma_val=gamma):
    x1, x2, z = state
    x_vec = np.array([x1, x2])
    fx = f(x_vec)
    gradE_x = z * (x_vec - x_hat)
    x_dot = fx - alpha_val * gradE_x
    z_dot = beta_val * np.dot(fx, fx) - gamma_val * z
    return [x_dot[0], x_dot[1], z_dot]

t_end = 2.0
t_points = np.linspace(0, t_end, 1000)

sol_nominal = solve_ivp(dynamics, [0, t_end], y0, t_eval=t_points)
sol_perturbed = solve_ivp(dynamics, [0, t_end], y0_perturbed, t_eval=t_points)

x_nominal = sol_nominal.y[:2, :]
z_nominal = sol_nominal.y[2, :]
x_perturbed = sol_perturbed.y[:2, :]
z_perturbed = sol_perturbed.y[2, :]

epsilon_0 = np.linalg.norm(np.array(y0) - np.array(y0_perturbed))
mu_t = epsilon_0 * np.exp(L * t_points)
errors = np.sqrt(np.sum((x_nominal - x_perturbed) ** 2, axis=0) + (z_nominal - z_perturbed) ** 2)

plt.figure(figsize=(6, 4))
plt.plot(t_points, errors, label='Actual error $\\|x(t)-\\tilde{x}(t)\\|$')
plt.plot(t_points, mu_t, '--', label='Bound $\\mu(t) = \\epsilon_0 e^{Lt}$')
plt.xlabel('Time (s)')
plt.ylabel('State difference')
plt.title('Continuous Dependence: Error vs Exponential Bound')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
