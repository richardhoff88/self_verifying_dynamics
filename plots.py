import numpy as np
from math import sin, cos
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

a1 = a2 = 1.0   # base dynamics parameters for link 1
b1 = b2 = 1.0   # base dynamics parameters for link 2
alpha = 0.2     
beta  = 1.0     
gamma = 1.0     
x_hat = np.array([0.0, 0.0])   # origin

def f(x):
    """Base dynamics for each link (nonlinear spring-damper)."""
    x1, x2 = x
    f1 = -a1 * sin(x1) - a2 * x1
    f2 = -b1 * sin(x2) - b2 * x2
    return np.array([f1, f2])

def dynamics(t, state, alpha_val=alpha, beta_val=beta, gamma_val=gamma):
    x1, x2, z = state
    x_vec = np.array([x1, x2])
    fx = f(x_vec)
    # Gradient of E w.rt x: z * (x - x_hat)
    gradE_x = z * (x_vec - x_hat)
    # Compute derivatives
    x_dot = fx - alpha_val * gradE_x
    z_dot = beta_val * (fx[0]**2 + fx[1]**2) - gamma_val * z
    return [x_dot[0], x_dot[1], z_dot]

# Initial conditions
x0 = np.array([0.5, -0.2])  
z0 = 0.5                  
y0 = [x0[0], x0[1], z0]

# Solving system with Runge-Kutta (solve_ivp) for t in [0, 1]
t_end = 1.0
t_points = np.linspace(0, t_end, 1001) 
sol = solve_ivp(dynamics, [0, t_end], y0, t_eval=t_points, args=(alpha, beta, gamma))
x1_true = sol.y[0];  x2_true = sol.y[1];  z_true = sol.y[2]

#  Picard iterations on [0, 1]
max_iter = 3
x1_iter = np.tile(x0[0], t_points.shape)
x2_iter = np.tile(x0[1], t_points.shape)
z_iter  = np.tile(z0,    t_points.shape)

iters = {1: None, 2: None, 3: None}

dt = t_points[1] - t_points[0]
for k in range(1, max_iter+1):
    x1_next = np.zeros_like(x1_iter)
    x2_next = np.zeros_like(x2_iter)
    z_next  = np.zeros_like(z_iter)
    x1_next[0] = x0[0];  x2_next[0] = x0[1];  z_next[0] = z0  # initial state
    for i in range(len(t_points)-1):
        xi = np.array([x1_iter[i], x2_iter[i]])
        zi = z_iter[i]
        fx_i = f(xi)
        x_dot_i = fx_i - alpha * zi * (xi - x_hat)
        z_dot_i = beta * (fx_i[0]**2 + fx_i[1]**2) - gamma * zi
        # Euler step
        x1_next[i+1] = x1_next[i] + x_dot_i[0] * dt
        x2_next[i+1] = x2_next[i] + x_dot_i[1] * dt
        z_next[i+1]  = z_next[i]  + z_dot_i * dt
    iters[k] = (x1_next, x2_next, z_next)
    x1_iter, x2_iter, z_iter = x1_next, x2_next, z_next

final_error = np.linalg.norm([x1_iter[-1]-x1_true[-1],
                              x2_iter[-1]-x2_true[-1],
                              z_iter[-1]-z_true[-1]])
print(f"Final error after 3 iterations: {final_error:.4f}")

alpha2 = 0.9
sol2 = solve_ivp(dynamics, [0, 6.0], y0, t_eval=np.linspace(0, 6.0, 601),
                 args=(alpha2, beta, gamma))
sol1 = solve_ivp(dynamics, [0, 6.0], y0, t_eval=np.linspace(0, 6.0, 601),
                 args=(alpha, beta, gamma))

t_long = sol1.t
x1_alpha1, x2_alpha1, z_alpha1 = sol1.y[0], sol1.y[1], sol1.y[2]
x1_alpha2, x2_alpha2, z_alpha2 = sol2.y[0], sol2.y[1], sol2.y[2]


plt.figure(figsize=(6,8))
plt.subplot(3,1,1); plt.plot(t_points, x1_true, 'k', label="True");
plt.plot(t_points, iters[1][0], 'r-', label="Picard k=1");
plt.plot(t_points, iters[2][0], 'b-', label="Picard k=2");
plt.plot(t_points, iters[3][0], 'g-', label="Picard k=3");
plt.ylabel('x1'); plt.legend()
plt.subplot(3,1,2); plt.plot(t_points, x2_true, 'k');
plt.plot(t_points, iters[1][1], 'r-'); plt.plot(t_points, iters[2][1], 'b-');
plt.plot(t_points, iters[3][1], 'g-');
plt.ylabel('x2');
plt.subplot(3,1,3); plt.plot(t_points, z_true, 'k');
plt.plot(t_points, iters[1][2], 'r-'); plt.plot(t_points, iters[2][2], 'b-');
plt.plot(t_points, iters[3][2], 'g-');
plt.ylabel('z'); plt.xlabel('Time (s)');
plt.suptitle('Picard Iterations vs True Solution');
plt.tight_layout()
plt.show()

plt.figure(figsize=(6,6))
plt.subplot(3,1,1); plt.plot(t_long, x1_alpha1, 'b', label="alpha=0.2");
plt.plot(t_long, x1_alpha2, 'm-', label="alpha=0.9");
plt.ylabel('x1'); plt.legend()
plt.subplot(3,1,2); plt.plot(t_long, x2_alpha1, 'b'); plt.plot(t_long, x2_alpha2, 'm-');
plt.ylabel('x2');
plt.subplot(3,1,3); plt.plot(t_long, z_alpha1, 'b'); plt.plot(t_long, z_alpha2, 'm-');
plt.ylabel('z'); plt.xlabel('Time (s)');
plt.suptitle('Solutions for Perturbed alpha Values');
plt.tight_layout()
plt.show()
