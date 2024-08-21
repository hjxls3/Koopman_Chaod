import numpy as np

from scipy.integrate import solve_ivp


def lorenz63_dynamics(t, x, sigma, rho, beta):
    xdot = np.zeros_like(x)
    xdot[0] = sigma * (x[1] - x[0])
    xdot[1] = x[0] * (rho - x[2]) - x[1]
    xdot[2] = x[0] * x[1] - beta * x[2]
    return xdot

# 随机初始值生成轨迹
def L63(sigma, rho, beta):
    x0 = np.random.uniform(low=-10, high=10, size=3)
    t_span = [0, 10000 * 0.01]
    t_eval = np.arange(0, 10000 * 0.01, 0.01)

    sol = solve_ivp(lorenz63_dynamics, t_span, x0, t_eval=t_eval, args=(sigma, rho, beta),
                    rtol=1e-12, atol=1e-12 * np.ones(3))
    return sol

# 固定初始值生成轨迹
def L63_with_init(x0, sigma, rho, beta):
    t_span = [0, 10000 * 0.01]
    t_eval = np.arange(0, 10000 * 0.01, 0.01)

    sol = solve_ivp(lorenz63_dynamics, t_span, x0, t_eval=t_eval, args=(sigma, rho, beta),
                    rtol=1e-12, atol=1e-12 * np.ones(3))
    return sol
