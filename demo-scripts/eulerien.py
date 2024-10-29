import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Parameters
gamma = 1.4 # ratio of specific heats
Mach = float(input("Enter the Mach number (0-3): "))
U = 1.0 # freestream velocity magnitude
p0 = 1.0 # freestream pressure
rho0 = p0/(U*U) # freestream density

# Cylinder geometry
R = 0.2 # cylinder radius
x0, y0 = -0.5, 0.0 # cylinder center

# Grid
N = 201 # number of grid points in each direction
dx = 1.0/N # grid spacing
dy = dx # grid spacing
x = np.linspace(-0.5, 0.5, N) - x0 # shifted grid
y = np.linspace(-0.5, 0.5, N) - y0 # shifted grid
X, Y = np.meshgrid(x, y) # grid

# Initial conditions
Vx = np.zeros_like(X)
Vy = np.zeros_like(Y)
P = np.ones_like(X)
Rho = np.ones_like(X)

# Boundary conditions
Vx[0, :] = U*Mach
Vx[-1, :] = U*Mach
Vy[:, 0] = 0.0
Vy[:, -1] = 0.0

# Time-stepping
dt = 0.001 # time step size
t_end = 1.0 # end time
t = 0.0 # current time

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

while t < t_end:
    # Update velocity
    Vx_new = Vx - dt*(Vx*np.gradient(Vx, dx, axis=0) + Vy*np.gradient(Vx, dy, axis=1) + np.gradient(P, X))/Rho
    Vy_new = Vy - dt*(Vx*np.gradient(Vy, dx, axis=0) + Vy*np.gradient(Vy, dy, axis=1) + np.gradient(P, Y))/Rho

    # Update density and pressure
    Rho_new = Rho - dt*(np.gradient(Vx*Rho, dx, axis=0) + np.gradient(Vy*Rho, dy, axis=1))
    P_new = P*(Rho_new/Rho)**gamma

    # Apply boundary conditions
    Vx_new[0, :] = U*Mach
    Vx_new[-1, :] = U*Mach
    Vy_new[:, 0] = 0.0
    Vy_new[:, -1] = 0.0
    Rho_new[0, :] = Rho0
    Rho_new[-1, :] = Rho0
    P_new[0, :] = P0
    P_new[-1, :] = P0

    # Update variables
    Vx, Vy, Rho, P = Vx_new, Vy_new, Rho_new, P_new
    t += dt

    # Plot
    ax.clear()
    ax.plot_surface(X, Y, P, cmap='coolwarm')
    plt.title('Pressure at time t = {:.2f}'.format(t))
    plt.pause(0.01)

# Calculate Mach number
Mach_num = np.sqrt(Vx*Vx + Vy*Vy)/U

# Plot Mach number contours
fig, ax = plt.subplots()
CS = ax.contourf(X, Y, Mach_num, levels=np.linspace(0, 3, 11), cmap='inferno')
ax.set_xlim(-0.5, 0.5)
ax.set_ylim(-0.5, 0.5)