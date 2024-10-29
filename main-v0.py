import numpy as np
from matplotlib import pyplot

plot_every = 50

def distance(x1, y1, x2, y2):
    return np.sqrt((x2-x1)**2 + (y2-y1)**2)

def main():

    # constants
    Nx=400
    Ny=100
    tau = 0.53 # kinematic viscosity
    Nt= 300 # iteration count

    #### lattice structure speed and mass ###

    NL=9
    csx= np.array([0, 0, 1, 1, 1, 0, -1, -1, -1])
    csy= np.array([0, 1, 1, 0, -1, -1, -1, 0, 1])
    weights = np.array([4/9, 1/9, 1/36, 1/9, 1/36, 1/9, 1/36, 1/9, 1/36])

    # initial conditions set up so that it moves right to left
    F=np.ones((Ny, Nx, NL)) + 0.01 * np.random.randn(Ny, Nx, NL)
    F[:, :, 3]= 2.3
    # defining walls
    cylinder = np.full((Ny, Nx), False)

    for y in range(0, Ny):
        for x in range(0, Nx):
            if (distance(Nx//4, Ny//2, x, y)<13):
                cylinder[y][x] = True

    ## main calculation loop
    for it in range(Nt):
        print(it)
        ## adding reflective boundary walls property to walls

        F[:, -1, [6, 7, 8]]=F[:, -2, [6, 7, 8]]
        F[:, 0, [2, 3, 4]]=F[:, 1, [2, 3, 4]]

        for i, cx, cy in zip(range(NL), csx, csy):
            F[:, :, i] = np.roll(F[:, :, i], cx, axis = 1)
            F[:, :, i] = np.roll(F[:, :, i], cy, axis = 0)

            bndryF= F[cylinder, :]
            bndryF=bndryF[:, [0 , 5, 6, 7, 8, 1, 2, 3, 4]] # reflecting back the wall speed in the opposite direction of the cylinder

            # finding density and u velocity
            ### flow's constants definition ###

            rho = np.sum(F, 2)
            ux = np.sum(F * csx, 2) / rho
            uy = np.sum(F * csy, 2) / rho

            F[cylinder, :] = bndryF
            ux[cylinder] = 0
            uy[cylinder] = 0

            ## collision detection loop

            Feq = np.zeros(F.shape)
            for i, cx, cy, w in zip(range(NL), csx, csy, weights):
                Feq[:, :, i] = rho * w * (
                    1+ 3 * (cx*ux + cy*uy) + 9 * (cx*ux + cy*uy)**2 / 2 - 3 * (ux**2 + uy**2)/2
                )

            F= F + -(1/tau)*(F-Feq)

            if (it%plot_every == 0):
                pyplot.imshow(np.sqrt(ux**2 + uy**2))
                pyplot.pause(.0001)
                pyplot.cla()

if __name__ == "__main__":
    main()