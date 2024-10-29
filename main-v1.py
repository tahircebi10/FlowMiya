
import numpy as np
from matplotlib import pyplot
from numba import jit  #numpy işlemlerini hızlandırmak için kullanılır

plot_every = 50
@jit(nopython=True)
def distance (x1, y1, x2, y2):
    return np.sqrt((x2-x1)**2 + (y2-y1)**2)

def main():
    
    # sabitlerin tanımlanması     
    Nx=400 #görsel alanı
    Ny=100 #görsel alanı
    tau = 0.53 #kinematik viscositesı
    Nt= 3000 #iterasyon sayısı
    
    ####latice yapısı hızı ve ağırlıkları###
    
    NL=9
    csx= np.array([0, 0, 1, 1, 1, 0, -1, -1, -1])
    csy= np.array([0, 1, 1, 0,-1,-1, -1,  0,  1])
    weights = np.array([4/9, 1/9, 1/36, 1/9, 1/36, 1/9, 1/36, 1/9, 1/36])
    
    #başlangıç koşulları sağa dönecek şekilde her hücre ayarlanacak
    F=np.ones((Ny, Nx, NL)) + 0.01 * np.random.randn(Ny, Nx, NL)
    F[:, :, 3]= 2.3
    #silinidiri tanımlıyoruz
    cylinder = np.full((Ny, Nx), False)
    
    for y in range(0, Ny):
        for x in range(0, Nx):
            if (distance(Nx//4, Ny//2, x, y)<13):
                cylinder[y][x] = True
    
    ##ana hesaplama döngüsü
    for it in range(Nt):
        print(it)
        ##sınır duvarlarının reflektif yansıtıcılık
        F[:, -1, [6, 7, 8]]=F[:, -2, [6, 7, 8]]
        F[:, 0, [2, 3, 4]]=F[:, 1, [2, 3, 4]]
        
        for i, cx, cy in zip(range(NL), csx, csy):
            F[:, :, i] = np.roll(F[:, :, i], cx, axis = 1)
            F[:, :, i] = np.roll(F[:, :, i], cy, axis = 0)
            
            bndryF= F[cylinder, :]
            bndryF=bndryF[:, [0 , 5, 6, 7, 8, 1, 2, 3, 4]] #silindire çarpan geri dönen zıt yöndeki hızların yönü
            
            #yoğunluğu ve u hızını bulmak lazım
            ###akışın sabitlerin tanımlanması###
            
            rho = np.sum(F, 2)
            ux = np.sum(F * csx, 2) / rho
            uy = np.sum(F * csy, 2) / rho
            
            F[cylinder, :]= bndryF
            ux[cylinder] = 0
            uy[cylinder] = 0
            
            ##çarpışma döngüsünü kontrol ediyoruz
            
            Feq = np.zeros(F.shape)
            for i, cx, cy, w in zip(range(NL), csx, csy, weights):
                Feq[:, :, i] = rho * w * (
                    1+ 3 * (cx*ux + cy*uy) + 9 * (cx*ux + cy*uy)**2 / 2 - 3 * (ux**2 + uy**2)/2
                )
            
            F= F + -(1/tau)*(F-Feq)
            
            if (it%plot_every == 0):
                dfydx= ux[2: , 1:-1] - ux[0:-2, 1:-1] 
                dfxdy = uy[1:-1, 2:] - uy[1:-1, 0:-2]
                curl = dfydx - dfxdy
                pyplot.imshow(curl)
                #pyplot.imshow(np.sqrt(ux**2 + uy**2))
                pyplot.pause(.0001)
                pyplot.cla()
                        
if __name__ == "__main__":
    main()