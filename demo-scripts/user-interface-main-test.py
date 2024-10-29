
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

N_POINTS= int(input('kaç noktada görselleştirme yapılsın?')) # miyawki
DOMAIN_SIZE = 1.0
N_ITERATIONS = int(input('iterasyon say giriiniz'))
TIME_STEP_LENGTH = 0.001
KINEMATIC_VISCOSITY = float(input('kinematik viscosite değeri nedir'))
DENSITY = float(input('yoğunluk nedir?'))
HORIZONTAL_VELOCITY_TOP = 1.0

N_PRESSURE_POISSON_ITERATIONS = 50
STABILITY_SAFETY_FACTOR = 0.5

def main():
    element_length = DOMAIN_SIZE / (N_POINTS - 1)
    x = np.linspace(0.0, DOMAIN_SIZE, N_POINTS)
    y = np.linspace(0.0, DOMAIN_SIZE, N_POINTS)
    
    X, Y= np.meshgrid(x, y)
    
    u_prev = np.zeros_like(X) 
    vars_prev = np.zeros_like(X) 