import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import matplotlib

matplotlib.colormaps.get_cmap

def gradient_of(xs: np.ndarray[np.ndarray[float]], f: callable) -> list[list[float]]:
    return list(map(lambda x: list(map(f, x)), xs))

# adaptado 'rainbow' de matplotlib
colors = [
    (255/255, 255/255, 255/255),   # Violeta
    (75/255, 0/255, 130/255),    # Indigo
    (0/255, 0/255, 255/255),     # Azul
    (0/255, 255/255, 0/255),     # Verde
    #(255/255, 200/255, 0/255),   # Amarillo
    (255/255, 127/255, 0/255),   # Naranja
    (255/255, 0/255, 0/255)      # Rojo
]

rainbow_cmap = LinearSegmentedColormap.from_list("custom_rainbow", colors, N=256)