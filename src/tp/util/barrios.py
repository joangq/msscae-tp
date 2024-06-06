from matplotlib import pyplot as plt
import numpy as np

def generar_cuatro_cuadrantes(path: str, L: int = 50) -> None:
    with open(path, 'w') as file:
        for i in range(L):
            x = 0
            for j in range(L):

                if i <= L // 2:
                    if j < L // 2:
                        x = 1
                    else:
                        x = 2
                else:
                    if j < L // 2:
                        x = 0
                    else:
                        x = 3

                file.write(f'{x} ')
            file.write('\n')

def cargar_barrios(path: str) -> np.ndarray:
    with open(path) as f:  # inicializa con el txt que contiene una grilla.
        lines = f.readlines()
        lines
        grid = []
        for line in lines:
            grid.append([int(val) for val in line.strip().split()])

    return np.array(grid)


def mostrar_barrios(matriz: np.ndarray):
    fig = plt.figure(figsize=(6,4))
    plt.imshow(cargar_barrios('./tp/mapas/mapa.txt'), cmap = 'rainbow')
    return plt