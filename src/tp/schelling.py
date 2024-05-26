import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from .util.types import Lattice
from .util.interfaces import mercado_inmobiliario_interface
from typing import Optional
from numpy.random import Generator as RandomNumberGenerator

def precio_propiedad(C_propio: float, C_distinto: float) -> float:
    A = 1/16
    B = 0.5

    return A*(C_propio - C_distinto) + B  # Ecuación 2.6 -  p.86

def utilidad(K: float, precio: float, alpha: float=0.5) -> float:
    return (K**(alpha)) * precio**(1-alpha) # Ecuación 2.5 -  p.85

class mercado_inmobiliario(mercado_inmobiliario_interface):
    rng: RandomNumberGenerator # generador de numeros aleatorios

    L: int # tamaño de la cuadricula (lattice)
    configuracion: Lattice[int] # cuadricula

    alpha: float # umbral de tolerancia
    K: Lattice[float] # capital de cada individuo
    U: Lattice[float] # utilidad de cada individuo


    def __init__(self, 
                 L: int, 
                 configuracion: Optional[Lattice[int]] = None,
                 alpha: float = 0.5,
                 rng: Optional[RandomNumberGenerator] = None):
        
        self.L = L
        self.alpha = alpha

        if not rng:
            rng = np.random.default_rng()

        self.rng = rng

        if not configuracion:
            # reemplaza np.random.randint
            configuracion = rng.integers(2, size=(self.L, self.L))
        
        self.configuracion = configuracion

        self.K = np.ones((self.L, self.L)) # Capital inicial de cada individuo, K=1

        self.U = np.zeros((self.L, self.L)) # Utilidad de cada individuo
        for i in range(self.L):
            for j in range(self.L):
                C_p, C_dist = self._num_vecinos(i, j, i, j)
                precio = precio_propiedad(C_p, C_dist)
                self.U[i, j] = utilidad(self.K[i, j], precio, self.alpha) # Utilidad inicial
        
    def utilidad_media(self) -> float:
        return np.mean(self.U.flatten())

    def capital_medio(self) -> float:
        return np.mean(self.K.flatten())

    def _num_vecinos(self, i_a, j_a, i, j) -> tuple[int, int]:
        """
        Calcula el número de vecinos del mismo tipo (C_p) y del tipo opuesto (C_dist) de un agente en una posición dada.

        Parámetros:
            i_a, j_a (int): Coordenadas del agente.
            i, j (int): Coordenadas de la posición de interés.

        Returns:
            tuple: Número de vecinos del mismo tipo (C_p) y del tipo opuesto (C_dist).
        """
        agente = self.configuracion[i_a, j_a]

        C_propio = 1  # Cuenta a sí mismo como un vecino - p. 86
        C_distinto = 0

        indices_vecinos = [(i    , j - 1), (i    , j + 1), 
                           (i - 1, j    ), (i + 1, j    ),
                           (i + 1, j - 1), (i + 1, j + 1), 
                           (i - 1, j - 1), (i - 1, j + 1)]

        for m, n in indices_vecinos:
            i_vecino = m % self.L
            j_vecino = n % self.L
            vecino_mn = self.configuracion[i_vecino, j_vecino]

            if agente == vecino_mn:
                C_propio += 1
            else:
                C_distinto += 1

        return C_propio, C_distinto

    def proponer_intercambio(self) -> None:
        """
        Propone y realiza un intercambio entre dos agentes de la cuadrícula si mejora la utilidad de ambos.
        Muta:
        - self.configuracion
        - self.K
        - self.U
        """
        # Seleccionar dos posiciones aleatorias en la cuadrícula
        i1, j1, i2, j2 = self.rng.integers(0, self.L, size=4)

        agente_1, agente_2 = self.configuracion[i1, j1], self.configuracion[i2, j2]
        K_1, K_2 = self.K[i1, j1], self.K[i2, j2]

        C_p_1_nuevo, C_dist_1_nuevo = self._num_vecinos(i1, j1, i2, j2)
        C_p_2_nuevo, C_dist_2_nuevo = self._num_vecinos(i2, j2, i1, j1)

        precio_1_nuevo, precio_2_nuevo = precio_propiedad(C_p_1_nuevo, C_dist_1_nuevo), precio_propiedad(C_p_2_nuevo, C_dist_2_nuevo)
        # delta_p = precio_2_nuevo - precio_1_nuevo
        p_promedio = (precio_2_nuevo + precio_1_nuevo) / 2

        # Verificar si ambos agentes tienen suficiente riqueza para la transacción
        #if np.min([K_1, K_2]) > np.abs(delta_p):
        if (precio_1_nuevo - p_promedio - K_1 < 0) and (precio_2_nuevo - p_promedio - K_2 < 0):

            # K_2_nuevo = K_2 - delta_p
            # K_1_nuevo = K_1 + delta_p

            K_1_nuevo = K_1 + precio_2_nuevo - p_promedio  # Cálculo del capital potencial del agente 1
            K_2_nuevo = K_2 + precio_1_nuevo - p_promedio  # Cálculo del capital potencial del agente 2

            
            utilidad_1_nueva = utilidad(K_1_nuevo, precio_1_nuevo, self.alpha)
            utilidad_2_nueva = utilidad(K_2_nuevo, precio_2_nuevo, self.alpha)
            
            delta_u_1 = utilidad_1_nueva - self.U[i1, j1]
            delta_u_2 = utilidad_2_nueva - self.U[i2, j2]

            # Verificar si la utilidad mejora para ambos agentes
            
            if delta_u_1 > 0 and delta_u_2 > 0:  # Si mejora, se realiza el intercambio
                
                # Actualizar la configuración, riqueza y utilidad después del intercambio
                
                self.configuracion[i1, j1] = agente_2
                self.configuracion[i2, j2] = agente_1
                
                self.K[i1, j1] = K_2_nuevo
                self.K[i2, j2] = K_1_nuevo
                
                self.U[i1, j1] = utilidad_2_nueva
                self.U[i2, j2] = utilidad_1_nueva
    
    def ronda_intercambio(self) -> None:
        N_intercambios = self.L**2
        for _ in range(N_intercambios):
            self.proponer_intercambio()

    def lattice(self) -> Lattice[float]:
        lattice = np.copy(self.configuracion)

        for i in range(self.L):
            for j in range(self.L):
                U_ij = self.U[i, j]

                # Marcar agentes insatisfechos dependiendo del tipo de agente
                if U_ij < 0.85:
                    lattice[i, j] = 4 if self.configuracion[i, j] == 1 else 5
        
        return lattice

    def lattice_plot(self):
        """
        Crea un gráfico de malla basado en los niveles de satisfacción de los agentes.

        """
        lattice = self.lattice()

        fig = plt.figure(figsize=(6,4))
        plt.imshow(lattice, cmap = 'inferno')
        plt.colorbar()
        plt.title("Sistema L = {} y alpha = {} \n (sitios naranjas y amarillos corresponden a agentes insatisfechos)".format(self.L, self.alpha))
        plt.show()

    def generar_animacion(self, frames) -> FuncAnimation:
        fig, ax = plt.subplots()
        img = ax.imshow(self.lattice(), cmap='inferno', interpolation='nearest')

        def actualizar(i):
            self.ronda_intercambio()
            img.set_array(self.lattice())
            return img,

        animacion = FuncAnimation(fig, actualizar, frames=frames, interval=200, blit=True)
        plt.title("Evolución del sistema L = {} y alpha = {} \n (sitios amarillos y naranjas corresponden a agentes insatisfechos)".format(self.L, self.alpha))
        plt.close(fig)
        
        return animacion