import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from .util.types import Lattice
from .util.interfaces import mercado_inmobiliario_interface
from typing import Optional
from numpy.random import Generator as RandomNumberGenerator
from .util.barrios import Barrio, Mapa

def precio_propiedad(C_propio: float, C_distinto: float) -> float:
    A = 1/16
    B = 0.5

    return A*(C_propio - C_distinto) + B  # Ecuación 2.6 -  p.86

def utilidad(K: float, precio: float, alpha: float=0.5) -> float:
    with np.errstate(all='ignore'):
        return (K**(alpha)) * precio**(1-alpha) # Ecuación 2.5 -  p.85

def distance(i1, j1, i2, j2):
    return np.linalg.norm(np.array([i1, j1])-np.array([i2,j2]))    

class mercado_inmobiliario(mercado_inmobiliario_interface):
    rng: RandomNumberGenerator # generador de numeros aleatorios

    L: int # tamaño de la cuadricula (lattice)
    configuracion: Lattice[int] # cuadricula

    alpha: float # umbral de tolerancia
    K: Lattice[float] # capital de cada individuo
    U: Lattice[float] # utilidad de cada individuo

    precios_barrios: list[float]
    mapa_barrios: Lattice[int]

    def __init__(self, 
                 L: int,
                 # precios_barrios: list[float], #precios de mudarse a ese barrio
                 # precios_prop_barrios: list[float],
                 # configuracion_barrios: Lattice[int], # los valores están entre 0 y len(barrios)
                 mapa: Mapa,
                 configuracion: Optional[Lattice[int]] = None,
                 alpha: float = 0.5,
                 rango_de_vision: float = 1,
                 rng: Optional[RandomNumberGenerator] = None,
                 capital_inicial = None
                 ):
        
        self.L = L
        self.alpha = alpha

        self.rango_de_vision = rango_de_vision

        self.mapa = mapa

        #self.precios_barrios = precios_barrios
        self.precios_barrios = [barrio.precio_mudanza for barrio in mapa.barrios_definidos]

        #self.mapa_barrios = configuracion_barrios
        self.mapa_barrios = mapa.mapa

        #self.precios_prop_barrios = precios_prop_barrios
        self.precios_prop_barrios = [barrio.precio_propiedades for barrio in mapa.barrios_definidos]

        if not rng:
            rng = np.random.default_rng()

        self.rng = rng

        if not configuracion:
            # reemplaza np.random.randint
            configuracion = rng.integers(2, size=(self.L, self.L))
        
        self.configuracion = configuracion


        if capital_inicial is None:
            self.K = np.ones((self.L, self.L)) # Capital inicial de cada individuo, K=1
        else:
            self.K = capital_inicial.copy()

        self.U = np.zeros((self.L, self.L)) # Utilidad de cada individuo
        for i in range(self.L):
            for j in range(self.L):
                C_p, C_dist = self._num_vecinos(i, j, i, j)
                precio = precio_propiedad(C_p, C_dist)
                self.U[i, j] = utilidad(self.K[i, j], precio, self.alpha) # Utilidad inicial

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

        dist = distance(i1,j1, i2,j2) / (self.L * np.sqrt(2))

        # print(dist)

        if dist >= self.rango_de_vision:
            return

        agente_1, agente_2 = self.configuracion[i1, j1], self.configuracion[i2, j2]
        barrio_1, barrio_2 = self.mapa_barrios[i1, j1], self.mapa_barrios[i2, j2]

        if barrio_1 == barrio_2:
            costo_mudanza_a_1 = 0
            costo_mudanza_a_2 = 0
        else:
            costo_mudanza_a_1 = self.precios_barrios[barrio_1]
            costo_mudanza_a_2 = self.precios_barrios[barrio_2]

        K_1, K_2 = self.K[i1, j1], self.K[i2, j2]

        C_p_1_nuevo, C_dist_1_nuevo = self._num_vecinos(i1, j1, i2, j2)
        C_p_2_nuevo, C_dist_2_nuevo = self._num_vecinos(i2, j2, i1, j1)

        precio_1_nuevo, precio_2_nuevo = precio_propiedad(C_p_1_nuevo, C_dist_1_nuevo), precio_propiedad(C_p_2_nuevo, C_dist_2_nuevo)
        precio_1_nuevo *= self.precios_prop_barrios[barrio_2]
        precio_2_nuevo *= self.precios_prop_barrios[barrio_1]

        # delta_p = precio_2_nuevo - precio_1_nuevo
        p_promedio = (precio_2_nuevo + precio_1_nuevo) / 2

        precio_mudarse_a_1 = costo_mudanza_a_1 + precio_2_nuevo
        precio_mudarse_a_2 = costo_mudanza_a_2 + precio_1_nuevo

        # Verificar si ambos agentes tienen suficiente riqueza para la transacción
        #if np.min([K_1, K_2]) > np.abs(delta_p):
        if (precio_mudarse_a_2 - p_promedio - K_1 < 0) and (precio_mudarse_a_1 - p_promedio - K_2 < 0):

            # K_2_nuevo = K_2 - delta_p
            # K_1_nuevo = K_1 + delta_p

            K_1_nuevo = K_1 + precio_mudarse_a_2 - p_promedio  # Cálculo del capital potencial del agente 1
            K_2_nuevo = K_2 + precio_mudarse_a_1 - p_promedio  # Cálculo del capital potencial del agente 2

            
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