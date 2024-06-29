from typing import Optional
from abc import ABC as AbstractClass, abstractmethod
from numpy.random import Generator as RandomNumberGenerator
from matplotlib.animation import FuncAnimation
from tp.util.types import Lattice
from matplotlib import pyplot as plt
import numpy as np
from tqdm.auto import tqdm

class mercado_inmobiliario_interface(AbstractClass):
    """
    Interfaz para el modelo de mercado inmobiliario de Schelling.

    La diferencia que tiene esto con el modelo original es que usa inyección de dependencias
    para el generador de números aleatorios, dado que numpy no soporta el cambio global de
    la semilla de los números aleatorios.
    """
    @abstractmethod
    def __init__(self, 
                 L: int, 
                 configuracion: Optional[Lattice[int]] = None, 
                 alpha: float = 0.5, 
                 rng: Optional[RandomNumberGenerator] = None): ...
    
    @abstractmethod
    def proponer_intercambio(self) -> None: ...

    def satisfechos(self) -> Lattice[float]:
        return np.where(self.U >= 0.85, 1, 0)
    
    def utilidad_media(self) -> float:
        return np.mean(self.U.flatten())
    
    def capital_medio(self) -> float:
        return np.mean(self.K.flatten())
    
    def _num_vecinos(self, i_a: int, j_a: int, i, j: int) -> tuple[int, int]:
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
    
    def ronda_intercambio(self) -> None:
        N_intercambios = self.L**2
        for _ in range(N_intercambios):
            self.proponer_intercambio()
    
    def lattice(self) -> Lattice[int]:
        lattice = np.copy(self.configuracion)

        for i in range(self.L):
            for j in range(self.L):
                U_ij = self.U[i, j]

                # Marcar agentes insatisfechos dependiendo del tipo de agente
                if U_ij < 0.85:
                    lattice[i, j] = 4 if self.configuracion[i, j] == 1 else 5
        
        return lattice
    
    def lattice_plot(self) -> None:
        """
        Crea un gráfico de malla basado en los niveles de satisfacción de los agentes.
        """
        lattice = self.lattice()

        fig = plt.figure(figsize=(6,4))
        plt.imshow(lattice, cmap = 'inferno')
        plt.colorbar()
        plt.title("Sistema L = {} y alpha = {} \n (sitios naranjas y amarillos corresponden a agentes insatisfechos)".format(self.L, self.alpha))
        plt.show()
    
    def generar_animacion(self, frames: int) -> FuncAnimation:
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

class simulador_abstracto(AbstractClass):
    __current_step: int = 0
    __injections = list()
    """
    Interfaz para un simulador de un modelo de mercado inmobiliario de Schelling.
    """
    _cache: dict[str]
    """
    IMPORTANTE: El cache espera tener una clave 'utilidad' que tenga una lista de tamaño variable
    para poder pasarselo como argumento a la función de criterio de equilibrio.
    """

    @abstractmethod
    def __init__(self, 
                 modelo: mercado_inmobiliario_interface, 
                 max_steps: int, 
                 tol: float): ...
    
    @abstractmethod
    def on_step(self):
        """
        Actualiza el cache con los valores actuales del modelo.
        Se garantiza ser ejecutada en cada paso del modelo.
        """
        raise NotImplementedError("Implementation should complete this method's body.")
    

    def on_finish(self, alcanzo_equilibrio: bool, pasos: int) -> None:
        """
        Callback que se ejecuta al finalizar la simulación.

        Parámetros:
            alcanzo_equilibrio (bool): Indica si se alcanzó el equilibrio.
            pasos (int): Número de pasos ejecutados.
        """
        pass

    def step(self) -> None:
        """
        Ejecuta un paso del modelo, indistintamente de si se cumple el criterio de equilibrio.
        Además, actualiza el cache.
        """
        if (self.__current_step % 10 == 0):
            n_coords = 10
            random_coords = self.modelo.rng.integers(0, self.modelo.L, size=(n_coords, 2))
            random_amount = self.modelo.rng.uniform(0, 10, size=(n_coords,))

            injection = list(zip(random_coords, random_amount))

            for (i, j), amount in injection:
                self.__injections.append(((i, j), amount))
                self.modelo.K[i, j] += amount


        self.modelo.ronda_intercambio()
        self.on_step()
        self.__current_step += 1
    
    def run(self, use_tqdm=False) -> None:
        """
        Ejecuta todos los pasos hasta que se cumpla el criterio de equilibrio o se alcance el máximo de pasos.
        """
        iterator = tqdm if use_tqdm else iter
        equilibrio = False
        step = -1
        for step in iterator(range(self.max_steps)):
            self.step()
            if self.criterio_equilibrio(self._cache['utilidad'], self.lag, self.tol):
                equilibrio = True
                break
                
        self.on_finish(equilibrio, step)

    def export(self, **kwargs):
        return {k:v(self) for k,v in kwargs.items()}
        

