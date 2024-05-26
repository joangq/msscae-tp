from typing import Optional
from abc import ABC as AbstractClass, abstractmethod
from numpy.random import Generator as RandomNumberGenerator
from matplotlib.animation import FuncAnimation
from tp.util.types import Lattice

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
    def utilidad_media(self) -> float: ...
    
    @abstractmethod
    def capital_medio(self) -> float: ...
    
    @abstractmethod
    def _num_vecinos(self, i_a: int, j_a: int, i, j: int) -> tuple[int, int]: ...
    
    @abstractmethod
    def proponer_intercambio(self) -> None: ...
    
    @abstractmethod
    def ronda_intercambio(self) -> None: ...
    
    @abstractmethod
    def lattice(self) -> Lattice[int]: ...
    
    @abstractmethod
    def lattice_plot(self) -> None: ...
    
    @abstractmethod
    def generar_animacion(self, frames: int) -> FuncAnimation: ...

class simulador_abstracto(AbstractClass):
    """
    Interfaz para un simulador de un modelo de mercado inmobiliario de Schelling.
    """
    @abstractmethod
    def __init__(self, 
                 modelo: mercado_inmobiliario_interface, 
                 max_steps: int, 
                 tol: float): ...

    @abstractmethod
    def step(self) -> None: ...

    @abstractmethod
    def run(self) -> None: ...
