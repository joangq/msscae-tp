from .interfaces import simulador_abstracto, mercado_inmobiliario_interface
from typing import Callable, Any, Iterable
from numpy import ndarray as numpy_array

class simulador(simulador_abstracto):
    """
    Simulador concreto para el modelo de mercado inmobiliario de Schelling.
    
    Encapsula al código que toma el modelo y lo simula.
    """
    def __init__(self, 
                 modelo: mercado_inmobiliario_interface, 
                 criterio_equilibrio: Callable[[numpy_array[float], Any], bool],
                 max_steps: int, 
                 lag: int, 
                 tol: float):
        self.modelo = modelo
        self.max_steps = max_steps
        self.lag = lag
        self.tol = tol
        self.criterio_equilibrio = criterio_equilibrio
        self._utilidad = []
        self._capital = []

    @property
    def utilidad(self) -> Iterable[float]:
        return self._utilidad
    
    @property
    def capital(self) -> Iterable[float]:
        return self._capital

    def step(self) -> None:
        """
        Ejecuta un paso del modelo, indistintamente de si se cumple el criterio de equilibrio.
        """
        self.modelo.ronda_intercambio()
        self.utilidad.append(self.modelo.utilidad_media())
        self.capital.append(self.modelo.capital_medio())
    
    def run(self) -> None:
        """
        Ejecuta todos los pasos hasta que se cumpla el criterio de equilibrio o se alcance el máximo de pasos.
        """
        for step in range(self.max_steps):
            self.step()
            if self.criterio_equilibrio(self.utilidad, self.lag, self.tol):
                break