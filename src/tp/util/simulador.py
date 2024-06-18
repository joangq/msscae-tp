from .interfaces import simulador_abstracto, mercado_inmobiliario_interface
from typing import Callable, Any, Iterable
from dataclasses import dataclass
import numpy as np

@dataclass
class simulador(simulador_abstracto):
    """
    Simulador concreto para el modelo de mercado inmobiliario de Schelling.
    
    Encapsula al código que toma el modelo y lo simula.
    """
    modelo: mercado_inmobiliario_interface

    # TODO: Otra forma de hacer esto es pasar directamente una funcion    
    # que tenga los parametros que necesita parcialmente aplicados,
    # y que sólo tome la lista de la utilidad promedio como parámetro
    # (y luego devuelva el booleano).
    criterio_equilibrio: Callable[[np.ndarray[float], Any], bool]
    max_steps: int
    lag: int
    tol: float

    cache_actions: None | Iterable[Callable[[mercado_inmobiliario_interface], Any]] = None
    """
    En cada paso, cada funcion de 'cache_actions' se ejecuta, y su resultado se guarda en el cache.
    (ver 'cache')
    """


    _cache: dict = None
    """
    El cache guarda en 'utilidad' una slice de 'utilidad_media' que va creciendo a medida que se ejecutan los pasos.
    Mientras tanto, 'utilidad_media' se pre-genera con el largo máximo de pasos.

    IMPORTANTE: Las claves del diccionario son los nombres de las funciones (no los qualnames), por ende dos funciones
    con **exactamente** el mismo nombre van a colisionar.
    """

    def __post_init__(self):
        self.paso_actual = 0
        if self.cache_actions is None:
            self.cache_actions = (mercado_inmobiliario_interface.utilidad_media, )

        self._cache = {k.__name__: np.full(self.max_steps, -1.0, dtype=float) for k in self.cache_actions}
        self._cache['utilidad'] = self._cache['utilidad_media'][0:self.paso_actual+1]


    def cache(self):
        for action in self.cache_actions:
            name = action.__name__
            self._cache[name][self.paso_actual] = action(self.modelo)
        
        self._cache['utilidad'] = self._cache['utilidad_media'][0:self.paso_actual+1]
        self.paso_actual += 1

    def on_finish(self, alcanzo_equilibrio: bool, pasos: int) -> None:
        print(f"La simulación {'alcanzó' if alcanzo_equilibrio else 'no alcanzó'} el equilibrio en {pasos} pasos.")

        del self._cache['utilidad'] # borro la ref a utilidad_media

        for k in self._cache.keys():
            self._cache[k].resize(pasos) # refcheck=false si quedan referencias