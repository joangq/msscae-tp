from .interfaces import simulador_base, mercado_inmobiliario_base
from typing import Callable, Any, Iterable
from dataclasses import dataclass
import numpy as np

@dataclass
class simulador(simulador_base):
    """
    Simulador concreto para el modelo de mercado inmobiliario de Schelling.
    
    Encapsula al código que toma el modelo y lo simula.
    Implementa `on_step` y `on_finish`
    (Ver `simulador_base`)
    """
    modelo: mercado_inmobiliario_base

    # TODO: Otra forma de hacer esto es pasar directamente una funcion    
    # que tenga los parametros que necesita parcialmente aplicados,
    # y que sólo tome la lista de la utilidad promedio como parámetro
    # (y luego devuelva el booleano).
    criterio_equilibrio: Callable[[np.ndarray[float], Any], bool]
    max_steps: int
    lag: int
    tol: float

    cache_actions: None | Iterable[Callable[[mercado_inmobiliario_base], Any]] = None
    on_finish_print: bool = True
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
            self.cache_actions = (mercado_inmobiliario_base.utilidad_media, )

        self._cache = {k.__name__: np.full(self.max_steps, -1.0, dtype=float) for k in self.cache_actions}
        self._cache['utilidad'] = self._cache['utilidad_media'][0:self.paso_actual+1] # actualizo ref


    def on_step(self):
        """
        Callback garantizado a ser llamado en cada paso de la simulación.
        """
        for action in self.cache_actions:
            name = action.__name__
            self._cache[name][self.paso_actual] = action(self.modelo)
        
        self._cache['utilidad'] = self._cache['utilidad_media'][0:self.paso_actual+1] # actualizo ref
        self.paso_actual += 1

    def on_finish(self, alcanzo_equilibrio: bool, pasos: int) -> None:
        """
        Callback garantizado a ser llamado al finalizar la simulación.
        """
        if self.on_finish_print:
            print(f"La simulación {'alcanzó' if alcanzo_equilibrio else 'no alcanzó'} el equilibrio en {pasos} pasos.")

        del self._cache['utilidad'] # borro la ref a utilidad_media

        for k in self._cache.keys():
            self._cache[k].resize(pasos) # refcheck=false si quedan referencias


class SimuladorFactory:
    """
    Fabrica simuladores según los parámetros.
    Está diseñado para que se genere en `tp.definiciones.generar_inputs`.

    De cierta manera aplica parcialmente todos los parámetros de
    simulador.__init___, excepto el modelo.
    """
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    def __call__(self, modelo):
        return simulador(modelo, *self.args,**self.kwargs)