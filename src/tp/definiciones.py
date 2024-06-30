from tp.util.barrios import Mapa
from tp.util.simulador import SimuladorFactory
from numpy.random import default_rng as random_number_generator
from tp.schelling import mercado_inmobiliario
from typing import Iterable
import numpy as np
import warnings
import os.path

def generate_filename(mapa_name, start_max, alphas, subdivisiones, seed):
    """
    Crea un nombre de archivo a partir de los parametros de la simulacion con
    el siguiente formato:
        {mapa_name}_{start_max}_{alphas}_{subdivisiones}_{seed}.json

    Ejemplo:
    >>> generate_filename('mapa', 10, [0.1, .3, 0.2], 10, 42)
    >>> 'mapa_10_12_10_42.json'
    """
    filename = '_'.join((mapa_name, str(start_max) ,''.join(str(int(max(x*10,1))) for x in sorted(alphas, reverse=True)), str(subdivisiones), str(seed)))
    filename += '.json'
    return os.path.join('./tp/data/resultados', filename)

def non_colliding_name(name: str) -> str:
    """
    Crea un nombre de archivo que no colisione con los ya existentes.
    Si el nombre no existe, lo devuelve tal cual.

    Ejemplo:
    >>> non_colliding_name('archivo.txt')
    >>> 'archivo_1.txt' # Si ya existe 'archivo.txt'
    >>> 'archivo.txt' # Si no existe 'archivo.txt'
    """
    if not os.path.exists(name):
        return name
    
    i = 1
    name, extension = os.path.splitext(name)
    while True:
        new_name = f"{name}_{i}{extension}"
        if not os.path.exists(new_name):
            return new_name
        i += 1

def criterio_equilibrio(serie: np.ndarray[float], lag: int, tol: float) -> bool:
    """
    Define cuando una serie temporal se encuentra en equilibrio.

    En particular, este criterio considera que el sistema
    se encuentra en equilibrio si los ultimos 'lag' valores
    de la serie no superan una variacion relativa de 'tol'.

    **Importante:** Esto asume que la 'serie' es un arreglo que va
    mutando en el tiempo, donde el ultimo valor es el mas reciente.
    """
    
    L = len(serie)
    if L <= lag:
        return False
    
    for i in range(L-1-lag , L-1):
        if np.abs((serie[i] - serie[i-1])/serie[i-1]) >= tol:
            return False
        
    return True


# Adptado de: https://en.wikipedia.org/wiki/Gini_coefficient#Alternative_expressions
# $$
#   {\displaystyle G={\frac {1}{n}}\left(n+1-2\left({\frac {\sum _{i=1}^{n}(n+1-i)y_{i}}{\sum _{i=1}^{n}y_{i}}}\right)\right).}
# $$
def _gini(_ys: np.ndarray[float]) -> float:
    """
    Calcula el coeficiente de Gini de una lista de valores.
    Donde:
        0 = igualdad total (ejemplo: todos los valores son iguales)
        1 = desigualdad absoluta (ejemplo: un solo valor es distinto a cero)
    """
    if len(_ys) == 0:
        return None
    
    if not isinstance(_ys[0], (float, np.floating, int, np.integer)):
        raise TypeError(f"Expected an iterable of floating-like numbers, got '{type(_ys[0]).__name__}'.")

    n = len(_ys)
    ys = np.sort(_ys)
    sumas_parciales = np.cumsum(ys)
    sumas_indexadas = np.sum(np.array([i*x for i,x in enumerate(ys, start=1)]))
    
    total = sumas_parciales[-1]

    return (2*sumas_indexadas / (n * total)) - ((n+1)/n)

def gini(data: Iterable[float]) -> float:
    """
    Es la versión type-safe de `_gini`.
    Toma un iterable de valores y calcula el coeficiente de Gini.
    - Si el input no es un arreglo de numpy, se castea.
    - Si el input es una matriz, se aplanan los valores.
    - Si el input es un arreglo de numpy, se calcula el coeficiente de Gini.
    """
    if not isinstance(data, np.ndarray):
        warnings.warn("Casteando entrada a un arreglo de numpy para calcular el coeficiente de Gini.")
        return gini(np.array(data))
    
    # { type(data) = np.ndarray }

    # Esto es una pequeña conveniencia
    # que abusa de la sintaxis de match
    # sobre 'guardas estructurales'.
    # Definida en https://peps.python.org/pep-0622/
    match data.shape:
        case (_,): # Si tiene *exactamente* una dimension
            return _gini(data) # ok
        case _: # Si tiene otra forma
            warnings.warn("Aplanando la matriz para calcular el coeficiente de Gini.")
            return gini(data.flatten())
        
# Observaciones sobre el modelo - Utilidades para poder medir la simulación en un momento dado. ==========================================

def gini_total(modelo):
    return gini(modelo.K)

def gini_barrio_k(i: int) -> callable:
    """
    Fabrica de funciones, devuelve una funcion 'gini_barrio_i' que calcula el gini del capital de los agentes en el barrio i.
    """

    def _(modelo) -> float:
        """
        Calcula el gini del capital de los agentes en el barrio i.
        """
        mask = modelo.mapa_barrios == i
        capital_barrio = modelo.K * mask
        capital_barrio = capital_barrio.flatten()
        capital_barrio = capital_barrio[mask.flatten().nonzero()[0]]
        return gini(capital_barrio)
    _.__name__ = f'gini_barrio_{i}' # Importante cambiar el nombre para que no colisione en el cache.

    return _

def gini_barrio_u(i: int) -> callable:
    """
    Fabrica de funciones, devuelve una funcion 'gini_barrio_i' que calcula el gini del capital de los agentes en el barrio i.
    """

    def _(modelo) -> float: # auxiliar para currificar
        """
        Calcula el gini del capital de los agentes en el barrio i.
        """
        mask = modelo.mapa_barrios == i
        utilidad_barrio = modelo.U * mask
        utilidad_barrio = utilidad_barrio.flatten()
        utilidad_barrio = utilidad_barrio[mask.flatten().nonzero()[0]]
        return gini(utilidad_barrio)
    _.__name__ = f'gini_barrio_u_{i}' # Importante cambiar el nombre para que no colisione en el cache.

    return _

def satisfechos_en(i: int) -> callable:
    """
    Fabrica de funciones, devuelve una funcion 'satisfechos_por_barrio_i' que calcula la proporcion de agentes satisfechos en el barrio i.
    """

    def _(modelo) -> float: # auxiliar para currificar
        """
        Calcula la proporcion de agentes satisfechos en el barrio i.
        """
        return modelo.satisfechos()[modelo.mapa_barrios == i].sum()
    _.__name__ = f'satisfechos_en_{i}' # Importante cambiar el nombre para que no colisione en el cache.

    return _

def cantidad_de_pasos(sim) -> int:
     return sim.paso_actual

def observaciones_de(sim) -> dict[str, any]:
    return {k:list(v) for k,v in sim._cache.items()}

def alpha_del_modelo(sim) -> float:
    return sim.modelo.alpha

def rango_del_modelo(sim) -> float:
    return sim.modelo.rango_de_vision

def satisfechos_finales_del_modelo(sim) -> np.integer:
    return sim.modelo.satisfechos().sum()

def gini_capital_modelo(sim) -> float:
    return gini(sim.modelo.K.flatten())

def gini_utilidad_modelo(sim) -> float:
    return gini(sim.modelo.U.flatten())

def gini_capital_por_barrio(sim) -> list[float]:
        return [gini_barrio_k(x)(sim.modelo) for x in sim.modelo.mapa.barrios]

def gini_utilidad_por_barrio(sim) -> list[float]:
     return [gini_barrio_u(x)(sim.modelo) for x in sim.modelo.mapa.barrios]

def satisfechos_por_barrio(sim) -> list[np.integer]:
     return [satisfechos_en(x)(sim.modelo) for x in sim.modelo.mapa.barrios]

# Utilidades para ejecutar el modelo, secuencial o paralelamente. ========================================================================

def correr_modelo(args):
    """
    Ejecuta un modelo de mercado inmobiliario con los argumentos dados.
    Es usado por `correr_secuencialmente` y `correr_en_paralelo`.
    """
    alpha, r, rng, m, cap_inicial, simulador_factory = args
    modelo = mercado_inmobiliario(alpha=alpha, 
                                  rango_de_vision=r, 
                                  mapa=m, 
                                  rng=rng, 
                                  capital_inicial=cap_inicial)

    observaciones = (
        mercado_inmobiliario.utilidad_media,
        mercado_inmobiliario.capital_medio
    )

    sim = simulador_factory(modelo)
    sim.run(use_tqdm=False)
    return sim;

# Probablemente esto se pueda hacer
# simplemente usando properties, pero
# no voy a cambiarlo a último momento.
class MutableStorage:
    """
    Contenedor mutable, necesario para poder
    almacenar el resultado de la simulación
    dentro de un subproceso.

    (Ver 
    """
    def __init__(self):
        self.__storage__ = None

    def set_store(self, x):
        self.__storage__ = x

    def get_store(self):
        return self.__storage__
    
def correr_secuencialmente(inputs, use_tqdm=True):
    """
    Ejecuta los inputs del modelo de mercado_inmobiliario.
    Lo hace secuencialmente.
    """
    if use_tqdm:
        from tqdm.auto import tqdm
        iterator = tqdm
    else:
        iterator = iter
        
    return [correr_modelo(xs) for xs in iterator(inputs)]

def correr_en_paralelo(inputs, chunksize=3, use_tqdm=True):
    """
    Ejecuta los inputs con el modelo de mercado_inmobiliario.
    Usa múltiples procesos.

    En Windows, si se ejecuta este código sin guardas (`if __name__ == '__main__'`)
    se va a romper la pipe de procesos.
    """
    from tqdm.contrib.concurrent import process_map
    return process_map(correr_modelo, inputs, chunksize=chunksize, disable=(not use_tqdm))

def generar_inputs(alphas, subdivisiones, seed, start_max, distribucion, barrios_path, mapa_path):
    """
    Genera M estados para ser consumidos por un executor.
    Con M = len(alphas) * subdivisiones

    Ver `correr_modelo`.
    """
    mapa = Mapa.load(mapa_path, barrios_definidos=barrios_path)
    rng = random_number_generator(seed)

    caching_actions = (
        mercado_inmobiliario.utilidad_media,
        mercado_inmobiliario.capital_medio,
    )

    rangos_de_vision = np.linspace(0,1, subdivisiones)

    M = len(alphas) * len(rangos_de_vision)
    N = mapa.mapa.shape[0]
    method = getattr(rng, distribucion)
    config_iniciales = rng.uniform(0, start_max, (M, N, N)) # M matrices de N x N

    i = 0
    inputs = []
    for alpha in alphas:
        for r in rangos_de_vision:
            inputs.append((alpha, r, rng, mapa, config_iniciales[i], 
                           SimuladorFactory(criterio_equilibrio=criterio_equilibrio, 
                                            max_steps=150, lag=20, tol=1e-3, on_finish_print=False, 
                                            cache_actions=caching_actions)))
            i += 1

    return inputs