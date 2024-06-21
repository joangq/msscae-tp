from typing import Iterable
from matplotlib import pyplot as plt
import numpy as np

def criterio_equilibrio(serie: np.ndarray[float], lag: int, tol: float) -> bool:
    """
    Define cuando una serie temporal se encuentra en equilibrio
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
def gini(_ys: np.ndarray[float]) -> float:
    """
    Calcula el coeficiente de Gini de una lista de valores.
    Donde:
        0 = igualdad total (ejemplo: todos los valores son iguales)
        1 = desigualdad absoluta (ejemplo: un solo valor es distinto a cero)
    """
    if len(_ys) == 0:
        return None
    
    if not isinstance(_ys[0], (float, np.floating)):
        raise TypeError(f"Expected an iterable of floating-like numbers, got '{type(_ys[0]).__name__}'.")

    n = len(_ys)
    ys = np.sort(_ys)
    sumas_parciales = np.cumsum(ys)
    sumas_indexadas = np.sum(np.array([i*x for i,x in enumerate(ys, start=1)]))
    
    total = sumas_parciales[-1]

    return (2*sumas_indexadas / (n * total)) - ((n+1)/n)


def graficar_gini(data: Iterable[float], coef: float):
    plt.style.use('ggplot')
    sorted_data = np.sort(data)
    
    n = len(sorted_data)
    lorenz_curve = np.cumsum(sorted_data) / np.sum(sorted_data)
    lorenz_curve = np.insert(lorenz_curve, 0, 0) # Agrego el (0,0)
    
    # Línea de Igualdad
    equality_line = np.linspace(0, 1, len(lorenz_curve))
    
    # Curva de Lorenz
    plt.figure(figsize=(8, 8))
    plt.plot(equality_line, lorenz_curve, label='Curva de Lorenz', color='red')
    plt.plot(equality_line, equality_line, label='Línea de Igualdad', linestyle='--', color='blue')

    # Relleno entre la línea de igualdad y la curva de Lorenz
    plt.fill_between(equality_line, equality_line, lorenz_curve, color='blue', alpha=0.1)

    # Relleno abajo de la curva de Lorenz
    plt.fill_between(equality_line, 0, lorenz_curve, color='red', alpha=0.2)
    
    # Muestro el índice
    plt.text(0.6, 0.2, f'Gini Index = {coef:.2f}', fontsize=12, bbox=dict(facecolor='white', alpha=0.5), color='black')
    plt.title('Lorenz Curve and Gini Index', fontsize=16)
    plt.xlabel('Proporción acumulada de la población de menor a mayor ingreso', fontsize=14)
    plt.ylabel('Proporción acumulada de riqueza', fontsize=14)
    legend = plt.legend(loc='best')
    for t in legend.get_texts():
        t.set_color('black')
    
    plt.grid(True)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    
    # Numeros como porcentajes
    plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x * 100)}%'))
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{int(y * 100)}%'))

    plt.tight_layout()
    
    return plt

# Funciones de observacion del modelo ==================================================================================================

def gini_barrio(i: int) -> callable:
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

def satisfechos_en(i: int) -> callable:
    """
    Fabrica de funciones, devuelve una funcion 'satisfechos_por_barrio_i' que calcula la proporcion de agentes satisfechos en el barrio i.
    """

    def _(modelo) -> float:
        """
        Calcula la proporcion de agentes satisfechos en el barrio i.
        """
        return modelo.satisfechos()[modelo.mapa_barrios == i].sum()
    _.__name__ = f'satisfechos_en_{i}' # Importante cambiar el nombre para que no colisione en el cache.

    return _

# Pre-generacion ============================================================================================

from numpy.random import Generator as RNG
from tp.util.barrios import Mapa
from typing import Optional
from tp.util.types import Lattice
from tp.schelling import mercado_inmobiliario
from tp.util import simulador


def crear_modelo(alpha: float, 
                 rango_de_vision: float,
                 rng: RNG,
                 mapa: Mapa,
                 L: Optional[float] = None,
                 capital_inicial: Optional[Lattice[float]] = None,
                 min_capital: int = 0,
                 max_capital: int = 10,
) -> mercado_inmobiliario:
    """
    L = 50
    capital_inical = random.uniform(0,1, (N,N))
    """

    if L is None:
        L = 50

    if capital_inicial is None:
        capital_inicial = rng.uniform(min_capital, max_capital, (L,L))

    modelo = mercado_inmobiliario(
    L=L, 
    alpha=alpha,
    rng=rng, 
    mapa = mapa,
    rango_de_vision=rango_de_vision,
    capital_inicial=capital_inicial,
    )
    
    return modelo


def crear_simulador(caching_actions: list[callable],
                    max_steps: Optional[int] = None, 
                    tol: Optional[float] = None,
                    lag: Optional[int] = None,
                    equilibrio: Optional[callable] = None
                    ) -> callable:
    """
    tol = 1e-3
    lag = 20
    max_steps = 150
    """
    
    if tol is None:
        tol = 1e-3

    if lag is None:
        lag = 20

    if max_steps is None:
        max_steps = 150

    if equilibrio is None:
        equilibrio = criterio_equilibrio
    
    def _simulador_parcial(modelo: mercado_inmobiliario) -> simulador:
        """
        Fabrica de simuladores
        """
        return simulador(modelo=modelo, 
                         criterio_equilibrio=equilibrio, 
                         max_steps=max_steps, 
                         lag=lag, 
                         tol=tol, 
                         cache_actions=caching_actions
                         )

    return _simulador_parcial

