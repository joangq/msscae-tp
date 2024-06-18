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