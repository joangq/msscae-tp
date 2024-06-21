from numpy.random import default_rng as random_number_generator, Generator as RNG
from tp.schelling import mercado_inmobiliario
from typing import Optional
from tp.util.barrios import Mapa
from tp.util.types import Lattice
from tp.util import simulador
import numpy as np
from matplotlib import pyplot as plt
from fundar import json
from fundar.json import JSONEncoder
from fundar.utils.time import now
from tqdm.auto import tqdm
from definiciones import (
    graficar_gini, 
    gini, 
    criterio_equilibrio, 
    gini_barrio, 
    satisfechos_en,
    crear_modelo,
    crear_simulador
)

class JsonEncoder(JSONEncoder):
    def default(self, o: object):
        if isinstance(o, np.ndarray):
            return list(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.integer):
            return int(o)

if __name__ == '__main__':
    N = 50
    alphas = [.1, .4, .8]
    rangos_de_vision = np.linspace(0,1, 10)

    rng = random_number_generator(seed=1)
    m = Mapa.load(mapa='./tp/mapas/cuatro_cuadrantes.txt',  barrios='./tp/barrios.json')
    m.as_image();
    m.show()

    caching_actions = (
    mercado_inmobiliario.utilidad_media,
    mercado_inmobiliario.capital_medio,
    )

    M = len(alphas) * len(rangos_de_vision)
    config_iniciales = rng.uniform(0, 10, (M, N, N))  # M matrices de NxN

    modelos = [None for _ in range(M)]
    simuladores = [crear_simulador(caching_actions) for _ in range(M)]

    i = 0
    for alpha in tqdm(alphas):
        for r in rangos_de_vision:
            modelos[i] = crear_modelo(alpha, r, rng, m, capital_inicial=config_iniciales[i])
            simuladores[i] = simuladores[i](modelos[i])
            i += 1

    for sim in tqdm(simuladores):
        sim.run()

    resultados_por_alpha = dict()

    c_rangos = len(rangos_de_vision)
    for i in range(0, M, c_rangos):
        alpha = alphas[i//c_rangos]
        resultados_por_alpha[alpha] = simuladores[i:i+c_rangos]

    result = []

    for s in simuladores:
        observaciones = s._cache
        alpha = s.modelo.alpha
        rango = s.modelo.rango_de_vision
        satisfechos_finales = s.modelo.satisfechos().sum()
        result.append({
            'observaciones': observaciones,
            'alpha': alpha,
            'rango': rango,
            'satisfechos': [
                satisfechos_en(0)(s.modelo),
                satisfechos_en(1)(s.modelo),
                satisfechos_en(2)(s.modelo),
                satisfechos_en(3)(s.modelo)
            ]
        })

    json.dump(result, f'results_{now.string}.json', cls=JsonEncoder)