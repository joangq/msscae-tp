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
from fundar.parallelization import Multiprocess
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

def utilidad(modelo):
        return modelo.utilidad_media()

def correr_modelo(args):
    alpha, r, rng, m, cap_inicial = args
    modelo = mercado_inmobiliario(50, alpha=alpha, rango_de_vision=r, mapa=m, rng=rng, capital_inicial=cap_inicial)

    caching_actions = (
        mercado_inmobiliario.utilidad_media,
        mercado_inmobiliario.capital_medio
    )

    sim = simulador(modelo, criterio_equilibrio, max_steps=150, lag=20, tol=1e-3, cache_actions=caching_actions)
    sim.run()
    return sim;

def main():
    N = 50
    alphas = [.1, .4, .8]
    rangos_de_vision = np.linspace(0,1, 1000)

    rng = random_number_generator(seed=1)
    m = Mapa.load(mapa='./src/tp/mapas/tercios.txt',  barrios_definidos='./src/tp/barrios.json')
    m.show()

    M = len(alphas) * len(rangos_de_vision)
    config_iniciales = rng.uniform(0, 1, (M, N, N))  # M matrices de NxN

    i = 0
    inputs = []
    for alpha in tqdm(alphas):
        for r in rangos_de_vision:
            inputs.append((alpha, r, rng, m, config_iniciales[i]))
            i += 1

    simuladores = \
        Multiprocess.map(
            correr_modelo,
            inputs,
            chunksize=3
        )

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
            'pasos': s.paso_actual,
            'satisfechos': [
                satisfechos_en(0)(s.modelo),
                satisfechos_en(1)(s.modelo),
                satisfechos_en(2)(s.modelo),
                satisfechos_en(3)(s.modelo)
            ],
            'gini_indices': [
                gini_barrio(0)(s.modelo),
                gini_barrio(1)(s.modelo),
                gini_barrio(2)(s.modelo),
                gini_barrio(3)(s.modelo)
            ],
            'gini_t': gini(s.modelo.K.flatten())
        })

    # Result schema:
    # {
    #     'observaciones': {
    #         'utilidad_media': List[float],
    #         'capital_medio': List[float]
    #     },
    #     'alpha': float,
    #     'rango': float,
    #     'pasos': int,
    #     'satisfechos': List[int],
    #     'gini_indices': List[float],
    #     'gini_t': float
    # }


    json.dump(result, f'results__{now.string}.json', cls=JsonEncoder)


if __name__ == '__main__':
    main()
