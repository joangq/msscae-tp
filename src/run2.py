import argparse
import random
# from fundar.utils import find_file
from typing import Optional
import os.path

from numpy.random import default_rng as random_number_generator
from tp.schelling import mercado_inmobiliario
from tp.presentacion import (
    mostrar_texto, 
    generar_capital_inicial, 
    mostrar_satisfechos_antes_despues, 
    plot_satisfacciones_para_alpha,
    create_os_buttons,
    detect_os,
    opciones_ejecutar_modelo
)

from tp.definiciones import (
    gini, 
    criterio_equilibrio, 
    gini_barrio_k, 
    gini_barrio_u, 
    gini_total,
    satisfechos_en,
    observaciones_de,
    alpha_del_modelo,
    rango_del_modelo,
    gini_capital_modelo,
    gini_utilidad_modelo,
    gini_capital_por_barrio,
    gini_utilidad_por_barrio,
    satisfechos_por_barrio,
    cantidad_de_pasos,
    correr_modelo,
    correr_en_paralelo,
    correr_secuencialmente,
    generate_filename,
    non_colliding_name,
    generar_inputs,
    MutableStorage
)
from tp.util import Mapa, simulador, SimuladorFactory
import tp.util.json as json
import numpy as np
from tqdm.auto import tqdm


def search_downwards(name: str, start_path: str, max_depth: int, current_depth: int=0) -> Optional[str]:
    if current_depth > max_depth:
        return None
    
    for root, dirs, files in os.walk(start_path):
        if name in files or name in dirs:
            return os.path.join(root, name)
        
        if current_depth + 1 > max_depth:
            break
        
        for d in dirs:
            result = search_downwards(name, os.path.join(root, d), max_depth, current_depth + 1)
            if result:
                return result

    return None

def search_upwards(name: str, start_path: str, max_up_depth: int, max_down_depth: int, current_up_depth: int=0) -> Optional[str]:
    if current_up_depth > max_up_depth:
        return None
    
    result = search_downwards(name, start_path, max_down_depth)
    if result:
        return result
    
    parent_dir = os.path.abspath(os.path.join(start_path, '..'))
    
    if parent_dir == start_path:
        return None
    
    return search_upwards(name, parent_dir, max_up_depth, max_down_depth, current_up_depth + 1)

def find_file(name: str, path: str, max_up_depth: int=3, max_down_depth: int=3, throw_error: bool=False) -> Optional[str]:
    result = search_downwards(name, path, max_down_depth)

    if result:
        result = os.path.abspath(result)
        return result
    

    result = search_upwards(name, path, max_up_depth, max_down_depth)
    if result:
        result = os.path.abspath(result)
        return result
    
    if throw_error:
        raise FileNotFoundError(f"File '{name}' not found.")
    

def is_pathlike(s: str) -> bool:
    return os.path.exists(s)

def has_extension(s: str) -> bool:
    return '.' in s

class ArgumentParser(argparse.ArgumentParser):
    def __init__(self):
        super(ArgumentParser, self).__init__()

        # str
        self.add_argument('-a', '--alphas', type=str, default='[.1, .4, .8]', help='Alphas.')

        # int
        self.add_argument('-s', '--subdivisiones', type=int, default=10, help='Subdivisiones.')

        # str
        self.add_argument('-r', '--seed', type=str, default='random', help='Seed.')

        # float
        self.add_argument('-x', '--start_max', type=float, default=10, help='Start max.')

        # str
        self.add_argument('-d', '--distribucion', type=str, default='uniform', help='Distribution.')

        # str
        self.add_argument('-b', '--barrios', type=str, default='barrios', help='Barrios.')

        # str
        self.add_argument('-m', '--mapa', type=str, default='cuatro_cuadrantes', help='Mapa.')

        # str
        self.add_argument('-o', '--output', type=str, default=None, help='Output.')

    def parse_args(self):
        self.args = super(ArgumentParser, self).parse_args().__dict__

        for key, value in self.args.items():
            setattr(self, key, value)

        # Param initialization
        
        self.alphas = eval(self.alphas, {}, {})
        
        if self.seed.isnumeric():
            self.seed = int(self.seed)
        else:
            self.seed = random.randint(0, 1000)

        if not is_pathlike(self.barrios):
            if not has_extension(self.barrios):
                self.barrios = self.barrios + '.json'

            self.barrios = find_file(self.barrios, os.getcwd())

        if not is_pathlike(self.mapa):

            if not has_extension(self.mapa):
                self.mapa = self.mapa + '.txt'

            self.mapa = find_file(self.mapa, os.getcwd())

        # Update args

        for key, value in self.args.items():
            self.args[key] = getattr(self, key)
        
        return self



def main():
    parser = ArgumentParser().parse_args()

    for k,v in parser.args.items():
        print(f'{k} = {v}')
    print()
    
    inputs = generar_inputs(parser.alphas, parser.subdivisiones, parser.seed, parser.start_max, parser.distribucion, parser.barrios, parser.mapa)

    if not parser.output:
        mapa_name = os.path.basename(parser.mapa)
        mapa_name = os.path.splitext(mapa_name)[0]
        name = generate_filename(mapa_name, parser.start_max, parser.alphas, parser.subdivisiones, parser.seed)
        parser.output = name
    
    output = non_colliding_name(parser.output)

    print(f'Output: {output}')

    simuladores = correr_en_paralelo(inputs)

    resultados = [
        s.export(
            observaciones = observaciones_de,
            alpha = alpha_del_modelo,
            rango = rango_del_modelo,
            gini_tk = gini_capital_modelo,
            gini_tu = gini_utilidad_modelo,
            ginis_k = gini_capital_por_barrio,
            ginis_u = gini_utilidad_por_barrio,
            satisfechos = satisfechos_por_barrio,
            pasos = cantidad_de_pasos)
        
        for s in simuladores
    ]


    json.dump(resultados, output)



if __name__ == '__main__':
    main()