from IPython.display import display, HTML
from matplotlib import pyplot as plt
from tp.util.types import Dataclass, memoize
from tp.util.colors import Palette
from base64 import b64encode
from typing import Optional
from io import BytesIO
import numpy as np
import pandas as pd
import warnings
import json

class Barrio(Dataclass(init=True, frozen=True, unsafe_hash=True)):
    precio_propiedades: float
    precio_mudanza: float
    color: str

def generar_cuatro_cuadrantes(path: str, L: int = 50) -> None:
    with open(path, 'w') as file:
        for i in range(L):
            x = 0
            for j in range(L):

                if i <= L // 2:
                    if j < L // 2:
                        x = 1
                    else:
                        x = 2
                else:
                    if j < L // 2:
                        x = 0
                    else:
                        x = 3

                file.write(f'{x} ')
            file.write('\n')

class Mapa(Dataclass(init=True, frozen=True, eq=False)):
    mapa: np.ndarray[float]
    barrios: list[Barrio]
    image_bytes: Optional[bytes] = None

    def __hash__(self):
        _mapa = hash(self.mapa.flatten().tobytes())
        _barrios = tuple(hash(b) for b in self.barrios)
        return hash((_mapa, _barrios))
    
    def __eq__(self, value: object) -> bool:
        return hash(self) == hash(value)

    DEFAULT_COLORS = [ # Tableau 20
        '#E15759', # red
        '#4E79A7', # blue
        '#59A14F', # green
        '#F28E2B', # orange
        '#B6992D', # yellow
        '#499894', # teal
        '#79706E', # grey
        '#FABFD2', # pink
        '#B07AA1', # purple
        '#9D7660', # brown
        '#A0CBE8', # light blue
        '#D7B5A6', # light orange
        '#8CD17D', # light green
        '#F1CE63', # light yellow
        '#86BCB6', # light teal
        '#D37295', # light red
        '#BAB0AC', # light grey
        '#D4A6C8', # light purple
    ]

    @classmethod
    def load(cls, mapa, barrios, colores = None):
        """
        Carga un Mapa desde archivos locales.
        Espera:
        - 'mapa': un archivo de texto con la matriz de enteros que representa el mapa.
        - 'barrios': un archivo JSON con la info. de los barrios.
        - <opcional 'colores': una lista de colores en formato hexadecimal para cada barrio>

        El esquema de 'barrios' es:
        ```json
        [
            {
                "precio_propiedades": float,
                "precio_mudanza": float,
                // <opcional> "color": str
            },
            ...
        ]
        ```
        """

        # TODO: Podríamos implementar una verificacion
        # que mire si el numero max. de barrio en el
        # mapa es menor o igual a la cant. de barrios
        # en la lista.
        
        with open(mapa) as f:
            lines = f.readlines()
            lines
            grid = []
            for line in lines:
                grid.append([int(val) for val in line.strip().split()])
        
        _mapa = np.array(grid)
    
        with open(barrios, 'r') as f:
            _barrios = json.load(f)

        has_color = all('color' in b for b in _barrios)
        
        if has_color and colores is not None:
            warnings.warn("Both 'colores' and 'color' in the JSON file are specified, using 'colores'.", Warning)
            has_color = False
        
        if not has_color:
            warnings.warn("Colors not specified in the JSON file, using default palette.", Warning)
            
            if not colores:
                colores = Mapa.DEFAULT_COLORS

            for i,b in enumerate(_barrios):
                b['color'] = colores[i]
        
        return cls(_mapa, list(map(Barrio.from_dict, _barrios)))
    
    @memoize(debug=False)
    def _as_image(self, figsize=(5,5)) -> bytes:
        """
        Renderiza la cuadricula del mapa y guarda el resultado
        como bytes. 
        """
        
        plt.ioff() # Desactivo el modo interactivo
        fig = plt.figure(figsize=figsize)
        palette = Palette.from_hex(b.color for b in self.barrios)
        cmap, norm = palette.to_cmap()
        buffer = BytesIO()

        # Aca uso 'imshow' porque grafica un ndarray con cmap y norm de manera
        # sencilla. Para que no se muestre, usé 'ioff' mas arriba.
        plt.imshow(self.mapa, cmap=cmap, norm=norm)
        plt.savefig(buffer, format='png')
        plt.close(fig)
        plt.clf()
        return buffer.getvalue()
    
    def as_image(self, figsize=(5,5), force=False) -> bytes:
        return self._as_image(figsize, force=force)
    
    @memoize(debug=False)
    def barrios_info(self) -> pd.DataFrame:
        """
        Devuelve un DataFrame con la info. de los barrios como tabla.
        """
        info = {
            'Precio Propiedades': [b.precio_propiedades for b in self.barrios],
            'Precio Mudanza': [b.precio_mudanza for b in self.barrios],
        }

        result = pd.DataFrame(info)

        result.index = [f'Barrio {i+1}' for i in range(len(self.barrios))]

        # suma de 'Precio Propiedades' y 'Precio Mudanza'
        result['Peor caso'] = result.sum(axis=1)

        return result
    
    def _repr_html_(self, redraw=False):
        return f'<img src="data:image/png;base64,{b64encode(self.as_image(force=redraw)).decode()}">'

    def show(self, redraw=False):
        display(HTML(self._repr_html_(redraw)))
    