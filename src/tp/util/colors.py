"""
Este archivo contiene utilidades y conveniencias para trabajar con colores
y paletas de colores. Específcamente es para poder visualizar los colores y
pasarlos fácilmente a 'colormaps' de matplotlib.

No es una parte central de la simulación, pero fue útil para fines de desarrollo.
"""

from IPython.display import display
from PIL import Image
import numpy as np
from io import BytesIO
from base64 import b64encode
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import pickle

def dict_as_colormap(color_dict, name=None):
    """
    Convierte un diccionario de colores en un colormap de matplotlib.
    Está hecho para ser usado por Color.to_cmap.
    """
    if not name:
        name = '-'.join(x for x in color_dict.values())
    
    keys = sorted(color_dict.keys())
    keys_nums = [i for i,_ in enumerate(keys)]


    colors = [color_dict[key] for key in keys]
    cmap = mcolors.ListedColormap(colors, name)
    
    
    bounds = np.array(keys_nums + [max(keys_nums)+1])
    norm = mcolors.BoundaryNorm(bounds, cmap.N)
    
    return cmap, norm


def plot_square(l: int, color: tuple[np.floating]) -> Image:
    """
    Dado un color, muestra un cuadrado de ese color.

    Uso:
    >>> plot_square(100, (1, 0, 0))
    """
    if isinstance(color, str):
        return Image.new('RGB', (l,l), color)
    return Image.new('RGB', (l, l), tuple(int(255 * c) for c in color[:3]))

def concat_images(images) -> Image:
    """
    Concatena varias imágenes en una sola.
    Está hecha para ser usada en conjunto con 'plot_square'.
    """
    widths, heights = zip(*(i.size for i in images))

    total_width = sum(widths)
    max_height = max(heights)

    new_im = Image.new('RGB', (total_width, max_height))

    x_offset = 0
    for im in images:
        new_im.paste(im, (x_offset, 0))
        x_offset += im.size[0]

    return new_im

def plot_squares(l, colors):
    """
    Dada una serie de colores, los muestra como una tira de cuadrados coloreados.
    Ejemplo en: https://gist.github.com/joangq/b43e71b3b7cb2af077908702a1436c2f#file-zzz-plot-colors-ipynb
    """
    squares = [plot_square(l, c) for c in colors]
    return concat_images(squares)

def pil_to_base64(image: Image) -> str:
    """
    Codifica una imagen en Base64.
    Útil para graficarlo con HTML.
    """
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    img_bytes = buffer.getvalue()
    img_base64 = b64encode(img_bytes)
    img_base64_str = img_base64.decode('utf-8')
    return img_base64_str

def pil_to_html(image: Image) -> str:
    """
    Toma una imagen y la convierte en un tag HTML para mostrarla.
    """
    img_base64_str = pil_to_base64(image)
    return f'<div><img src="data:image/png;base64,{img_base64_str}"></div>'

def cmap_norm_to_pickle(cmap, norm, filename):
    """
    Guarda un colormap y una normalización en un archivo pickle.
    """
    with open(filename, 'wb') as f:
        pickle.dump((cmap, norm), f)

def pickle_to_cmap_norm(filename) -> tuple:
    """
    Carga un colormap y una normalización desde un archivo pickle.
    """
    with open(filename, 'rb') as f:
        return pickle.load(f)

class Color:
    """
    Representación de un color.
    Tiene utilidades para convertirlo a distintos formatos y mostrarlo.
    """
    img = None
    name = None

    def __init__(self, r: int, g: int, b: int, a: int = 255):
        self.r = r
        self.g = g
        self.b = b
        self.a = a

    @classmethod
    def from_hex(cls, hex: str):
        """
        Crea un color a partir de un string hexadecimal.
        Uso:
        >>> Color.from_hex("#FF0000")
        """
        hex = hex.lstrip("#")
        return cls(*[int(hex[i:i+2], 16) for i in (0, 2, 4)])
    
    @classmethod
    def from_rgba(cls, rgba: tuple[int]):
        """
        Crea un color a partir de una tupla de valores RGBA.
        Uso:
        >>> Color.from_rgba((255, 0, 0, 255))
        """
        return cls(*rgba)

    @classmethod
    def from_rgb(cls, rgb: tuple[int]):
        """
        Crea un color a partir de una tupla de valores RGB.
        Uso:
        >>> Color.from_rgb((255, 0, 0))
        """
        return cls(*rgb, 255)
    
    @classmethod
    def from_float(cls, r: float, g: float, b: float, a: float = 1.0):
        """
        Crea un color a partir de valores RGB en formato decimal.
        Uso:
        >>> Color.from_float(1.0, 0.0, 0.0)
        """
        return cls(*[int(255 * c) for c in (r, g, b, a)])
    
    @classmethod
    def from_iterable(cls, iterable):
        """
        Toma un objeto iterable (a.k.a que tenga '__iter__') y lo convierte en un color.
        Uso:
        >>> Color.from_iterable((1.0, 0.0, 0.0))
        >>> Color.from_iterable([1.0, 0.0, 0.0])
        >>> Color.from_iterable((x for x in range(0,1,0.1)))
        """
        if all(isinstance(i, float) for i in iterable):
            return cls.from_float(*iterable)
        return cls(*iterable)
    
    @classmethod
    def from_dict(cls, d):
        """
        Toma un diccionario de la forma:
        {'value': hex, 'name': str}

        Uso:
        >>> Color.from_dict({'value': '#FF0000', 'name': 'red'})
        """
        result = cls.from_hex(d['value'])
        result.name = d.get('name')
        return result


    @classmethod
    def from_records(cls, d, **kwargs):
        """
        Toma un diccionario y usa los kwargs para acceder a los
        contenidos.

        Uso:
        >>> Color.from_records({'color': '#FF0000', 'title': 'red'}, value='color', name='title')
        """
        return cls.from_dict({'value': d[kwargs['value']], 'name': d.get(kwargs.get('name'))})
    
    def as_hex(self):
        """
        Convierte el color a un string hexadecimal.
        Uso:
        >>> Color(255, 0, 0).as_hex() # '#FF0000'
        """
        return "#{:02x}{:02x}{:02x}".format(self.r, self.g, self.b).upper()

    def as_rgba(self):
        """
        Convierte el color a una tupla de valores RGBA.
        Uso:
        >>> Color(255, 0, 0).as_rgba() # (255, 0, 0, 255)
        """
        return (self.r, self.g, self.b, self.a)
    
    def as_rgb(self):
        """
        Convierte el color a una tupla de valores RGB.
        Uso:
        >>> Color(255, 0, 0).as_rgb() # (255, 0, 0)
        """
        return (self.r, self.g, self.b)
    
    def __str__(self):
        """
        Representación en string del color.
        >>> str(Color(255, 0, 0)) # "Color(255, 0, 0, 255)"
        """
        return "Color({}, {}, {}, {})".format(self.r, self.g, self.b, self.a)

    def as_image(self, size=100):
        """
        Crea una imagen de un cuadrado del color.
        """
        if self.img is None or self.img.size != (size, size):
            self.img = Image.new('RGB', (size, size), self.as_rgb())
        
        return self.img
    
    def show(self, size=100):
        """
        Muestra el color como una imagen.
        Diseñado para ser usado en un entorno IPython.
        """
        display(self.as_image(size))

    def _repr_html_(self):
        """
        Muestra el color como un tag HTML.
        """
        img = self.as_image()
        return f'<div style="background-color: {self.as_hex()}; width: {img.size[0]}px; height: {img.size[1]}px;"></div>'
    

class Palette(list[Color]):
    """
    Representación de una paleta de colores.
    Tiene utilidades para convertirlo a distintos formatos y mostrarlo,
    especialmente para ser exportado como cmap de matplotlib.
    """
    img = None
    name = None

    @classmethod
    def from_hex(cls, hexes: list[str]):
        """
        Toma una lista de strings hexadecimales y crea una paleta de colores.
        Uso:
        >>> Palette.from_hex(['#FF0000', '#00FF00', '#0000FF'])
        """
        return cls(Color.from_hex(hex) for hex in hexes)
    
    @classmethod
    def from_rgba(cls, rgbas: list[tuple[int]]):
        """
        Toma una lista de tuplas RGBA y crea una paleta de colores.
        Uso:
        >>> Palette.from_rgba([(255, 0, 0, 255), (0, 255, 0, 255), (0, 0, 255, 255)])
        """
        return cls(Color.from_rgba(rgba) for rgba in rgbas)
    
    @classmethod
    def from_rgb(cls, rgbs: list[tuple[int]]):
        """
        Toma una lista de tuplas RGB y crea una paleta de colores.
        Uso:
        >>> Palette.from_rgb([(255, 0, 0), (0, 255, 0), (0, 0, 255)])
        """
        return cls(Color.from_rgb(rgb) for rgb in rgbs)
    
    @classmethod
    def from_float(cls, floats: list[tuple[float]]):
        """
        Toma una lista de tuplas de valores RGB en formato decimal y crea una paleta de colores.
        Uso:
        >>> Palette.from_float([(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)])
        """
        return cls(Color.from_float(*f) for f in floats)
    
    @classmethod
    def from_iterable(cls, iterables: list):
        """
        Toma una lista de objetos iterables y los convierte en una paleta de colores.
        Uso:
        >>> Palette.from_iterable([(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)])
        """
        return cls(Color.from_iterable(i) for i in iterables)

    @classmethod
    def from_dict(cls, d: dict):
        """
        Toma un diccionario de la forma
        {name: {name: hex, ...}}
        y crea una paleta de colores.

        Uso:
        >>> Palette.from_dict({'primary': {'red': '#FF0000', 'green': '#00FF00', 'blue': '#0000FF'}})
        """
        name = tuple(d.keys())[0]
        colors = [Color.from_dict({'value': v, 'name': k}) for k, v in d[name].items()]
        result = cls(colors)
        result.name = name
        return result
    
    def to_coolors(self):
        """
        Coolors es una página web para crear paletas de colores.
        Esta función convierte la paleta en un link para Coolors.
        Uso:
        >>> Palette.from_hex(['#FF0000', '#00FF00', '#0000FF']).to_coolors() # 'https://www.coolors.co/ff0000-00ff00-0000ff'
        """
        return "https://www.coolors.co/"+ "-".join(c.as_hex()[1:].lower() for c in self)
    
    @classmethod
    def from_coolors(cls, url):
        """
        Crea una paleta a partir de un link de Coolors.
        Uso:
        >>> Palette.from_coolors('https://www.coolors.co/ff0000-00ff00-0000ff') 
        >>> Palette(Color(255, 0, 0, 255), Color(0, 255, 0, 255), Color(0, 0, 255, 255))
        """
        if '/' in url:
            url = url.rsplit("/", 1)[-1]

        url = url.split("-")
        return cls.from_hex(x+"#" for x in url)
    
    def as_image(self, size=100):
        """
        Crea una imagen de la paleta.
        Ver `concat_images` y `plot_square`.
        """
        if self.img is None or self.img.size != (size * len(self), size):
            self.img = concat_images([c.as_image(size) for c in self])
        
        return self.img
    
    def as_dict(self, names: list[str]|bool = None) -> dict:
        """
        Crea un diccionario de la forma {name: hex} a partir de la paleta.
        Uso:
        >>> Palette.from_hex(['#FF0000', '#00FF00', '#0000FF']).as_dict()
        >>> {'0': '#FF0000', '1': '#00FF00', '2': '#0000FF'}
        """
        if not names:
            names = [str(i) for i,_ in enumerate(self)]

        if names is True:
            names = [c.name for c in self]
        
        return {name: color.as_hex() for name, color in zip(names, self)}

    def as_hexes(self) -> list[str]:
        """
        Convierte la paleta en una lista de strings hexadecimales.
        Uso:
        >>> Palette.from_hex(['#FF0000', '#00FF00', '#0000FF']).as_hexes()
        >>> ['#FF0000', '#00FF00', '#0000FF']
        """
        return [c.as_hex() for c in self]
    
    def to_cmap(self, names: list[str] = None):
        """
        Convierte la paleta en un colormap de matplotlib.
        """
        return dict_as_colormap(self.as_dict(names), name=self.name)        
    
    def show(self, size=100):
        """
        Muestra la paleta como una imagen.
        Diseñado para ser usado en un entorno IPython.
        """
        display(self.as_image(size))
    
    def _repr_html_(self):
        """
        Muestra la paleta como un tag HTML.
        """
        img = self.as_image()
        return pil_to_html(img)

    def __str__(self):
        """
        Representación en string de la paleta.
        """
        return "Palette({})".format(", ".join(str(c) for c in self))
    
    def __repr__(self):
        """
        Representación en string de la paleta.
        """
        return str(self)