from IPython.display import display
from PIL import Image
import numpy as np
from io import BytesIO
from base64 import b64encode
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

def dict_as_colormap(color_dict, name=None):
    if not name:
        name = '-'.join(x for x in color_dict.values())
    
    keys = sorted(color_dict.keys())
    keys_nums = [i for i,_ in enumerate(keys)]


    colors = [color_dict[key] for key in keys]
    cmap = mcolors.ListedColormap(colors, name)
    
    
    bounds = np.array(keys_nums + [max(keys_nums)+1])
    norm = mcolors.BoundaryNorm(bounds, cmap.N)
    
    return cmap, norm


def plot_square(l: int, color: tuple[np.floating]):
    if isinstance(color, str):
        return Image.new('RGB', (l,l), color)
    return Image.new('RGB', (l, l), tuple(int(255 * c) for c in color[:3]))

def concat_images(images) -> Image:
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
    squares = [plot_square(l, c) for c in colors]
    return concat_images(squares)

def pil_to_base64(image):
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    img_bytes = buffer.getvalue()
    img_base64 = b64encode(img_bytes)
    img_base64_str = img_base64.decode('utf-8')
    return img_base64_str

def pil_to_html(image):
    img_base64_str = pil_to_base64(image)
    return f'<div><img src="data:image/png;base64,{img_base64_str}"></div>'

import pickle
def cmap_norm_to_pickle(cmap, norm, filename):
    with open(filename, 'wb') as f:
        pickle.dump((cmap, norm), f)

def pickle_to_cmap_norm(filename) -> tuple:
    with open(filename, 'rb') as f:
        return pickle.load(f)

class Color:
    img = None
    name = None

    def __init__(self, r: int, g: int, b: int, a: int = 255):
        self.r = r
        self.g = g
        self.b = b
        self.a = a

    @classmethod
    def from_hex(cls, hex: str):
        hex = hex.lstrip("#")
        return cls(*[int(hex[i:i+2], 16) for i in (0, 2, 4)])
    
    @classmethod
    def from_rgba(cls, rgba: tuple[int]):
        return cls(*rgba)

    @classmethod
    def from_rgb(cls, rgb: tuple[int]):
        return cls(*rgb, 255)
    
    @classmethod
    def from_float(cls, r: float, g: float, b: float, a: float = 1.0):
        return cls(*[int(255 * c) for c in (r, g, b, a)])
    
    @classmethod
    def from_iterable(cls, iterable):
        if all(isinstance(i, float) for i in iterable):
            return cls.from_float(*iterable)
        return cls(*iterable)
    
    @classmethod
    def from_dict(cls, d):
        # expects a dict with [value: hex, <optional name>: str]
        result = cls.from_hex(d['value'])
        result.name = d.get('name')
        return result


    @classmethod
    def from_records(cls, d, **kwargs):
        # use kwargs to access the dict
        # for example: value = 'color', name = 'title'
        return cls.from_dict({'value': d[kwargs['value']], 'name': d.get(kwargs.get('name'))})
    
    def as_hex(self):
        return "#{:02x}{:02x}{:02x}".format(self.r, self.g, self.b).upper()

    def as_rgba(self):
        return (self.r, self.g, self.b, self.a)
    
    def as_rgb(self):
        return (self.r, self.g, self.b)
    
    def __str__(self):
        return "Color({}, {}, {}, {})".format(self.r, self.g, self.b, self.a)

    def as_image(self, size=100):
        if self.img is None or self.img.size != (size, size):
            self.img = Image.new('RGB', (size, size), self.as_rgb())
        
        return self.img
    
    def show(self, size=100):
        display(self.as_image(size))

    def _repr_html_(self):
        img = self.as_image()
        return f'<div style="background-color: {self.as_hex()}; width: {img.size[0]}px; height: {img.size[1]}px;"></div>'
    

class Palette(list[Color]):
    img = None
    name = None

    @classmethod
    def from_hex(cls, hexes: list[str]):
        return cls(Color.from_hex(hex) for hex in hexes)
    
    @classmethod
    def from_rgba(cls, rgbas: list[tuple[int]]):
        return cls(Color.from_rgba(rgba) for rgba in rgbas)
    
    @classmethod
    def from_rgb(cls, rgbs: list[tuple[int]]):
        return cls(Color.from_rgb(rgb) for rgb in rgbs)
    
    @classmethod
    def from_float(cls, floats: list[tuple[float]]):
        return cls(Color.from_float(*f) for f in floats)
    
    @classmethod
    def from_iterable(cls, iterables: list):
        return cls(Color.from_iterable(i) for i in iterables)

    @classmethod
    def from_dict(cls, d: dict):
        # expects a dict with {name: {name: hex, ...}}
        name = tuple(d.keys())[0]
        colors = [Color.from_dict({'value': v, 'name': k}) for k, v in d[name].items()]
        result = cls(colors)
        result.name = name
        return result
    
    def to_coolors(self):
        return "https://www.coolors.co/"+ "-".join(c.as_hex()[1:].lower() for c in self)
    
    @classmethod
    def from_coolors(cls, url):
        if '/' in url:
            url = url.rsplit("/", 1)[-1]

        url = url.split("-")
        return cls.from_hex(x+"#" for x in url)
    
    def as_image(self, size=100):
        if self.img is None or self.img.size != (size * len(self), size):
            self.img = concat_images([c.as_image(size) for c in self])
        
        return self.img
    
    def as_dict(self, names: list[str]|bool = None) -> dict:
        if not names:
            names = [str(i) for i,_ in enumerate(self)]

        if names is True:
            names = [c.name for c in self]
        
        return {name: color.as_hex() for name, color in zip(names, self)}

    def as_hexes(self) -> list[str]:
        return [c.as_hex() for c in self]
    
    def to_cmap(self, names: list[str] = None):
        return dict_as_colormap(self.as_dict(names), name=self.name)        
    
    def show(self, size=100):
        display(self.as_image(size))
    
    def _repr_html_(self):
        img = self.as_image()
        return pil_to_html(img)

    def __str__(self):
        return "Palette({})".format(", ".join(str(c) for c in self))
    
    def __repr__(self):
        return str(self)