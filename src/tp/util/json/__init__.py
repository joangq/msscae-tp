# from fundar.utils import load_from_str_or_buf
from functools import wraps
import json as json_
import io, os

def load_from_str_or_buf(input_data) -> io.BytesIO:
    match input_data:
        case string if isinstance(input_data, str):
            if not os.path.exists(input_data):
                raise FileNotFoundError(f'File {string} not found.')
            if not os.path.isfile(string):
                raise ValueError("Input is a folder, not a file.")
            
            with open(string, 'rb') as file:
                return io.BytesIO(file.read())
            
        case buffer if isinstance(input_data, (io.BytesIO, io.StringIO)):
            return buffer
        case _:
            raise TypeError("Unsupported input type. Please provide a valid file path or buffer.")

@wraps(json_.load)
def load(path_or_buf, **kwargs):
    return json_.load(fp=load_from_str_or_buf(path_or_buf), **kwargs)

from numpy import (
    ndarray as numpy_array,
    floating as numpy_floating,
    integer as numpy_int
)

class JsonEncoder(json_.JSONEncoder):
    def default(self, o: object):
        if isinstance(o, numpy_array):
            return list(o)
        if isinstance(o, numpy_floating):
            return float(o)
        if isinstance(o, numpy_int):
            return int(o)

@wraps(json_.dump)
def dump(obj, path_or_buf, **kwargs):
    if isinstance(path_or_buf, str):
        with open(path_or_buf, 'w', encoding='utf-8') as fp:
            return dump(obj, path_or_buf=fp, **kwargs)
    
    return json_.dump(obj=obj, fp=path_or_buf, cls=JsonEncoder, **kwargs)

def __getattr__(x):
    return getattr(json_, x)