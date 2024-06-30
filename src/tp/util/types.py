from typing import TypeVar, Iterable, Generic, Hashable
from dataclasses import dataclass, asdict
import warnings
import functools
from numpy import ndarray as numpy_array

T = TypeVar('T')
class _Lattice(Iterable[Iterable[T]], Generic[T]): 
   """
   Tipo genérico para tener una matriz de cualquier tipo de dato.
   """
   pass

# Esto está acá para que no chille el type-checker
Lattice = (_Lattice[T]
         | _Lattice[T,T]
         | list[list[T]]
         | numpy_array[T]
)

def Dataclass(**factory_kwargs):
    """
    Esto es un wrapper de 'dataclass', funciona como una fábrica de clases wrappeadas con dataclass.
    Los keyword-arguments van directo al decorador 'dataclass'.
    Además, suma métodos 'from_dict' y 'to_dict' que no tiene dataclass por defecto.

    Está diseñado para que las clases lo subclasseen.
    """
    return type('', (), {
        '__init_subclass__': lambda cls, **kwargs: dataclass(cls, **factory_kwargs, **kwargs),
        'from_dict': classmethod(lambda cls, d: cls(**d)),
        'to_dict': lambda self: asdict(self)
    })


class memoized(object):
   """
   Un decorador que cachea el resultado de una función cada vez que es llamada.
   Si se llama más tarde con los mismos argumentos, se devuelve el valor en caché
   """

   def __init__(self, f, debug=False):
      self.func = f
      self.cache = {}
      self.debug = debug

   def __debug_print__(self, *args, **kwargs):
      if self.debug:
         print(*args, **kwargs)

   def __call__(self, *args, **kwargs):
      force = kwargs.get('force', False)

      if force:
         _w = f"Forcing cache update for '{self.func.__qualname__}'"
         warnings.warn(_w, Warning)
      
      if force or (isinstance(args, Hashable) and args not in self.cache):
         self.__debug_print__(f"Caching result for '{self.func.__qualname__}, current cache size is {len(self.cache)}")
         value = self.func(*args)
         self.cache[args] = value
         return value

      if not isinstance(args, Hashable):
         self.__debug_print__(f"Unhashable arguments, not caching result for '{self.func.__qualname__}'")
         return self.func(*args)
      
      if args in self.cache:
         self.__debug_print__(f"Returning cached result for '{self.func.__qualname__}'")
         return self.cache[args]
      
   def __repr__(self):
      return self.func.__doc__
   
   def __get__(self, obj, objtype): # Para que funcione con instance methods
      return functools.partial(self.__call__, obj)
   
def memoize(f=None, debug=False):
   """
   Decorador para memoizar funciones (ver `memoized`).
   """

   def wrap(f):
      return memoized(f, debug=debug)
   
   # Nos están llamando como @memoize o @memoize()?
   if f is None: # Nos están llamando como @memoize()
      return wrap

   # Nos están llamando como @memoize
   return wrap(f)