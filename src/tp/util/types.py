from typing import TypeVar, Iterable, Generic

T = TypeVar('T')
class Lattice(Iterable[Iterable[T]], Generic[T]): ...