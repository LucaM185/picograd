import numpy as np
import tqdm


# Notes
# - Enforce lazyness
# - Enforce types
# - Make tests before features

class Tensor():
    def __init__(self, value, requires_grad=False) -> None:
        self.realized = True
        self.value = value
        self.shape = value.shape

    def realize(self): return self
    def numpy(self): return self.value if self.realized else self.realize().value

    def __add__(self, other): return LazyTracker(self, other, np.add)
    def __sub__(self, other): return LazyTracker(self, other, np.subtract)
    def __mul__(self, other): return LazyTracker(self, other, np.multiply)
    def __truediv__(self, other): return LazyTracker(self, other, np.divide)
    def __pow__(self, other): return LazyTracker(self, other, np.power)
    def __matmul__(self, other): return LazyTracker(self, other, np.matmul)
    def __neg__(self): return LazyTracker(self, self, np.negative)
    def __abs__(self): return LazyTracker(self, self, np.abs)

    def __str__(self) -> str: return f"Tensor: {self.value}"
        

class LazyTracker(Tensor):
    def __init__(self, a, b, f) -> None:
        self.realized = False
        self.a = a
        self.b = b
        self.f = f

    def __str__(self) -> str:
        return "(LazyTracker: " + str(self.a) + "  " +  str(self.b) + " " + str(self.f) + ")"

    def realize(self):
        self.value = self.f(self.a.realize().value, self.b.realize().value)
        return self


def test(tensor: Tensor) -> None:
    pass

