import tinygrad
import picograd
import numpy as np

def test(a, b):
    results = []
    
    results.append(a + b)
    results.append(a - b)
    results.append(a * b)
    results.append(a / b)
    results.append(a ** b)
    results.append(a + b - (a*b) + (a/b) - (a**b))
    
    return [result.numpy() for result in results]

def close_enough(a, b):
    try: return (np.logical_or(np.abs(a - b) < 1e-6, np.logical_and(np.isnan(a), np.isnan(b)))).all()
    except: return False

a, b = np.random.randn(3), np.random.randn(3)
at, bt = tinygrad.Tensor(a), tinygrad.Tensor(b)
ap, bp = picograd.Tensor(a), picograd.Tensor(b)

for a, b in zip(test(at, bt), test(ap, bp)):
    print([a, b])

print([close_enough(a, b) for a, b in zip(test(at, bt), test(ap, bp))])

