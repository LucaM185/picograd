import tinygrad
import picograd
import numpy as np

def test_ops(a, b):
    results = []
    
    results.append(a + b)
    results.append(a - b)
    results.append(a * b)
    results.append(a / b)
    results.append(a ** b)
    results.append(a + b - (a*b) + (a/b) - (a**b))
    
    return [result.numpy() for result in results]

def test_backward(a, b):
    loss = (a + b - (a*b) + (a/b) - (a**b)).softmax().sum()
    loss.backward()
    return a.grad, b.grad

def close_enough(a, b):
    try: return (np.logical_or(np.abs(a - b) < 1e-3, np.logical_and(np.isnan(a), np.isnan(b)))).all()
    except: return False

def test():
    # Testing pure ops
    a, b = np.random.randn(100), np.random.randn(100)
    at, bt = tinygrad.Tensor(a), tinygrad.Tensor(b)
    ap, bp = picograd.Tensor(a), picograd.Tensor(b)
    print([close_enough(a, b) for a, b in zip(test_ops(at, bt), test_ops(ap, bp))])

    # Testing ops with broadcasting
    a, b = np.random.randn(100, 100), np.random.randn(100)
    at, bt = tinygrad.Tensor(a), tinygrad.Tensor(b)
    ap, bp = picograd.Tensor(a), picograd.Tensor(b)
    print(at == ap.numpy())
    print([close_enough(a, b) for a, b in zip(test_ops(at, bt), test_ops(ap, bp))])

    # Testing backward
    a, b = np.random.randn(100), np.random.randn(100)
    at, bt = tinygrad.Tensor(a), tinygrad.Tensor(b)
    ap, bp = picograd.Tensor(a), picograd.Tensor(b)
    print([close_enough(a, b) for a, b in zip(test_backward(at, bt), test_backward(ap, bp))])



test()


