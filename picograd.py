import numpy as np

def mean(mylist): return sum(mylist) / len(mylist) if len(mylist) > 0 else 0
def prod(a): return np.array(a).prod() 

### SHAPETRACKER ###

class ShapeTracker:
    def __init__(self, shape) -> None:
        self.shape = shape 
        self.permutes = None
        self.stride = [1]
        self.pad = [0]


    def __eq__(self, other):  # Incomplete
        return mean([a == b for a, b in zip(self.shape, other.shape)]) == 1 

    def __str__(self) -> str:
        return "t" + str(self.shape)

    def __getitem__(self, index): 
        if isinstance(index, int): return self.shape[index]
        if isinstance(index, slice): return self.shape[index]
        # if isinstance(index, tuple): return ShapeTracker([self.shape[i] for i in index])
        # Also handle tensor

    def pop(self, axis):
        if axis is not None: self.shape = tuple([i for n, i in enumerate(self.shape) if n != axis%len(self.shape)]) 
        else: self.shape = (1,) 

    def copy(self): return ShapeTracker(self.shape)
    def flat(self): return prod(self.shape)

### LAZYBUFFER ###

class LazyBuffer:
    def __init__(self) -> None:
        self.buffer = []
        self.grad = 0
        # print(type(self).__name__)
    
    def numpy(self):
        return self.forward()
    
    def debug(self):
        print(self.buffer)
    
    def __add__(self, other): return Add(self, other)
    def __radd__(self, other): return Add(other, self)
    def __mul__(self, other): return Mul(self, other)
    def __truediv__(self, other): return Div(self, other)
    def __rtruediv__(self, other): return Div(other, self) # Really untested
    def __sub__(self, other): return Add(self, -other)
    def __matmul__(self, other): return MatMul(self, other)
    def __neg__(self): return Mul(self, -1)
    def __pow__(self, exp): return Pow(self, exp)
    def __exp__(self): return Exp(self)

    def sum(self, axis=None): return Sum(self, axis)
    def mean(self, axis=None): return Sum(self, axis) * (1 / self.shapeTrack.shape[axis])  # Untested
    def onehot(self, dict_size=None): return OneHot(self, dict_size) # NOT LAZY
    def softmax(self, axis=-1): return softmax(self, axis=axis)  # Untested
    def max(self, axis=None): return Max(self, axis)  # Untested

    def __str__(self): return self.numpy().__str__()

    def shapeTrack(self, shape):
        self.shapeTrack = ShapeTracker(shape)

### TENSOR ###

class Tensor(LazyBuffer):  # Tensor is just lazybuffer that contains data 
    def __init__(self, value, shape=None, requires_grad=False) -> None:
        self.data = np.array(value)
        self.shape = ShapeTracker(self.data.shape) if shape is None else shape # shape is a shapeTracker
        self.grad = 0

    def forward(self):
        return self.data
    def backward(self, grad):  # Leaf tensors don't need to do anything
        self.grad += grad

    def numpy(self):
        return self.data

    def __getitem__(self, index):   # THIS IS NOT LAZY!!
        if isinstance(index, slice):
            start, stop, step = index.indices(len(self.data))
            return Tensor(self.data[start:stop:step])
        else:
            return Tensor(self.data[index])


### OPERATION TYPES ###

class Unary(LazyBuffer):
    def __init__(self, a) -> None:
        super().__init__()
        self.a = Tensor(a) if not isinstance(a, LazyBuffer) else a
        self.shape = self.a.shape.copy()

class Binary(LazyBuffer):
    def __init__(self, a, b) -> None:
        super().__init__()
        self.a = Tensor(a) if not isinstance(a, LazyBuffer) else a
        self.b = Tensor(b) if not isinstance(b, LazyBuffer) else b
        self.shape = self.a.shape.copy()  
        if self.a.shape.flat() != self.b.shape.flat() and self.a.shape.shape == (self.b.shape.shape + (self.a.shape.shape[-1],)): self.b = Adapt(self.b, self.a)  # Untested
        # assert self.a.shape == self.b.shape, f"Shapes {self.a.shape} and {self.b.shape} are not compatible" 
        # This ^ doesnt take b scalars into account

class Reduce(LazyBuffer):
    def __init__(self, a, axis=None) -> None:
        super().__init__()
        self.grad = 0
        self.a = Tensor(a) if not isinstance(a, LazyBuffer) else a  # Deal with shapetrack
        self.shape = self.a.shape.copy()
        self.axis = axis
        self.shape.pop(axis)
        # self.shape = 1 if axis is None else (elm for i, elm in enumerate(self.a.shape) if i != axis)

class Broadcast(LazyBuffer):
    def __init__(self, a, b) -> None:
        super().__init__()
        self.a = Tensor(a) if not isinstance(a, LazyBuffer) else a
        self.b = Tensor(b) if not isinstance(b, LazyBuffer) else b
        self.shape = ShapeTracker(self.a.shape.shape + (self.b.forward(),))

### OPERATIONS ###

def softmax(tensor, axis=-1): # safe softmax
    # exp = Exp(tensor - tensor.max())
    return Exp(tensor - tensor.max()) / (Exp(tensor - tensor.max()).sum(axis=axis) )

def OneHot(tensor, dict_size):
    if dict_size is None: dict_size = int(tensor.data.max()) + 1
    result = np.zeros((tensor.shape[0], dict_size))
    result[np.arange(tensor.shape[0]), np.int32(tensor.data)] = 1
    return Tensor(result)    

class Exp(Unary):
    def forward(self):  # Exp
        self.data = np.exp(self.a.forward())
        return self.data

    def backward(self, grad=1):
        self.grad += grad
        self.a.backward(grad * self.data)


class Div(Binary):
    def forward(self):  # a / b
        self.data = self.a.forward() / self.b.forward()
        return self.data

    def backward(self, grad=1):
        self.grad += grad
        self.a.backward(grad / self.b.data)
        self.b.backward(grad * -self.a.data / (self.b.data ** 2)) 

class Pow(Binary):
    def forward(self):  # a ** b
        self.data = self.a.forward() ** self.b.forward() 
        return self.data

    def backward(self, grad=1):
        self.grad += grad
        self.a.backward(grad * self.b.data * self.a.data ** (self.b.data - 1))
        # I'm not calculating the derivative on b... If you care do it yourself
        
class Add(Binary):
    def forward(self):  # a + b
        self.data = self.a.forward() + self.b.forward()
        return self.data

    def backward(self, grad=1):
        self.grad += grad
        self.a.backward(grad)
        self.b.backward(grad)

class Mul(Binary):
    def forward(self):
        self.data = self.a.forward() * self.b.forward()
        return self.data

    def backward(self, grad=1):
        self.grad += grad
        self.a.backward(grad * self.b.data)
        self.b.backward(grad * self.a.data)

class MatMul(Binary):
    def __init__(self, a, b) -> None:
        super().__init__(a, b)
        self.shape = ShapeTracker(self.a.shape.shape[:-1] + self.b.shape.shape[1:])

    def forward(self):
        self.data = self.a.forward() @ self.b.forward()
        return self.data

    def backward(self, grad=1):
        self.grad += grad
        self.a.backward(grad @ self.b.data.T)
        self.b.backward(self.a.data.T @ grad)


class Sum(Reduce): 
    def forward(self):
        self.data = np.sum(self.a.forward(), axis=self.axis)
        return self.data

    def backward(self, grad=Tensor((1,))):
        self.grad += grad
        self.a.backward(Adapt(self.grad, self.a).numpy())

class Mean(Reduce):  # Untested
    def forward(self):
        self.data = np.mean(self.a.forward(), axis=self.axis)
        return self.data

    def backward(self, grad=1):
        self.grad += grad
        self.a.backward(grad * np.ones(self.a.shape) / self.a.shapeTrack.shape[self.axis])

class Max(Reduce): 
    def forward(self):
        self.data = np.max(self.a.forward(), axis=self.axis)
        return self.data
    
    def backward(self, grad=1):  # WTF is the gradient of max? Definetly untested
        self.grad += grad
        self.a.backward(grad * (self.a.data == self.data))

class Adapt(Broadcast):
    def forward(self):
        # given a tensor and b shape, adapt a to b
        self.original_shape = self.a.shape
        times_smaller = self.b.shape.flat() // self.a.shape.flat()
        self.data = np.repeat(self.a.forward(), times_smaller).reshape(*self.b.shape)
        return self.data
    
    def backward(self, grad=1):
        self.grad += grad
        self.a.backward(np.sum(grad, axis=-1).reshape(self.original_shape))
        

class Rand(Tensor):
    def __init__(self, shape, requires_grad=False) -> None:
        self.data = np.random.rand(*shape)
        super().__init__(self.data, requires_grad=requires_grad)

class Randn(Tensor):
    def __init__(self, shape, requires_grad=False) -> None:
        self.data = np.random.randn(*shape)
        super().__init__(self.data, requires_grad=requires_grad)

class ReadCSV(Tensor):
    def __init__(self, path) -> None:
        # assume first line is the headerRepe
        self.data = np.genfromtxt(path, delimiter=',', skip_header=1)
        super().__init__(self.data)

### MODULES ###

class Module():
    pass

class Linear(Module):
    def __init__(self, in_features, out_features) -> None:
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Randn((in_features, out_features), requires_grad=True) * (1 / np.sqrt(in_features)) 
        self.bias = Randn((1, out_features), requires_grad=True) * (1 / np.sqrt(in_features)) # double check the 1 in (1, out_features)

    def __call__(self, x):
        return x @ self.weight + self.bias

    # def backward(self, grad):
    #     self.weight.grad += self.x.T @ grad
    #     self.bias.grad += grad.sum(axis=0)
    #     return grad @ self.weight.T
