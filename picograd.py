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

    def __len__(self): return len(self.shape)

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
    def __lt__(self, other): return LessThan(self, other)  # Half lazy, other is a Tensor
    def __gt__(self, other): return GreaterThan(self, other)  # Half lazy, other is a Tensor


    def sum(self, axis=None): return Sum(self, axis)
    def mean(self, axis=None): return Sum(self, axis) * (1 / self.shapeTrack.shape[axis])  # Untested
    def onehot(self, dict_size=None): return OneHot(self, dict_size) # NOT LAZY
    def softmax(self, axis=-1): return softmax(self, axis=axis)  
    def max(self, axis=None): return Max(self, axis)  
    def argmax(self, axis=None): return ArgMax(self, axis)  # NOT LAZY
    def tanh(self): return tanh(self) 
    def relu(self): return self * (self > 0) 

    def __str__(self): return self.numpy().__str__()

    def shapeTrack(self, shape):
        self.shapeTrack = ShapeTracker(shape)
 
    def zero_grad(self, scale=0):
        self.grad *= scale

    def __getitem__(self, index):
        if isinstance(index, LazyBuffer):
            return Select(self, index)
        if isinstance(index, slice):
            start, stop, step = index.indices(len(self.data))
            return Tensor(self.data[start:stop:step])
        else:
            return Tensor(self.data[index])

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

### OPERATION TYPES ###

class Unary(LazyBuffer):
    def __init__(self, a) -> None:
        super().__init__()
        self.a = Tensor(a) if not isinstance(a, LazyBuffer) else a
        self.shape = self.a.shape.copy()

    def zero_grad(self, scale=0):
        self.grad *= scale
        self.a.zero_grad(scale)

class Binary(LazyBuffer):
    def __init__(self, a, b, skip_reshapes=False) -> None:
        super().__init__()
        self.a = Tensor(a) if not isinstance(a, LazyBuffer) else a
        self.b = Tensor(b) if not isinstance(b, LazyBuffer) else b
        self.shape = self.a.shape.copy()  
        # Refactor this (and test it) 
        if not skip_reshapes:
            if self.a.shape.flat() > self.b.shape.flat():
                self.b = Adapt(self.b, self.a)  
                self.b.shape.shape = self.a.shape.shape
            if self.a.shape.flat() < self.b.shape.flat():
                self.a = Adapt(self.a, self.b)  
                self.a.shape.shape = self.b.shape.shape
        # This ^ doesnt take b scalars into account

    def zero_grad(self, scale=0):
        self.grad *= scale
        self.a.zero_grad(scale)
        self.b.zero_grad(scale)
        

class Reduce(LazyBuffer):
    def __init__(self, a, axis=None) -> None:
        super().__init__()
        self.grad = 0
        self.a = Tensor(a) if not isinstance(a, LazyBuffer) else a  # Deal with shapetrack
        self.shape = self.a.shape.copy()
        self.axis = axis
        self.shape.pop(axis)
        # self.shape = 1 if axis is None else (elm for i, elm in enumerate(self.a.shape) if i != axis)

    def zero_grad(self, scale=0):
        self.grad *= scale
        self.a.zero_grad(scale)

class Broadcast(LazyBuffer):
    def __init__(self, a, b) -> None:
        super().__init__()
        self.a = Tensor(a) if not isinstance(a, LazyBuffer) else a
        self.b = Tensor(b) if not isinstance(b, LazyBuffer) else b
        self.shape = ShapeTracker(self.a.shape.shape)

    def zero_grad(self, scale=0):
        self.grad *= scale
        self.a.zero_grad(scale)

### OPERATIONS ###
        
def tanh(tensor): 
    return (Exp(tensor) - Exp(-tensor)) / (Exp(tensor) + Exp(-tensor))

def softmax(tensor, axis=-1): # safe softmax
    return Exp(tensor - tensor.max()) / (Exp(tensor - tensor.max()).sum(axis=axis) )

def ArgMax(tensor, axis=None):
    return Tensor(np.argmax(tensor.data, axis=axis))

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

class LessThan(Binary):
    def forward(self):  # a < b
        self.data = self.a.forward() < self.b.forward()
        return self.data

    def backward(self, grad=1):
        self.grad += grad
        self.a.backward(grad * self.data)  # b is not derived, dont care for now

class GreaterThan(Binary):
    def forward(self):  # a > b
        self.data = self.a.forward() > self.b.forward()
        return self.data

    def backward(self, grad=1):
        self.grad += grad
        self.a.backward(grad * self.data)  # b is not derived, dont care for now

class Select(Binary):
    def forward(self):
        self.data = self.a.forward()[self.b.forward()]
        return self.data
    
    def backward(self, grad=1):
        self.grad += grad
        self.a.backward(grad * self.b.data)

class Div(Binary):
    def forward(self):  # a / b
        self.data = self.a.forward() / self.b.forward()
        return self.data

    def backward(self, grad=1):
        self.grad += grad
        self.a.backward(grad / (self.b.data + 1e-8))
        self.b.backward(grad * -self.a.data / (self.b.data ** 2 + 1e-8)) 

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
        super().__init__(a, b, skip_reshapes=True)
        self.shape = ShapeTracker(self.a.shape.shape[:-1] + self.b.shape.shape[1:])

    def forward(self):
        self.data = self.a.forward() @ self.b.forward()
        return self.data

    def backward(self, grad=0):
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
    
    def backward(self, grad=1):  # untested
        self.grad += grad
        self.a.backward(grad * (self.a.data == self.data))

class Adapt(Broadcast):
    def forward(self):
        # given a tensor and b shape, adapt a to b
        self.original_shape = self.a.shape
        self.original_flat = self.a.shape.flat()
        times_smaller = self.b.shape.flat() // self.a.shape.flat()
        if len(self.a.shape) != len(self.b.shape): self.repeated_axis = -1
        else: self.repeated_axis = [n for n, z in enumerate(zip(self.a.shape, self.b.shape)) if z[0] != z[1]][-1]
        self.data = np.repeat(self.a.forward(), times_smaller, axis=self.repeated_axis).reshape(*self.b.shape)
        return self.data
    
    def backward(self, grad=1):
        self.grad += grad
        # print(self.grad.shape, self.a.shape, self.b.shape)
        if self.original_flat == 1: self.repeated_axis = None
        self.a.backward(np.sum(grad, axis=self.repeated_axis).reshape(self.original_shape))


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
        self.weight = Randn((in_features, out_features), requires_grad=True) #* (1 / np.sqrt(in_features)) 
        self.bias = Randn((1, out_features), requires_grad=True) #* (1 / np.sqrt(in_features)) # double check the 1 in (1, out_features)

    def __call__(self, x):
        return x @ self.weight + self.bias

    def params(self):
        return [self.weight, self.bias]

class Optimizer(Module):
    def __init__(self, parameters, lr=0.01) -> None:
        self.lr = lr
        self.parameters = []
        for p in parameters:  # this could be written better prob
            if isinstance(p, LazyBuffer):
                self.parameters.append(p)
            elif isinstance(p, Module):
                for pp in p.params():
                    self.parameters.append(pp)

class SGD(Optimizer):
    def __init__(self, parameters, lr=0.01, momentum=0) -> None:
        super().__init__(parameters, lr)
        self.momentum = momentum

    def step(self):
        for param in self.parameters:
            param.data -= param.grad * self.lr

    def zero_grad(self):
        for param in self.parameters:
            param.zero_grad(self.momentum)

class Adam(Optimizer):
    def __init__(self, parameters, lr=0.01, beta1=0.9, beta2=0.999, epsilon=1e-8) -> None:
        super().__init__(parameters, lr)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        for param in self.parameters:
            param.first_deriv = 0
            param.second_deriv = 0

    def step(self):
        for param in self.parameters:
            param.first_deriv = self.beta1 * param.first_deriv + (1 - self.beta1) * param.grad
            param.second_deriv = self.beta2 * param.second_deriv + (1 - self.beta2) * (param.grad**2)
            param.data -= self.lr * param.first_deriv / (np.sqrt(param.second_deriv) + self.epsilon)
    
    def zero_grad(self):
        for param in self.parameters:
            param.zero_grad()
