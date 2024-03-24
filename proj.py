from picograd import * 
import pickle

with open('./mnist.pickle', 'rb') as f:
    dataset = pickle.load(f)

fc0 = Linear(784, 10)

X_train, y_train = dataset[:20, 1:]/255, dataset[:20, 0]
out = fc0(X_train)

loss = ((out - y_train)**2).sum()
print(loss.numpy())
loss.backward()

print(fc0.weight.grad)
