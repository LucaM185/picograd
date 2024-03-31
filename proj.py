import matplotlib.pyplot as plt
from picograd import * 
import pickle

# Dataset loading 
with open('./mnist.pickle', 'rb') as f:
    dataset = pickle.load(f)

# Dataset preprocessing
X_train, y_train = dataset[:, 1:]/255, dataset[:, 0]
print(X_train.sum(-1))

# Model definition
batch_size = 128
classes = 10
fc0 = Linear(784, classes)
optim = SGD([fc0.weight, fc0.bias], lr=0.01)

# Training loop
lossi = []
acci = []
for i in range(0, 800):
    idx = np.random.randint(0, dataset.shape[0], (batch_size,))
    labels = y_train[idx].onehot(classes)

    n1 = fc0(X_train[idx])
    loss = ((n1.softmax() - labels)**2).sum()
    loss.numpy()
    loss.backward()
    optim.step()

    if i % 100 == 0: 
        print("Loss: ", round(loss.numpy()/batch_size, 2))
        print("Accuracy: ", round(((n1.argmax(-1).data == y_train[idx].data)).mean(), 2))
        print()

    loss.zero_grad()

    lossi.append(loss.numpy())
    acci.append(((n1.argmax(-1).data == y_train[idx].data)).mean())

# DRAW LOSS and ACCURACY in the same plot
plt.plot(np.array(lossi)/max(lossi), label='loss')
plt.plot(acci, label='accuracy')
plt.legend()
plt.show()
