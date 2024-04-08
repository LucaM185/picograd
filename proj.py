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
batch_size = 1024
epochs = 900
classes = 10
fc0 = Linear(784, 128)
fc5 = Linear(128, 128)
fc1 = Linear(128, classes)#, fc1, fc5
optim = SGD([fc0, fc1, fc5], lr=0.01, momentum=0.4)

# Training loop
lossi = []
acci = []
for i in range(epochs):
    idx = np.random.randint(0, dataset.shape[0], (batch_size,))
    labels = y_train[idx].onehot(classes)

    n1 = fc0(X_train[idx]).tanh()
    n5 = fc5(n1).tanh()
    n2 = fc1(n5)
    loss = ((n2.softmax() - labels)**2).sum()
    loss.numpy()
    loss.backward()
    optim.step()

    if i % (epochs//10) == 0: 
        print("Loss: ", round(loss.numpy()/batch_size, 2))
        print("Accuracy: ", round(((n2.argmax(-1).data == y_train[idx].data)).mean(), 2))
        print()

    optim.zero_grad()

    lossi.append(loss.numpy())
    acci.append(((n2.argmax(-1).data == y_train[idx].data)).mean())

# DRAW LOSS and ACCURACY in the same plot
plt.plot(np.array(lossi)/max(lossi), label='loss')
plt.plot(acci, label='accuracy')
plt.legend()
plt.show()

