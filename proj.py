import matplotlib.pyplot as plt
from picograd import * 

# Dataset loading 
dataset = ReadCSV("tiny.csv")

# Dataset preprocessing
X_train, y_train = dataset[:, 1:]/255, dataset[:, 0]
print(X_train.sum(-1))

# Model definition
batch_size = 64
epochs = 100
classes = 10
fc0 = Linear(784, 128)
fc1 = Linear(128, 128)
fc2 = Linear(128, classes)

optim = SGD([fc0, fc1, fc2], lr=1e-3, momentum=0.9, clip=1.0)
# optim = Adam([fc0, fc1, fc2], lr=1e-4, clip=1.0)

# Training loop
lossi = []
acci = []
for i in range(epochs):
    idx = np.random.randint(0, dataset.shape[0], (batch_size,))
    labels = y_train[idx].onehot(classes)

    n0 = X_train[idx]
    n1 = fc0(n0).relu()
    n2 = fc1(n1).relu()
    n3 = fc2(n2)

    loss = ((n3.softmax() - labels)**2).sum()
    loss.numpy()
    loss.backward()
    optim.step()

    if i % (epochs//10) == 0: 
        print("Loss: ", round(loss.numpy()/batch_size, 2))
        print("Accuracy: ", round(((n3.argmax(-1).data == y_train[idx].data)).mean(), 2))
        print()

    optim.zero_grad()

    lossi.append(loss.numpy()/batch_size)
    acci.append(((n3.argmax(-1).data == y_train[idx].data)).mean())

# DRAW LOSS and ACCURACY in the same plot
plt.plot(np.array(lossi)/max(lossi), label='loss')
plt.plot(acci, label='accuracy')
plt.legend()
plt.show()

