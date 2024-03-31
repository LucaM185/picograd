from picograd import * 
import pickle

with open('./mnist.pickle', 'rb') as f:
    dataset = pickle.load(f)

# dataset = [(dataset[i]).data for i in range(dataset.shape[0]) if dataset[i, 0].data.item() in [0, 1]]
# dataset = Tensor(np.array(dataset))

print(dataset.shape)

classes = 10
fc0 = Linear(784, classes)

in_features, out_features = 784, classes
weight = Randn((in_features, out_features), requires_grad=True) 
bias = Randn((1, out_features), requires_grad=True) # double check the 1 in (1, out_features)


X_train, y_train = dataset[:, 1:]/255, dataset[:, 0]

print(X_train.sum(-1))
batch_size = 128

lossi = []
acci = []
optim = SGD([fc0.weight, fc0.bias], lr=0.03)
for i in range(0, 800):
    idx = np.random.randint(0, dataset.shape[0], (batch_size,))
    labels = y_train[idx].onehot(classes)

    pro = fc0(X_train[idx])# @ weight + bias
    # pro = X_train[idx] @ weight + bias
    out = (pro).softmax()  # (5, 10) + (1, 10)
    diff = out - labels # (5, 10) 
    diffs = (diff)**2
    loss = (diffs).sum()
    loss.numpy()
    loss.backward()
    optim.step()

    if i % 100 == 0: 
        print("Loss: ", round(loss.numpy()/batch_size, 2))
        print("Accuracy: ", round(((out.argmax(-1).data == y_train[idx].data)).mean(), 2))
        print()

    loss.zero_grad()


    lossi.append(loss.numpy())
    acci.append(((out.argmax(-1).data == y_train[idx].data)).mean())


import matplotlib.pyplot as plt

# DRAW LOSS and ACCURACY in the same plot
plt.plot(np.array(lossi)/max(lossi), label='loss')
plt.plot(acci, label='accuracy')
plt.legend()
plt.show()
