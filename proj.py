from picograd import * 
import pickle

with open('./mnist.pickle', 'rb') as f:
    dataset = pickle.load(f)

fc0 = Linear(784, 10)

in_features, out_features = 784, 10
weight = Randn((in_features, out_features), requires_grad=True) 
bias = Randn((1, out_features), requires_grad=True) # double check the 1 in (1, out_features)


X_train, y_train = dataset[:, 1:]/255, dataset[:, 0]

print(X_train.sum(-1))

# optim = SGD([weight, bias], lr=0.001)
for i in range(2000):
    idx = np.random.randint(0, dataset.shape[0], (50,))

    pro = X_train[idx] @ weight + bias
    out = (pro).softmax()  # (5, 10) + (1, 10)
    # print(out.numpy().shape)
    # print(y_train[idx].onehot().shape)
    diff = out - y_train[idx].onehot(10) # (5, 10) 
    diffs = (diff)**2
    loss = (diffs).sum()
    loss.numpy()
    #print(loss.numpy())
    loss.backward()
    print(bias.grad)
    weight.data -= weight.grad * 0.001
    bias.data -= bias.grad * 0.001
    # optim.step()
    loss.zero_grad()


    # if i % 50 == 0: 
    #     print(weight.data)
    #     print(loss)






