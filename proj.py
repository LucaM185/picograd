from picograd import * 
import pickle

with open('./mnist.pickle', 'rb') as f:
    dataset = pickle.load(f)

fc0 = Linear(784, 10)

in_features, out_features = 784, 10
weight = Randn((in_features, out_features), requires_grad=True) 
bias = Randn((1, out_features), requires_grad=True) # double check the 1 in (1, out_features)


X_train, y_train = dataset[:5, 1:]/255, dataset[:5, 0]

print(X_train.sum(-1))

optim = SGD([weight, bias], lr=0.0001)
for i in range(5):
    # bias.data += 10
    pro = X_train @ weight
    out = pro + bias  # (5, 10) + (1, 10)
    diff = out - y_train # (5, 10) 
    diffs = (diff)**2
    loss = (diffs).sum()

    print(loss.numpy())
    loss.backward()
    weight.data -= weight.grad * 0.0001
    bias.data -= bias.grad * 0.0001
    # optim.step()
    loss.zero_grad()






