import torch
import matplotlib.pyplot as plt


def generate_sample():
    w, b = 10., 2.
    x = torch.empty(200)
    x.uniform_(-10., 10.)
    y = x * w + b
    return x, y


x, y = generate_sample()


class LinearModule():
    def __init__(self):
        self.w = torch.nn.Parameter(torch.Tensor([0.]))
        self.b = torch.nn.Parameter(torch.Tensor([0.]))

    def __call__(self, x):
        return x * self.w + self.b


module = LinearModule()

optimizer = torch.optim.SGD([module.w, module.b], lr=0.01)
loss_fn = torch.nn.MSELoss()

epoch = 10
for i in range(epoch):
    y_hat = module(x)
    loss = loss_fn(y_hat, y)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    print(f'epoch:{i}   loss:{loss}  w:{module.w.item()}  b:{module.b.item()}')
    plt.cla()
    plt.plot(x, y)
    plt.plot(x, y_hat.data)
    plt.pause(1.0)
