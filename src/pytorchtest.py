import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
import matplotlib.pyplot as plt

# class 1
# x = torch.unsqueeze(torch.linspace(-1, 1, 100), dim=1) # x data (tensor), shape(100, 1)
# y = x.pow(2) + 0.2 * torch.randn(x.size()) # noisy y data (tensor), shape(100, 1)
# x, y = Variable(x), Variable(y)

# class 2
n_data = torch.ones(100, 2)


class Net(nn.Module):
    def __init__(self, n_features, n_hidden, n_output):
        super(Net, self).__init__()
        self.hidden = nn.Linear(n_features, n_hidden)
        self.predict = nn.Linear(n_hidden, n_output)

    def forward(self, x):
        x = F.relu(self.hidden(x))
        x = self.predict(x)
        return x


net = Net(1, 10, 1)

optimizer = torch.optim.SGD(net.parameters(), lr=0.1)
loss_fn = nn.MSELoss()

plt.ion()
plt.show()
epochs = 200
for epoch in range(epochs):
    prediction = net(x)

    loss = loss_fn(prediction, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 5 == 0:
        # print('Epoch: {}, loss: {:.4f}'.format(epoch, loss.item()))
        plt.cla()
        plt.scatter(x.data.numpy(), y.data.numpy())
        plt.plot(x.data.numpy(), prediction.data.numpy(), 'r-', lw=5)
        plt.text(0.5, 0, 'Loss=%.4f' % loss.item(), fontdict={'size': 20, 'color': 'red'})
        plt.pause(0.1)

plt.ioff()
plt.show()