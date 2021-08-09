import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
import matplotlib.pyplot as plt
import torchvision
import torch.utils.data as Data

EPOCH = 1
BATCH_SIZE = 64
TIME_STEP = 28  # RNN time step / image height
INPUT_SIZE = 28  # RNN input size / image width
LR = 0.001

DOWNLOAD_MNIST = False






class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()
        self.rnn = nn.LSTM(
            input_size=INPUT_SIZE,
            hidden_size=64,
            num_layers=1,
            batch_first=True,  # (batch_size 在第一个维度
        )
        self.out = nn.Linear(64, 10)

    def forward(self, x):
        r_out, (h_n, h_c) = self.rnn(x, None)  # x (batch, time_step, input_size)
        out = self.out(r_out[:, -1, :])  # (batch, time step, input)
        return out


if __name__ == '__main__':
    train_data = torchvision.datasets.MNIST('./mnist', train=True,
                                            transform=torchvision.transforms.ToTensor(),  # (0, 1)
                                            download=DOWNLOAD_MNIST)

    train_loader = Data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

    test_data = torchvision.datasets.MNIST('./mnist', train=False)
    # !!!!!对比test_x数据size不同的原因
    test_x = Variable(test_data.test_data).type(torch.FloatTensor)[:2000] / 255.
    test_y = test_data.test_labels[:2000]

    rnn = RNN()
    print(rnn)

    optimizer = torch.optim.Adam(rnn.parameters(), lr=LR)
    loss_func = nn.CrossEntropyLoss()

    for epoch in range(EPOCH):
        for step, (batch_x, batch_y) in enumerate(train_loader):
            b_x = Variable(batch_x.view(-1, 28, 28))  # reshape x to (batch_size, time step, input_size)
            b_y = Variable(batch_y)
            output = rnn(b_x)

            loss = loss_func(output, b_y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step % 50 == 0:
                test_output = rnn(test_x)
                pred_y = torch.max(test_output, 1)[1].data.squeeze()
                accuracy = sum(pred_y == test_y) / test_y.size(0)
                print('Epoch:', epoch, '| train loss: %.4f' % loss.item(),
                      '|test accuracy: ', accuracy)

    test_output = rnn(test_x[:10])
    pred_y = torch.max(test_output, 1)[1].data.numpy().squeeze()
    print(pred_y, 'prediction number')
    print(test_y[:10].numpy())