import torch
import torch.nn as nn
import numpy as np

MEMORY_CAPACITY = 2000
EPISILO = 0.9
Q_NETWORK_ITERATION = 100
GAMMA = 0.9
class LinearNet(nn.Module):
    def __init__(self, size, hidden=4096):
        super(LinearNet, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(size, hidden),
            nn.LeakyReLU(),
            nn.Linear(hidden, 200),
            nn.Dropout(0.5),
        )

    def forward(self, x):
        return self.fc(x)



class CnnNet(nn.Module):
    def __init__(self):
        super(CnnNet, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 16, 5, 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, 5, 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.fc = nn.Sequential(
            nn.Linear(32 * 12 * 4, 4096),
            nn.LeakyReLU(),
            nn.Linear(4096, 200)
        )


    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)  # (batch, 32, 7, 7)
        x = x.view(x.size(0), -1)  # (batch, 32 * 7 * 7)
        output = self.fc(x)
        return output


class LSTMNet(nn.Module):
    # lstm 模型参数的意义，sequence还有layer number
    def __init__(self, input_dim, hidden_dim, n_layers, output_size):
        super(LSTMNet, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, n_layers, batch_first=True)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(hidden_dim, output_size)
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim

    def forward(self, x, hidden):
        x, hidden = self.lstm(x, hidden)
        # lstm_out = lstm_out.contiguous().view(-1, self.hidden_dim)
        x = self.dropout(x)
        x = x[:, -1, :]
        x = self.fc(x)
        return x, hidden

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_(),
                  weight.new(self.n_layers, batch_size, self.hidden_dim).zero_())
        return hidden



# DQNnet, base on CNN
# class DQNNet(nn.Module):
#     def __init__(self, size, action_size, lr = 0.0001):
#         super(DQNNet, self).__init__()
#         self.action_size = action_size
#         self.size = size
#         self.eval_net, self.target_net = LinearNet(), LinearNet()
#         self.learn_setp_counter = 0
#         self.memory_counter = 0
#         self.memory = np.zeros(MEMORY_CAPACITY, size * 2 + 2)
#         self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr = lr)
#         self.loss_func = nn.MSELoss()
#
#     def choose_action(self, state):
#         state = torch.unsqueeze(torch.FloatTensor(state), 0)
#         if np.random.randn() < EPISILO:
#             action_value = self.eval_net.forward(state)
#             action = torch.max(action_value, 1)[1].data.numpy()
#             action = action[0]
#         else:
#             action = np.random.randint(0, self.action_size)
#         return action
#
#     def store_transition(self, state, action, reward, next_state):
#         transition = np.stack(state, [action, reward], next_state)
#         index = self.memory_counter % MEMORY_CAPACITY
#         self.memory[index, :] = transition
#         self.memory_counter += 1
#
#     def learn(self):
#         if self.learn_setp_counter % Q_NETWORK_ITERATION == 0:
#             self.target_net.load_state_dict(self.eval_net.state_dict())
#         self.learn_setp_counter += 1
#         sample_index = np.random.choice(MEMORY_CAPACITY, 1)
#         batch_memory = self.memory[sample_index, :]
#         batch_state = torch.FloatTensor(batch_memory[:, :self.size])
#         batch_action = torch.FloatTensor(batch_memory[:, self.size :self.size+1].astype(int))
#         batch_reward = torch.FloatTensor(batch_memory[: self.size + 1: self.size + 2])
#         batch_next_state = torch.FloatTensor(batch_memory[:, -self.size:])
#         q_eval = self.eval_net(batch_state).gather(1, batch_action)
#         q_next = self.target_net(batch_next_state).detach()
#         q_target = batch_reward + GAMMA * q_next.max(1)[0].view(1, 1)
#         loss = self.loss_func(q_eval, q_target)
#
#         self.optimizer.zero_grad()
#         loss.backward()
#         self.optimizer.step()
#
#     def reward_func(self, ):



