import random

import torch
from torch import nn
import utils
import os


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, batch_first=False):
        super(RNN, self).__init__()
        self.batch_first = batch_first
        self.w = nn.ModuleList(
            (nn.ModuleList((nn.Linear(input_size, hidden_size, bias=False), nn.Linear(hidden_size, hidden_size))),))

        for _ in range(num_layers - 1):
            self.w.append(
                nn.ModuleList((nn.Linear(hidden_size, hidden_size, bias=False), nn.Linear(hidden_size, hidden_size))))

    def forward(self, x: torch.Tensor, state):
        hidden_layers = x.permute(1, 0, 2) if self.batch_first else x
        hidden_steps = []
        out_state = []

        for h_s, (w_l, w_s) in zip(state, self.w):
            for h_l in hidden_layers:
                h_s = torch.tanh(w_l(h_l) + w_s(h_s))
                hidden_steps.append(h_s)
            hidden_layers = hidden_steps
            hidden_steps = []
            out_state.append(h_s)

        hidden_steps_tensor = torch.stack(hidden_layers)    # (seq_len, batch_size, hidden_size)
        return (hidden_steps_tensor.permute(1, 0, 2)        # -> (batch_size, seq_len, hidden_size)
                if self.batch_first
                else hidden_steps_tensor,
                torch.stack(out_state))                     # (num_layers, batch_size, hidden_size)


class RNNModel(nn.Module):
    __init_state: torch.Tensor
    def __init__(self, vocab_size, embed_size, num_layers, hidden_size):
        super(RNNModel, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.rnn = RNN(embed_size, hidden_size, num_layers, batch_first=True)
        # self.rnn = nn.RNN(embed_size, hidden_size, num_layers, batch_first=True)
        self.decoder = nn.Linear(hidden_size, vocab_size)

    def set_init_state(self, state):
        self.__init_state = state

    def forward(self, x, state=None):
        x = self.embed(x)
        y, state = self.rnn(x, self.__init_state if state is None else state)
        y = self.decoder(y)
        return y, state

def loss_f(warm_up=0):
    cross_entropy = nn.CrossEntropyLoss()
    if warm_up == 0:
        def __loss_f(y_hat:torch.Tensor, y:torch.Tensor):
            return cross_entropy(y_hat[0].reshape(-1, len(corpus.vocab)), y.reshape(-1))
    else:
        def __loss_f(y_hat:torch.Tensor, y:torch.Tensor):
            y_hat = y_hat[:, warm_up:]
            y = y[:, warm_up:]
            return cross_entropy(y_hat[0].reshape(-1, len(corpus.vocab)), y.reshape(-1))
    return __loss_f

corpus = utils.load_wiki_text()
corpus.build_vocab(24)
loader = corpus.get_loader(256, 8, random_sample=True)
model = RNNModel(len(corpus.vocab), 196, 2, 256)
mm = utils.ModelManager(os.path.join(utils.project_root, "model", "rnn.pt"))

# init_state = torch.zeros(2, 256, 256, dtype=torch.float32, device='cuda', requires_grad=False)
# mm.model.set_init_state(init_state)
# mm.train(loader, loss_f(2), 20)
# mm.save(os.path.join(utils.project_root, "model", "rnn.pt"))

pred_init_state = torch.zeros(2, 1, 256, dtype=torch.float32, requires_grad=False)
mm.model.set_init_state(pred_init_state)
start = random.randint(0, len(corpus) - 8)
t = corpus.corpus_idx[start:start + 8].reshape(1, 8)
print(corpus.vocab.decode(t[0]))
t_hat, state = mm.predict(t, device=torch.device('cpu'))
t_prev = t_hat.argmax(dim=2)[:, 7].reshape(1, 1)
t_pred = t_prev

for i in range(8):
    t_hat, state = mm.model(t_prev, state)
    t_hat = torch.argmax(t_hat, dim=2)
    t_pred = torch.cat((t_pred, t_hat), dim=1)
    t_prev = t_hat

print(corpus.vocab.decode(t_pred[0]))
