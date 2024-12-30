import torch
from torch import nn
import utils
import os


class RNNLayer(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(RNNLayer, self).__init__()
        self.w_l = nn.Linear(input_size, hidden_size, bias=False)
        self.w_t = nn.Linear(hidden_size, hidden_size)

    def forward(self, x, state):
        y = []
        for h_l in x:
            state = torch.tanh(self.w_l(h_l) + self.w_t(state))
            y.append(state)
        return y, state


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, batch_first=False):
        super(RNN, self).__init__()
        self.batch_first = batch_first
        layers = []
        for _ in range(num_layers):
            layers.append(RNNLayer(input_size, hidden_size))
            input_size = hidden_size

        self.rnn_layers = nn.Sequential(*layers)
        self.__init_state = None
        self.__hidden_size = hidden_size

    def forward(self, x: torch.Tensor, state=None):
        if state is None:
            if self.__init_state is None or x.shape[1] != self.__init_state.shape[0]:
                self.__init_state = torch.zeros(x.shape[1], self.__hidden_size, dtype=x.dtype,
                                                device=x.device, requires_grad=False)
            elif x.device != self.__init_state.device:
                self.__init_state.to(x.device)
            state = self.__init_state

        hidden_layers = x.permute(1, 0, 2) if self.batch_first else x
        out_state = []

        for layer, state_l in zip(self.rnn_layers, state):
            hidden_layers, state_l = layer(hidden_layers, state_l)
            out_state.append(state_l)

        hidden_tensor = torch.stack(hidden_layers)  # (seq_len, batch_size, hidden_size)
        return (hidden_tensor.permute(1, 0, 2)  # -> (batch_size, seq_len, hidden_size)
                if self.batch_first
                else hidden_tensor,
                torch.stack(out_state))  # (num_layers, batch_size, hidden_size)


class RNNModel(nn.Module):
    def __init__(self, vocab_size, embed_size, num_layers, hidden_size):
        super(RNNModel, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.rnn = RNN(embed_size, hidden_size, num_layers, batch_first=True)
        # self.rnn = nn.RNN(embed_size, hidden_size, num_layers, batch_first=True)
        self.decoder = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, state=None):
        x = self.embed(x)
        y, state = self.rnn(x, state)
        y = self.decoder(y)
        return y, state


corpus = utils.load_wiki_text()
vocab = corpus.build_vocab(8)
loader = corpus.get_loader(256, 16, random_sample=True)
model = RNNModel(len(vocab), 256, 4, 512)

# mm = utils.ModelManager(os.path.join(utils.project_root, "model", "rnn.pt"), seq2seq=True, output_state=True)
mm = utils.ModelManager(model, seq2seq=True, output_state=True)
mm.train(loader, nn.CrossEntropyLoss(), 10, 0.001, warmup_steps=4)
mm.save(os.path.join(utils.project_root, "model", "rnn.pt"))

import random

for _ in range(10):
    start = random.randint(0, len(corpus.tensor_dataset) - 8)
    t = corpus.tensor_dataset[start: start + 16]
    pred_t = mm.predict_seq(t.reshape(1, -1), 16)
    pred_t = vocab.decode(pred_t[0])
    print(" ".join(vocab.decode(t)))
    print(" ".join(pred_t))
    print()
