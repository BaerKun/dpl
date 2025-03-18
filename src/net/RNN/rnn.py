import torch
from torch import nn


class RNNLayer(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(RNNLayer, self).__init__()
        self.w_x = nn.Linear(input_size, hidden_size)
        self.w_h = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, x, state):
        y = []
        for _x in x:
            if state is None:
                state = torch.tanh(self.w_x(x))

            else:
                state = torch.tanh(self.w_x(x) + self.w_h(state))

            y.append(state)
        return y, state


class GRULayer(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(GRULayer, self).__init__()
        self.w_xr = nn.Linear(input_size, hidden_size)
        self.w_hr = nn.Linear(hidden_size, hidden_size, bias=False)
        self.w_xz = nn.Linear(input_size, hidden_size)
        self.w_hz = nn.Linear(hidden_size, hidden_size, bias=False)
        self.w_xh = nn.Linear(input_size, hidden_size)
        self.w_hh = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, x, state):
        y = []
        for _x in x:
            if state is None:
                update = torch.sigmoid(self.w_xz(_x))
                h_cand = torch.tanh(self.w_xh(_x))
                state = (1 - update) * h_cand

            else:
                reset = torch.sigmoid(self.w_xr(_x) + self.w_hr(state))
                update = torch.sigmoid(self.w_xz(_x) + self.w_hz(state))
                h_cand = torch.tanh(self.w_xh(_x) + self.w_hh(state * reset))
                state = update * state + (1 - update) * h_cand

            y.append(state)
        return y, state


class LSTMLayer(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(LSTMLayer, self).__init__()
        self.w_xi = nn.Linear(input_size, hidden_size)
        self.w_hi = nn.Linear(hidden_size, hidden_size, bias=False)
        self.w_xf = nn.Linear(input_size, hidden_size)
        self.w_hf = nn.Linear(hidden_size, hidden_size, bias=False)
        self.w_xo = nn.Linear(input_size, hidden_size)
        self.w_ho = nn.Linear(hidden_size, hidden_size, bias=False)
        self.w_xc = nn.Linear(input_size, hidden_size)
        self.w_hc = nn.Linear(hidden_size, hidden_size, bias=False)

    # state = (H, C)
    def forward(self, x, state):
        if state is None:
            h = c = None
        else:
            h, c = state

        y = []
        for _x in x:
            if h is None:
                _input = torch.sigmoid(self.w_xi(_x))
                _output = torch.sigmoid(self.w_xo(_x))
                cell_cand = torch.tanh(self.w_xc(_x))
                c = _input * cell_cand
                h = _output * torch.tanh(c)

            else:
                _input = torch.sigmoid(self.w_xi(_x) + self.w_hi(h))
                _forget = torch.sigmoid(self.w_xf(_x) + self.w_hf(h))
                _output = torch.sigmoid(self.w_xo(_x) + self.w_ho(h))

                cell_cand = torch.tanh(self.w_xc(_x) + self.w_hc(h))
                c = _forget * c + _input * cell_cand
                h = _output * torch.tanh(c)

            y.append(h)
        return y, (h, c)



class RNNBase(nn.Module):
    def forward(self, x: torch.Tensor, state=None):
        hidden = x.permute(1, 0, 2) if self.batch_first else x
        out_state = []

        for i, layer in enumerate(self.rnn_layers):
            state_l = None if state is None else state[i]
            hidden, state_l = layer(hidden, state_l)
            out_state.append(state_l)

        hidden_tensor = torch.stack(hidden)  # (seq_len, batch_size, hidden_size)
        return (hidden_tensor.permute(1, 0, 2)  # -> (batch_size, seq_len, hidden_size)
                if self.batch_first
                else hidden_tensor,
                out_state)  # (num_layers, batch_size, <2 if LSTM,> hidden_size)

    def _init(self, input_size, hidden_size, num_layers, batch_first, rnn_layer):
        super(RNNBase, self).__init__()
        self.batch_first = batch_first
        layers = []

        for _ in range(num_layers):
            layers.append(rnn_layer(input_size, hidden_size))
            input_size = hidden_size

        self.rnn_layers = nn.Sequential(*layers)
        self.__init_state = None
        self.__hidden_size = hidden_size


class RNN(RNNBase):
    def __init__(self, input_size, hidden_size, num_layers=1, batch_first=False):
        super(RNN, self).__init__()
        self._init(input_size, hidden_size, num_layers, batch_first, RNNLayer)


class GRU(RNNBase):
    def __init__(self, input_size, hidden_size, num_layers=1, batch_first=False):
        super(GRU, self).__init__()
        self._init(input_size, hidden_size, num_layers, batch_first, GRULayer)


class LSTM(RNNBase):
    def __init__(self, input_size, hidden_size, num_layers=1, batch_first=False):
        super(LSTM, self).__init__()
        self._init(input_size, hidden_size, num_layers, batch_first, LSTMLayer)


class SeqRNN(nn.Module):
    def __init__(self, vocab_size, embed_size, num_layers, hidden_size, rnn_mode:str = 'rnn'):
        super(SeqRNN, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.decoder = nn.Linear(hidden_size, vocab_size)

        rnn_mode = rnn_mode.lower()
        if rnn_mode == 'rnn':
            self.rnn = RNN(embed_size, hidden_size, num_layers, batch_first=True)
        elif rnn_mode == 'gru':
            self.rnn = GRU(embed_size, hidden_size, num_layers, batch_first=True)
        elif rnn_mode == 'lstm':
            self.rnn = LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        else:
            raise ValueError(f'Invalid rnn mode: {rnn_mode}')

    def forward(self, x, state=None):
        x = self.embed(x)
        y, state = self.rnn(x, state)
        y = self.decoder(y)
        return y, state