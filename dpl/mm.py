import torch
import time

try_cuda = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def _dict_add(_inout: dict, _in: dict):
    if not _inout:
        for key, value in _in.items():
            _inout[key] = value
        return
    for key in _in.keys():
        _inout[key] += _in[key]


class ModelManager:
    model: torch.nn.Module

    def __init__(self, model: torch.nn.Module, weights_path: str = None):
        self.model = model
        if weights_path is not None:
            self.model.load_state_dict(torch.load(weights_path, weights_only=True))

    def save(self, path: str):
        torch.save(self.model.state_dict(), path)

    # score_f: (y_hat, y) -> dict, 输出必须包括 'loss'
    def train(self, loader, score_f, epochs: int, lr: float = 0.001, device: torch.device = try_cuda):
        self.model.to(device)
        self.model.train()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        print_cuda_memory = device.type == "cuda"

        print("———————— train start ————————\n"
              "start training.\n"
              f"device: {device.type}.")

        for epoch in range(1, epochs + 1):
            _iter = loader.__iter__()  # SeqDateLoader会在调用迭代器时会刷新dataset，len(loader)可能会变化
            num_iter_per_epoch = str(len(loader))
            curr_iter = 0
            total_score = {}

            if not print_cuda_memory:
                print(f"———————— epoch {epoch}/{epochs} ————————")

            time_start = time.time()
            time_last = time_start
            for x, y in _iter:
                y_hat = self.model(x.to(device))
                score = score_f(y_hat, y.to(device))
                _dict_add(total_score, score)

                optimizer.zero_grad()
                score['loss'].backward()
                optimizer.step()

                curr_iter += 1
                if curr_iter % 5 == 0:
                    if print_cuda_memory:
                        print(f"cuda memory usage: {torch.cuda.memory_reserved() // 1024 // 1024} MB.\n"
                              f"———————— epoch 1/{epochs} ————————")
                        print_cuda_memory = False

                    time_now = time.time()
                    score_str = [f"| {key}: {value:.2f}" for key, value in score.items()]
                    print(f"\rbatch: {num_iter_per_epoch}/{num_iter_per_epoch} | "
                          f"speed: {(time_now - time_last) * 200:.2f}ms/batch",
                          *score_str, end="")
                    time_last = time_now

            print(f"\rbatch: {curr_iter}/{num_iter_per_epoch}.")
            for key, value in total_score.items():
                print(f"{key}: {value / curr_iter:.4f}")
            print(f"time taken: {time.time() - time_start:.2f}s.")

        print("———————— train over ————————")

    def test(self, loader, score_f, device: torch.device = try_cuda):
        self.model.to(device)
        self.model.eval()
        print_cuda_memory = device.type == "cuda"

        print("———————— test start ————————\n"
              "test training.\n"
              f"device: {device.type}.")

        _iter = loader.__iter__()
        num_iter_per_epoch = str(len(loader))
        curr_iter = 0
        total_score = {}

        time_start = time.time()
        time_last = time_start
        with torch.no_grad():
            for x, y in _iter:
                y_hat = self.model(x.to(device))
                score = score_f(y_hat, y.to(device))
                _dict_add(total_score, score)

                curr_iter += 1
                if curr_iter % 5 == 0:
                    if print_cuda_memory:
                        print(f"cuda memory usage: {torch.cuda.memory_reserved() // 1024 // 1024} MB.")
                        print_cuda_memory = False

                    time_now = time.time()
                    score_str = [f"| {key}: {value:.2f}" for key, value in score.items()]
                    print(f"\rbatch: {num_iter_per_epoch}/{num_iter_per_epoch} | "
                          f"speed: {(time_now - time_last) * 200:.2f}ms/batch",
                          *score_str, end="")
                    time_last = time_now

        print(f"\rbatch: {curr_iter}/{num_iter_per_epoch}.")
        for key, value in total_score.items():
            print(f"{key}: {value / curr_iter:.4f}")
        print(f"time taken: {time.time() - time_start:.2f}s.\n"
              "———————— test over ————————")

    def predict(self, x: torch.Tensor, device: torch.device = try_cuda):
        self.model.to(device)
        self.model.eval()
        with torch.no_grad():
            x = x.to(device)
            y = self.model(x)
        return y

    # return_shape = (batch_size, predict_steps, ...)
    def predict_seq(self, x: torch.Tensor, predict_steps, init_state: torch.Tensor = None,
                    device: torch.device = try_cuda):
        self.model.to(device)
        self.model.eval()

        x = x.to(device)
        state = init_state
        y_pred = []

        with torch.no_grad():
            for i in range(predict_steps):
                y_hat, state = self.model(x, state)
                y_pred.append(y_hat)
                x = y_hat[:, -1:]

        return torch.cat(y_pred, dim=1), state
