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


def _detach_recursively(view):
    if isinstance(view, torch.Tensor):
        return view.detach_()
    return tuple(_detach_recursively(item) for item in view)


class ModelManager:
    def __init__(self, model: torch.nn.Module, weights_path: str = None):
        self.model = model
        if weights_path is not None:
            self.model.load_state_dict(torch.load(weights_path, weights_only=True))

    def save(self, path: str):
        torch.save(self.model.state_dict(), path)

    def __infer(self, x: torch.Tensor, state, transfer_state):
        if transfer_state:
            result = self.model(x, state)
            state = result[-1]
        else:
            result = self.model(x)
        return result, state

    # score_f: (result, y) -> dict
    # 输出必须包括 'loss': torch.Tensor，其余随意
    # result是模型的直接输出，可能包括state等
    # 推荐使用 with torch.no_grad() 计算非 'loss' 值
    # print输出格式 key: value
    def train(self, loader, score_f, epochs: int, lr: float = 0.001, transfer_state=False,
              device: torch.device = try_cuda):
        self.model.to(device)
        self.model.train()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        print_cuda_memory = device.type == "cuda"

        print("———————— train start ————————\n"
              f"device: {device.type}.")

        for epoch in range(1, epochs + 1):
            if not print_cuda_memory:
                print(f"———————— epoch {epoch}/{epochs} ————————")

            _iter = loader.__iter__()  # SeqDateLoader会在调用迭代器时会刷新dataset，len(loader)可能会变化
            num_iter_per_epoch = str(len(loader))
            curr_iter = 0
            state = None
            total_score = {}

            time_start = time.time()
            time_last = time_start
            for x, y in _iter:
                result, state = self.__infer(x.to(device), state, transfer_state)

                score = score_f(result, y.to(device))
                _dict_add(total_score, score)

                state = _detach_recursively(state)

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

    def test(self, loader, score_f, transfer_state=False, device: torch.device = try_cuda):
        self.model.to(device)
        self.model.eval()
        print_cuda_memory = device.type == "cuda"

        print("———————— test start ————————\n"
              "test training.\n"
              f"device: {device.type}.")

        _iter = loader.__iter__()
        num_iter_per_epoch = str(len(loader))
        curr_iter = 0
        state = None
        total_score = {}

        time_start = time.time()
        time_last = time_start
        with torch.no_grad():
            for x, y in _iter:
                result, state = self.__infer(x.to(device), state, transfer_state)

                score = score_f(result, y.to(device))
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
