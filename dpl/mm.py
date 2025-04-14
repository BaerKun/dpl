import torch
import time

try_cuda = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class ModelManager:
    model: torch.nn.Module

    def __init__(self, model: torch.nn.Module, weights_path: str = None, seq2seq=False, output_state=False):
        self.model = model
        if weights_path is not None:
            self.model.load_state_dict(torch.load(weights_path, weights_only=True))
        self.__seq2seq = seq2seq
        self.__output_state = output_state

    def train(self, loader: torch.utils.data.DataLoader, loss_f, epochs: int, lr: float = 0.001,
              warmup_steps=0, device: torch.device = try_cuda):
        self.model.to(device)
        self.model.train()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        num_iter_per_epoch = str(len(loader))
        print_cuda_memory = device.type == "cuda"
        print("start training.")
        print(f"device: {device.type}.")

        for epoch in range(1, epochs + 1):
            total_loss = 0.
            curr_iter = 0
            if not print_cuda_memory:
                print(f"———————— epoch {epoch}/{epochs} ————————")

            time_start = time.time()
            time_last = time_start
            for x, y in loader:
                x = x.to(device)
                y = y.to(device)

                y_hat = self.model(x)
                if self.__seq2seq:
                    if self.__output_state:
                        y_hat = y_hat[0]
                    if warmup_steps != 0:
                        y_hat = y_hat[:, warmup_steps:]
                        y = y[:, warmup_steps:]
                    y_hat = y_hat.flatten(0, 1)
                    y = y.flatten(0, 1)

                loss = loss_f(y_hat, y)
                total_loss += loss.item()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                curr_iter += 1

                if print_cuda_memory:
                    print(f"CUDA memory usage: {torch.cuda.memory_reserved() // 1024 // 1024} MB.")
                    print(f"———————— epoch 1/{epochs} ————————")
                    print_cuda_memory = False

                if curr_iter % 5 == 0:
                    time_now = time.time()
                    print(f"\rbatch: {curr_iter}/{num_iter_per_epoch}    "
                          f"loss: {total_loss:.4f}    "
                          f"speed: {(time_now - time_last) * 200:.2f}ms/batch", end="")
                    time_last = time_now

            print(f"\r{num_iter_per_epoch}/{num_iter_per_epoch}.")
            print(f"sum of loss: {total_loss:.4f}.")
            print(f"time taken: {time.time() - time_start:.2f}s.")

        print("———————— train over ————————")

    def test(self, loader: torch.utils.data.DataLoader, score_f, device: torch.device = try_cuda):
        self.model.to(device)
        self.model.eval()

        num_iter_per_epoch = str(len(loader))
        curr_iter = 0
        score = 0.
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)

            with torch.no_grad():
                y_hat = self.model(x)
                if self.__output_state:
                    y_hat = y_hat[0]
                score += score_f(y_hat, y)

            curr_iter += 1
            if curr_iter % 5 == 0:
                print(f"\r{curr_iter}/{num_iter_per_epoch}.", end="")
        print(f"\r{num_iter_per_epoch}/{num_iter_per_epoch}.")
        print(f"mean score: {score / len(loader)}")

    def predict(self, x: torch.Tensor, device: torch.device = try_cuda):
        self.model.to(device)
        self.model.eval()
        with torch.no_grad():
            x = x.to(device)
            y = self.model(x)
        return y

    def save(self, path: str):
        torch.save(self.model.state_dict(), path)

    # return_shape = (batch_size, predict_steps, ...)
    def predict_seq(self, x: torch.Tensor, predict_steps, init_state: torch.Tensor = None,
                    device: torch.device = try_cuda) -> (torch.Tensor, torch.Tensor):
        self.model.to(device)
        self.model.eval()

        x = x.to(device)
        state = init_state
        y_pred = []
        for i in range(predict_steps - 1):
            y_hat, state = self.model(x, state)
            y_pred.append(y_hat)
            x = y_hat

        return torch.cat(y_pred, dim=1), state


def score_acc(y_hat: torch.Tensor, y: torch.Tensor):
    y_hat = y_hat.argmax(dim=1)
    return (y_hat == y).float().mean().item()
