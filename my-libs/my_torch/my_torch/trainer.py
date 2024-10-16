class Trainer:
    epoch: int
    current: int
    each_bitch: bool
    total_loss: float

    def __init__(self, dataset, dataloader, model, loss_function, optimizer):
        self.dataset = dataset
        self.dataloader = dataloader
        self.model = model
        self.loss_function = loss_function
        self.optimizer = optimizer

        optimizer.push_params(model.get_params())

    def train(self, epoch, each_bitch=False):
        self.epoch = epoch
        self.each_bitch = each_bitch
        self.dataloader.to_train()

        return self

    def __iter__(self):
        self.current = 0
        self.total_loss = 0.
        self.dataloader.__iter__()

        return self

    def __next__(self):
        if self.current >= self.epoch:
            raise StopIteration

        try:
            while True:
                indata, target = self.dataloader.__next__()
                output = self.model.forward(indata)
                loss = self.loss_function(output, target)
                self.model.backward(loss)
                self.optimizer.step()

                if self.each_bitch:
                    self.total_loss += loss.item()
                    return loss.item()

        except StopIteration:
            self.current += 1
            self.dataloader.__iter__()

            total_loss = self.total_loss
            self.total_loss = 0.

            return total_loss
