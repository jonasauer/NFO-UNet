class EarlyStopping:

    def __init__(self, patience: int):
        self.patience = patience
        self.patience_counter = 0
        self.best_validation_loss = float('inf')
        self.validation_loss = None

    def __call__(self, validation_loss: float):
        self.validation_loss = validation_loss
        if self.validation_loss < self.best_validation_loss:
            self.best_validation_loss = self.validation_loss
            self.patience_counter = 0
            return False

        self.patience_counter += 1
        return self.patience_counter > self.patience

    def save_net(self):
        return self.best_validation_loss == self.validation_loss
