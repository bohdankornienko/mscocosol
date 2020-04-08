class Unet:
    def __init__(self):
        pass

    def train_on_batch(self, x, y):
        pass

    def eval_on_batch(self, x, y):
        y_pred = self.predict(x)

    def predict(self, x):
        pass


def make_unet(**kwargs):
    return Unet(**kwargs)

