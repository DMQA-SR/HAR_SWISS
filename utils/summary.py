import os

from torch.utils.tensorboard import SummaryWriter


class TensorboardSummary(object):
    def __init__(self, directory):
        self.directory = directory
        if not os.path.exists(self.directory):
            os.makedirs(self.directory)

    def create_summary(self):
        writer = SummaryWriter(log_dir=os.path.join(self.directory))

        return writer
