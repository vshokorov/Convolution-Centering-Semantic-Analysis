import logging
import os
import sys


class AverageMeter(object):
    """Computes and stores the average and current value
    """

    def __init__(self, memory = 10):
        self.alpha = 1 / memory
        self.reset()

    def reset(self):
        self.val = None
        self.avg = None

    def update(self, val, n=1):
        self.val = val
        if self.avg is None:
            self.avg = val
        else:
            for _ in range(n):
                self.avg = self.alpha * val + (1 - self.alpha) * self.avg


def init_logging(rank, models_root):
    if rank == 0:
        log_root = logging.getLogger()
        log_root.setLevel(logging.INFO)
        formatter = logging.Formatter("Training: %(asctime)s-%(message)s")
        handler_file = logging.FileHandler(os.path.join(models_root, "training.log"))
        handler_stream = logging.StreamHandler(sys.stdout)
        handler_file.setFormatter(formatter)
        handler_stream.setFormatter(formatter)
        log_root.addHandler(handler_file)
        log_root.addHandler(handler_stream)
        log_root.info('rank_id: %d' % rank)
