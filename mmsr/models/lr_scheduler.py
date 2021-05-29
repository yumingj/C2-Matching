import math
from collections import Counter

from torch.optim.lr_scheduler import _LRScheduler


class MultiStepRestartLR(_LRScheduler):
    """ MultiStep with restarts learning rate scheme.

    Args:
        optimizer (torch.nn.optimizer): Torch optimizer.
        milestones (list): Iterations that will decrease learning rate.
        gamma (float): Decrease ratio. Default: 0.1.
        restarts (list): Restart iterations. Default: [0].
        restart_weights (list): Restart weights at each restart iteration.
            Default: [1].
        last_epoch (int): Used in _LRScheduler. Default: -1.
    """

    def __init__(self,
                 optimizer,
                 milestones,
                 gamma=0.1,
                 restarts=[0],
                 restart_weights=[1],
                 last_epoch=-1):
        self.milestones = Counter(milestones)
        self.gamma = gamma
        self.restarts = restarts
        self.restart_weights = restart_weights
        assert len(self.restarts) == len(
            self.restart_weights), 'restarts and their weights do not match.'
        super(MultiStepRestartLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch in self.restarts:
            weight = self.restart_weights[self.restarts.index(self.last_epoch)]
            return [
                group['initial_lr'] * weight
                for group in self.optimizer.param_groups
            ]
        if self.last_epoch not in self.milestones:
            return [group['lr'] for group in self.optimizer.param_groups]
        return [
            group['lr'] * self.gamma**self.milestones[self.last_epoch]
            for group in self.optimizer.param_groups
        ]


class CosineAnnealingRestartLR(_LRScheduler):
    """ Cosine annealing with restarts learning rate scheme.

    An example of config:
    period = [10, 10, 10, 10]
    restarts = [10, 20, 30]
    restart_weights = [1, 0.5, 0.5]
    eta_min=1e-7

    It has four cycles, each has 10 iterations. At 10th, 20th, 30th, the
    scheduler will restart with the weights in restart_weights.

    Args:
        optimizer (torch.nn.optimizer): Torch optimizer.
        period (list): Period for each cosine anneling cycle.
        restarts (list): Restart iterations. Default: [0].
        restart_weights (list): Restart weights at each restart iteration.
            Default: [1].
        eta_min (float): The mimimum lr. Default: 0.
        last_epoch (int): Used in _LRScheduler. Default: -1.
    """

    def __init__(self,
                 optimizer,
                 period,
                 restarts=[0],
                 restart_weights=[1],
                 eta_min=0,
                 last_epoch=-1):
        self.period = period
        self.restarts = restarts
        self.restart_weights = restart_weights
        self.eta_min = eta_min

        self.current_weight = 1
        self.nearest_restart = 0
        self.current_period = self.period[0]

        assert len(self.restarts) == len(
            self.restart_weights), 'restarts and their weights do not match.'
        assert (len(self.period) - len(self.restarts)
                ) == 1, 'period should have one more element then restarts.'
        super(CosineAnnealingRestartLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch in self.restarts:
            idx = self.restarts.index(self.last_epoch)
            self.current_weight = self.restart_weights[idx]
            self.current_period = self.period[idx + 1]
            self.nearest_restart = self.last_epoch

        return [
            self.eta_min + self.current_weight * 0.5 *
            (base_lr - self.eta_min) *
            (1 + math.cos(math.pi * ((self.last_epoch - self.nearest_restart) /
                                     self.current_period)))
            for base_lr in self.base_lrs
        ]
