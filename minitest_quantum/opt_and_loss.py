import torch 
import torch.nn as nn
import math

class ScheduledOptim():
    '''A simple wrapper class for learning rate scheduling'''

    def __init__(self, optimizer,initial_lr, n_warmup_steps,end_lr):
        self._optimizer = optimizer
        self.initial_lr = initial_lr
        self.n_warmup_steps = n_warmup_steps
        self.n_steps = 0
        self.end_lr=end_lr


    def step_and_update_lr(self):
        "Step with the inner optimizer"
        self._update_learning_rate()
        self._optimizer.step()


    def zero_grad(self):
        "Zero out the gradients with the inner optimizer"
        self._optimizer.zero_grad()


    def _get_lr_scale(self):
        step = min(self.n_steps, self.n_warmup_steps)
        return ((self.initial_lr - self.end_lr) *
                    (1 - step / self.n_warmup_steps)
                    ) + self.end_lr


    def _update_learning_rate(self):
        ''' Learning rate scheduling per step '''

        self.n_steps += 1
        lr = self._get_lr_scale()

        for param_group in self._optimizer.param_groups:
            param_group['lr'] = lr


def loss_fn(pred,labels):
    #loss=nn.MSELoss()
    #return torch.sqrt(loss(pred,labels))
    loss=log_cosh_loss(pred, labels)
    return loss

def log_cosh_loss(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    def _log_cosh(x: torch.Tensor) -> torch.Tensor:
      return x + torch.nn.functional.softplus(-2. * x) - math.log(2.0)
    return torch.mean(_log_cosh(y_pred - y_true))