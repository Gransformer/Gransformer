'''A wrapper class for scheduled optimizer '''
import numpy as np

class ScheduledOptim():
    '''A simple wrapper class for learning rate scheduling'''

    def __init__(self, optimizer, lr_mul, d_model, n_warmup_steps):
        self._optimizer = optimizer
        self.lr_mul = lr_mul
        self.d_model = d_model
        self.n_warmup_steps = n_warmup_steps
        self.n_steps = 0


    def step_and_update_lr(self):
        "Step with the inner optimizer"
        self._update_learning_rate()
        self._optimizer.step()


    def zero_grad(self):
        "Zero out the gradients with the inner optimizer"
        self._optimizer.zero_grad()


    def _get_lr_scale(self):
        d_model = self.d_model
        n_steps, n_warmup_steps = self.n_steps, self.n_warmup_steps
        return (d_model ** -0.5) * min(n_steps ** (-0.5), n_steps * n_warmup_steps ** (-1.5))


    def _update_learning_rate(self):
        ''' Learning rate scheduling per step '''

        self.n_steps += 1
        # lr = self.lr_mul * self._get_lr_scale()
 
        lr = 0.0002
        
        '''
        if self.n_steps < 25 * 25:
            lr = 0.0005
        elif self.n_steps < 25 * 50:
            lr = 0.00005
        elif self.n_steps < 25 * 75:
            lr = 0.000005
        else:
            lr = 0.0000005
        '''
        

        for param_group in self._optimizer.param_groups:
            param_group['lr'] = lr


class MyScheduledOptim():
    '''A simple wrapper class for learning rate scheduling'''

    def __init__(self, optimizer, sep_optimizer, milestones, lr_list, sep_optimizer_start_step):
        self._optimizer = optimizer
        self._sep_optimizer = sep_optimizer
        self.milestones = milestones
        self.lr_list = lr_list
        self.n_steps = 0
        self._sep_optimizer_start_step = sep_optimizer_start_step

    def set_n_steps(self, n_steps):
        self.n_steps = n_steps

    def _get_active_optimizer(self):
        '''
        if np.mod(self.n_steps, 5) != 0:
            return self._sep_optimizer
        else:
            return self._optimizer
        '''
        if self.n_steps >= self._sep_optimizer_start_step:
            return self._sep_optimizer
        else:
            return self._optimizer

    def step_and_update_lr(self):
        "Step with the inner optimizer"
        self._update_learning_rate()
        self._get_active_optimizer().step()

    def zero_grad(self):
        "Zero out the gradients with the inner optimizer"
        self._get_active_optimizer().zero_grad()

    def _update_learning_rate(self):
        ''' Learning rate scheduling per step '''

        self.n_steps += 1

        new_lr = self.lr_list[0]
        if len(self.milestones) > 0:
            for ml, lr in zip(self.milestones, self.lr_list[1:]):
                if self.n_steps > ml:
                    new_lr = lr
                else:
                    break

        for param_group in self._get_active_optimizer().param_groups:
            param_group['lr'] = new_lr

