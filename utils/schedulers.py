import torch

class WarmupExponetialSchedule(torch.optim.lr_scheduler.LambdaLR):
    
    def __init__(self, optimizer, warmup_steps, gamma=0.95, last_epoch=-1):

        def lr_lambda(step):
            if step < warmup_steps:
                return float(step) / float(max(1.0, warmup_steps))
            else:
                return gamma ** (step-warmup_steps)
        super(WarmupExponetialSchedule, self).__init__(optimizer, lr_lambda, last_epoch=last_epoch)
        

class WarmupConstantSchedule(torch.optim.lr_scheduler.LambdaLR):

    def __init__(self, optimizer, warmup_steps, last_epoch=-1):

        def lr_lambda(step):
            if step < warmup_steps:
                return float(step) / float(max(1.0, warmup_steps))
            return 1.

        super(WarmupConstantSchedule, self).__init__(optimizer, lr_lambda, last_epoch=last_epoch)
        

class WarmupStepLRSchedule(torch.optim.lr_scheduler.LambdaLR):
    
    def __init__(self, optimizer, warmup_steps, gamma=0.8, period=25 , last_epoch=-1):

        def lr_lambda(step):
            if step < warmup_steps:
                return float(step) / float(max(1.0, warmup_steps))
            
            else:
                quotient = (step-warmup_steps)//period
                return gamma ** quotient

        super(WarmupStepLRSchedule, self).__init__(optimizer, lr_lambda, last_epoch=last_epoch)