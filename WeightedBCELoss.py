import torch

class WeightedBCELoss():
    def __init__(self, zero_weight, one_weight):
        self.zero_weight = zero_weight
        self.one_weight = one_weight
    
    def loss(self, output, target):
        loss = self.one_weight * (target * torch.log(output)) + \
               self.zero_weight * ((1 - target) * torch.log(1 - output))
        
        # return torch.neg(torch.mean(loss))
        return torch.neg(torch.sum(loss))