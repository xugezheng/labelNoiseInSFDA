import torch
import torch.nn as nn

def cross_entropy(targets, output, args, reduction='mean', smooth=0.1):
    if targets.size() != output.size():
        ones = torch.eye(args.class_num, device=output.device)
        targets_2d = torch.index_select(ones, dim=0, index=targets)
    else:
        targets_2d = targets
    
    pred = nn.Softmax(dim=1)(output)
    
    if smooth > 0:
        targets_2d = (1 - smooth) * targets_2d + smooth / args.class_num
    
    if reduction == 'none':
        return torch.sum(- targets_2d * torch.log(pred), 1)
    elif reduction == 'mean':
        return torch.mean(torch.sum(- targets_2d * torch.log(pred), 1))
    
    
def Entropy(input_):
    bs = input_.size(0)
    epsilon = 1e-5
    entropy = -input_ * torch.log(input_ + epsilon)
    entropy = torch.sum(entropy, dim=1)
    return entropy