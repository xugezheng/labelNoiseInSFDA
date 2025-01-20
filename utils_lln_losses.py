import torch
import torch.nn.functional as F
import torch.nn as nn


def criterion_load(args):
   
    if args.lln_mask:
        if args.lln_type == 'elr':
            return ELR_loss_masked(args.beta, args.lamb, args.nb_samples, args.nb_classes)
        elif args.lln_type == 'gce':
            return GCE_loss_masked(args.beta, args.lamb, args.nb_samples, args.nb_classes)
        elif args.lln_type == 'sl':
            return SL_loss_masked(args.beta, args.lamb, args.nb_samples, args.nb_classes)
        elif args.lln_type == 'gjs':
            return JensenShannonDivergenceWeightedCustom_masked(args.beta, args.lamb, args.nb_samples, args.nb_classes)
        else:
            print("Unknown lln masked type")
        
        
    else:
        if args.lln_type == 'elr':
            return ELR_loss(args.beta, args.lamb, args.nb_samples, args.nb_classes)
        elif args.lln_type == 'gce':
            return GCE_loss(args.beta, args.lamb, args.nb_samples, args.nb_classes)
        elif args.lln_type == 'sl':
            return SL_loss(args.beta, args.lamb, args.nb_samples, args.nb_classes)
        elif args.lln_type == 'gjs':
            return JensenShannonDivergenceWeightedCustom(args.beta, args.lamb, args.nb_samples, args.nb_classes)
        else:
            print("Unknown lln type")


# ==================================================
# ================= original loss =================
# ==================================================

class ELR_loss(nn.Module):
    def __init__(self, beta, lamb, num, cls):
        super(ELR_loss, self).__init__()
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.ema = torch.zeros(num, cls).to(device)
        self.beta = beta
        self.lamb = lamb
        print('Load ELR loss.')
        self.all_previous_logits = []
        self.all_new_logits = []
        self.all_merged_logits = []

    def forward(self, index, outputs, targets, record_flag=False):
        # record_flag is only check elr effect
        y_pred = torch.nn.functional.softmax(outputs, dim=1)
        y_pred = torch.clamp(y_pred, 1e-4, 1.0 - 1e-4)
        y_pred_ = y_pred.data.detach()
        if record_flag:
            self.all_previous_logits.append(self.ema[index].detach().clone())
            self.all_new_logits.append(y_pred.detach().clone())
        self.ema[index] = self.beta * self.ema[index] + (1 - self.beta) * (
            (y_pred_) / (y_pred_).sum(dim=1, keepdim=True))
        
        elr_reg = ((1 - (self.ema[index] * y_pred).sum(dim=1)).log()).mean()
        final_loss = self.lamb * elr_reg
        
        if record_flag:
            self.all_merged_logits.append(self.ema[index].detach().clone())
            return final_loss, self.all_previous_logits, self.all_new_logits, self.all_merged_logits
        else:
            return final_loss


class GCE_loss(nn.Module):
    def __init__(self, beta=0.2, lamb=0.0, num=100, cls=12):
        super(GCE_loss, self).__init__()
        self.q = beta
        self.sample_num = num
        self.class_num = cls
        print('Load GCE loss.')

    def forward(self, ind, outputs, targets):
        targets = torch.zeros(targets.size(0), self.class_num).cuda().scatter_(
            1, targets.view(-1, 1), 1)
        pred = torch.nn.functional.softmax(outputs, dim=1)
        pred_y = torch.sum(targets * pred, dim=1)
        final_loss = torch.mean((1.0 - pred_y ** self.q) / self.q, dim=0)
        return final_loss


class CE_loss(nn.Module):
    def __init__(self, beta=0.2, lamb=0.0, num=100, cls=12):
        super(CE_loss, self).__init__()

    def forward(self, ind, outputs, targets):
        return torch.nn.functional.cross_entropy(outputs, targets)


class SL_loss(nn.Module):
    def __init__(self, beta=1.0, lamb=1.0, num=100, cls=12):
        super(SL_loss, self).__init__()
        self.q = beta
        self.b = lamb
        self.sample_num = num
        self.class_num = cls
        print('Load SL loss.')

    def forward(self, ind, outputs, targets):
        targets_ = torch.zeros(targets.size(0), self.class_num).cuda().scatter_(
            1, targets.view(-1, 1), 1)
        pred = torch.nn.functional.softmax(outputs, dim=1)
        targets_ = torch.clamp(targets_, 1e-4, 1.0)
        final_loss = - \
            torch.mean(torch.sum(torch.log(targets_)*pred, dim=1))*self.q # + torch.nn.functional.cross_entropy(outputs, targets)
        return final_loss


def get_criterion(num_classes, args):
    alpha = 0.1
    beta = 0.45
    loss_options = {
        'JSWC': JensenShannonDivergenceWeightedCustom(num_classes=num_classes, weights=args.beta),
    }

    criterion = loss_options['JSWC']

    return criterion

# ----
# Based on https://github.com/pytorch/pytorch/blob/0c474d95d9cdd000035dc9e3cd241ba928a08230/aten/src/ATen/native/Loss.cpp#L79


def custom_kl_div(prediction, target):
    output_pos = target * (target.clamp(min=1e-7).log() - prediction)
    zeros = torch.zeros_like(output_pos)
    output = torch.where(target > 0, output_pos, zeros)
    output = torch.sum(output, axis=1)
    return output.mean()


def custom_kl_div_mask(prediction, target, mask):
    output_pos = target * (target.clamp(min=1e-7).log() - prediction)
    zeros = torch.zeros_like(output_pos)
    output = torch.where(target > 0, output_pos, zeros)
    output = torch.sum(output, axis=1) * mask.float()
    return output.mean()


class JensenShannonDivergenceWeightedCustom(torch.nn.Module):
    def __init__(self, weights, lamb, num, num_classes):
        super(JensenShannonDivergenceWeightedCustom, self).__init__()
        self.num_classes = num_classes
        self.weights = [weights]
        self.weights.append((1-weights)/2)
        self.weights.append((1 - weights) / 2)
        assert abs(1.0 - sum(self.weights)) < 0.001
        print("JSWC loss~!!\tweights{}".format(self.weights))

    def forward(self, index, pred, labels):
        preds = list()
        if type(pred) == list:
            for i, p in enumerate(pred):
                preds.append(F.softmax(p, dim=1))
        else:
            preds.append(F.softmax(pred, dim=1))

        labels = F.one_hot(labels, self.num_classes).float()
        distribs = [labels] + preds
        assert len(self.weights) == len(distribs)

        mean_distrib = sum([w * d for w, d in zip(self.weights, distribs)])
        mean_distrib_log = mean_distrib.clamp(1e-7, 1.0).log()

        jsw = sum([w * custom_kl_div(mean_distrib_log, d)
                  for w, d in zip(self.weights, distribs)])
        return jsw


# ==================================================
# =================== Masked Loss ==================
# ==================================================
class ELR_loss_masked(nn.Module):
    def __init__(self, beta, lamb, num, cls):
        super(ELR_loss_masked, self).__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.ema = torch.zeros(num, cls).to(self.device)
        self.beta = beta
        self.lamb = lamb

    def forward(self, index, outputs, targets, mask):
        y_pred = torch.nn.functional.softmax(outputs, dim=1)
        y_pred = torch.clamp(y_pred, 1e-4, 1.0 - 1e-4)
        y_pred_ = y_pred.data.detach()
        # add mask
        index = index.to(self.device)


        self.ema[index] = self.beta * self.ema[index] + (1 - self.beta) * (
            (y_pred_) / (y_pred_).sum(dim=1, keepdim=True))
        elr_reg = ((1 - (self.ema[index] * y_pred).sum(dim=1)).log() * mask.float()).mean()
        final_loss = self.lamb * elr_reg
        return final_loss


class GCE_loss_masked(nn.Module):
    def __init__(self, beta=0.2, lamb=0.0, num=100, cls=12):
        super(GCE_loss_masked, self).__init__()
        self.q = beta
        self.sample_num = num
        self.class_num = cls
        print('Load GCE masked loss.')

    def forward(self, ind, outputs, targets, mask):
        targets = torch.zeros(targets.size(0), self.class_num).cuda().scatter_(
            1, targets.view(-1, 1), 1)
        pred = torch.nn.functional.softmax(outputs, dim=1)
        pred_y = torch.sum(targets * pred, dim=1)
        final_loss = torch.mean(
            ((1.0 - pred_y ** self.q) / self.q) * mask.float(), dim=0)
        return final_loss


class SL_loss_masked(nn.Module):
    def __init__(self, beta=1.0, lamb=1.0, num=100, cls=12):
        super(SL_loss_masked, self).__init__()
        self.q = beta
        self.b = lamb
        self.sample_num = num
        self.class_num = cls
        print('Load SL masked loss.')

    def forward(self, ind, outputs, targets, mask):
        targets_ = torch.zeros(targets.size(0), self.class_num).cuda().scatter_(
            1, targets.view(-1, 1), 1)
        pred = torch.nn.functional.softmax(outputs, dim=1)
        targets_ = torch.clamp(targets_, 1e-4, 1.0)
        final_loss = - \
            torch.mean(torch.sum(torch.log(targets_)*pred, dim=1)
                       * mask.float())*self.q
        return final_loss


class JensenShannonDivergenceWeightedCustom_masked(torch.nn.Module):
    def __init__(self, weights, lamb, num, num_classes):
        super(JensenShannonDivergenceWeightedCustom_masked, self).__init__()
        self.num_classes = num_classes
        # self.weights = [float(w) for w in weights.split(' ')]
        self.weights = [weights]
        self.weights.append((1-weights)/2)
        self.weights.append((1 - weights) / 2)
        assert abs(1.0 - sum(self.weights)) < 0.001
        print("JSWC masked loss~!!\tweights{}".format(self.weights))

    def forward(self, index, pred, labels, mask):
        preds = list()
        if type(pred) == list:
            for i, p in enumerate(pred):
                preds.append(F.softmax(p, dim=1))
        else:
            preds.append(F.softmax(pred, dim=1))

        labels = F.one_hot(labels, self.num_classes).float()
        distribs = [labels] + preds
        assert len(self.weights) == len(distribs)

        mean_distrib = sum([w * d for w, d in zip(self.weights, distribs)])
        mean_distrib_log = mean_distrib.clamp(1e-7, 1.0).log()

        jsw = sum([w * custom_kl_div_mask(mean_distrib_log, d, mask)
                  for w, d in zip(self.weights, distribs)])
        return jsw


