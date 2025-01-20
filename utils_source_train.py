import torch
import numpy as np
import random
from PIL import Image
from torch.utils.data import Dataset
import os
import os.path
import torch.nn as nn

# https://github.com/tim-learn/SHOT/blob/master/object/data_list.py
def make_dataset(image_list, labels):
    if labels:
      len_ = len(image_list)
      images = [(image_list[i].strip(), labels[i, :]) for i in range(len_)]
    else:
      if len(image_list[0].split()) > 2:
        images = [(val.split()[0], np.array([int(la) for la in val.split()[1:]])) for val in image_list]
      else:
        images = [(val.split()[0], int(val.split()[1])) for val in image_list]
    return images

def make_dataset_imb(image_list, labels, cfg):
    images = []
    if cfg.dset == 'VISDA-RSUT':
        root_dir = os.path.join(cfg.root, 'VISDA-C')
    else:
        root_dir = os.path.join(cfg.root, cfg.dset)
    if labels:
        len_ = len(image_list)
        images = [(image_list[i].strip(), labels[i, :]) for i in range(len_)]
    else:
        if len(image_list[0].split()) > 2:
            images = [(val.split()[0], np.array([int(la) for la in val.split()[1:]])) for val in image_list]
        else:
            for val in image_list:
                new_path = os.path.join(root_dir, '', val.split()[0])
                images.append((new_path, int(val.split()[1])))
    return images

def path_merge(imgs, root):
    imgs_new = []
    for i in range(len(imgs)):
        path, tar = imgs[i]
        if not os.path.isabs(path):
            path = os.path.join(root, '', path)
        imgs_new.append((path, tar))
    return imgs_new
            


def rgb_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')

def l_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('L')

class ImageList(Dataset):
    def __init__(self, image_list, labels=None, transform=None, target_transform=None, mode='RGB', root=None):
        imgs = make_dataset(image_list, labels)
        if root is not None:
            imgs = path_merge(imgs, root)
        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in subfolders of: " + root + "\n"))

        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
        if mode == 'RGB':
            self.loader = rgb_loader
        elif mode == 'L':
            self.loader = l_loader

    def __getitem__(self, index):
        path, target = self.imgs[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.imgs)

class ImageList_imb(Dataset):
    """Conventional Random Sampler or Class-balanced Sampler
    Percentage-unaware
    """
    def __init__(self, image_list, labels=None, transform=None, target_transform=None, mode='RGB', cfg=None,
                 balance_sample=True):
        """
        Initialize the ImageList class
        :param image_list: list of image paths
        :param labels: tensor of labels
        :param transform: transform for loaded images
        :param target_transform: target-image-specific transform
        :param mode: how to load images of a certain format
        :param cfg: arguments regarding training and dataset
        :param balance_sample: if True: balanced sampling, else: random sampling
        """
        imgs = make_dataset_imb(image_list, labels, cfg)
        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in subfolders of: " + {cfg.root} + "\n"))

        self.imgs = imgs
        assert cfg != None, 'Have not passed arguments needed.'
        self.cls_num = cfg.class_num
        self.balance_sample = balance_sample
        self.transform = transform
        self.target_transform = target_transform
        if mode == 'RGB':
            self.loader = rgb_loader
        elif mode == 'L':
            self.loader = l_loader
        self.class_dict = self._get_class_dict()  # each cls's samples' id

    def get_annotations(self):
        annos = []
        for (img_path, lb) in self.imgs:
            annos.append({'category_id': int(lb)})
        return annos

    def _get_class_dict(self):
        class_dict = dict()
        for i, anno in enumerate(self.get_annotations()):
            cat_id = anno["category_id"]
            if not cat_id in class_dict:
                class_dict[cat_id] = []
            class_dict[cat_id].append(i)
        return class_dict


    def __getitem__(self, index):
        if self.balance_sample:
            """Balanced Sampling
            step1. Select one class from uniform distribution.
            step2. Randomly select one sample from the selected class.
            """
            sample_class = random.randint(0, self.cls_num - 1)
            sample_indexes = self.class_dict[sample_class]
            index = random.choice(sample_indexes)

        path, target = self.imgs[index]

        img = self.loader(path)

        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.imgs)

# https://github.com/tim-learn/SHOT/blob/master/object/loss.py
class CrossEntropyLabelSmooth(nn.Module):
    """Cross entropy loss with label smoothing regularizer.
    Reference:
    Szegedy et al. Rethinking the Inception Architecture for Computer Vision. CVPR 2016.
    Equation: y = (1 - epsilon) * y + epsilon / K.
    Args:
        num_classes (int): number of classes.
        epsilon (float): weight.
    """

    def __init__(self, num_classes, epsilon=0.1, use_gpu=True, reduction=True):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.use_gpu = use_gpu
        self.reduction = reduction
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets):
        """
        Args:
            inputs: prediction matrix (before softmax) with shape (batch_size, num_classes)
            targets: ground truth labels with shape (num_classes)
        """
        log_probs = self.logsoftmax(inputs)
        targets = torch.zeros(log_probs.size()).scatter_(1, targets.unsqueeze(1).cpu(), 1)
        if self.use_gpu: targets = targets.cuda()
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        loss = (- targets * log_probs).sum(dim=1)
        if self.reduction:
            return loss.mean()
        else:
            return loss

    
    
    
