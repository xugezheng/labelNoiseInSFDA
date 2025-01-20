
import os
from torchvision import transforms
import random
from PIL import ImageFilter
from torch.utils.data import DataLoader
import common.vision.datasets as datasets


DSET2DATASET = {
    "domainnet": "DomainNet",
    "office-home": "OfficeHome",
    "office": "Office31",
    "VISDA-C": "VisDA2017",
}

# ==================== DATA LOADER ======================
# =======================================================


class TwoCropsTransform:
    def __init__(self, transform, transform1):
        self.transform = transform
        self.transform1 = transform
        self.transform_s = transform1

    def __call__(self, x):
        if self.transform is None:
            return x, x
        else:
            q = self.transform(x)
            k = self.transform1(x)
            p = self.transform_s(x)
            return [q, k, p]


class ThreeCropsTransform:
    def __init__(self, transform1, transform2, transform3):
        self.transformW = transform1
        self.transformS = transform2
        self.transform_base = transform3

    def __call__(self, x):
        if self.transformW is None:
            return x, x
        else:
            q = self.transformW(x)
            k = self.transformS(x)
            p = self.transform_base(x)
            return [q, k, p]
        


def image_train(resize_size=256, crop_size=224, alexnet=False):
    
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    return transforms.Compose(
        [
            transforms.Resize((resize_size, resize_size)),
            transforms.RandomCrop(crop_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]
    )


def image_train2(resize_size=256, crop_size=224, alexnet=False):
    

    return transforms.Compose(
        [
            transforms.Resize((resize_size, resize_size)),
            transforms.RandomCrop(crop_size),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(
                brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2
            ),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )


class GaussianBlur:
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[0.1, 2.0]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


def image_train_mocov2(resize_size=256, crop_size=224, alexnet=False):
   
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )


    return transforms.Compose(
        [
            transforms.RandomResizedCrop(224, scale=(0.2, 1.0)),
            transforms.RandomApply(
                [transforms.ColorJitter(0.8, 0.8, 0.5, 0.2)],
                p=0.8,
            ),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomRotation(degrees=[-2, 2]),
            transforms.RandomPosterize(8, p=0.2),
            transforms.RandomEqualize(p=0.2),
            transforms.RandomApply([GaussianBlur([0.1, 2])], p=0.5),
            # transforms.AugMix(5,5),           ## While Applying Augmix, comment out the ColorJitter
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]
    )


def image_test(resize_size=256, crop_size=224, alexnet=False):
    
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
   
    return transforms.Compose(
        [
            transforms.Resize((resize_size, resize_size)),
            transforms.CenterCrop(crop_size),
            transforms.ToTensor(),
            normalize,
        ]
    )



def data_load(args):
    # prepare data
    dset_loaders = {}
    train_bs = args.batch_size

    # "LABELNOISEINSFDA/DATASOURCE/"
    root_dir = os.path.join(args.root, args.dset)
    dataset = datasets.__dict__[DSET2DATASET[args.dset]]
    if args.dset == "domainnet":
        train_source_dataset = dataset(
            root=root_dir,
            task=args.target,
            r=0,
            split="train",
            download=False,
            list_name=args.list_name,
            transform=image_train(),
        )
        test_dataset = dataset(
            root=root_dir,
            task=args.target,
            r=0,
            split="test",
            download=False,
            list_name=args.list_name,
            transform=image_test(),
        )
       
        pl_dataset = dataset(
            root=root_dir,
            task=args.target,
            r=0,
            split="train",
            download=False,
            list_name=args.list_name,
            transform=image_test(),
        )
    else:
        train_source_dataset = dataset(
            root=root_dir,
            task=args.target,
            r=0,
            download=False,
            list_name=args.list_name,
            transform=image_train(),
        )
        test_dataset = dataset(
            root=root_dir,
            task=args.target,
            r=0,
            download=False,
            list_name=args.list_name,
            transform=image_test(),
        )
        pl_dataset = dataset(
            root=root_dir,
            task=args.target,
            r=0,
            download=False,
            list_name=args.list_name,
            transform=image_test(),
        )

    train_source_loader = DataLoader(
        train_source_dataset,
        batch_size=train_bs,
        shuffle=True,
        num_workers=args.num_workers,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=train_bs * 3,
        shuffle=False,
        num_workers=args.num_workers,
    )
    pl_loader = DataLoader(
        pl_dataset,
        batch_size=train_bs * 3,
        shuffle=False,
        num_workers=args.num_workers,
    )

    dset_loaders["target"] = train_source_loader
    dset_loaders["test"] = test_loader
    dset_loaders["pl"] = pl_loader


    num_classes = train_source_dataset.num_classes
    args.nb_classes = num_classes
    args.nb_samples = len(train_source_dataset)
    print("training samples size: ", args.nb_samples)
    args.out_file.write(
        f"Dataset: {args.dset}\nTask: {args.source} -> {args.target}; Training (target) sample size: {args.nb_samples}\n"
    )
    args.out_file.flush()

    return dset_loaders


def data_load_data_aug(args):
    # prepare data
    dset_loaders = {}
    train_bs = args.batch_size

    # "LABELNOISEINSFDA/DATASOURCE/"
    root_dir = os.path.join(args.root, args.dset)
    dataset = datasets.__dict__[DSET2DATASET[args.dset]]
    if args.dset == "domainnet":
        if args.is_data_aug == True:
          
            train_source_dataset = dataset(
                root=root_dir,
                task=args.target,
                r=0,
                split="train",
                download=False,
                list_name=args.list_name,
                transform=TwoCropsTransform(
                    image_train_mocov2(), image_train()),
            )
        else:
            # for simple lln = gjs
            train_source_dataset = dataset(
                root=root_dir,
                task=args.target,
                r=0,
                split="train",
                download=False,
                list_name=args.list_name,
                transform=TwoCropsTransform(image_train2(), image_train()),
            )
        pl_dataset = dataset(
            root=root_dir,
            task=args.target,
            r=0,
            split="train",
            download=False,
            list_name=args.list_name,
            transform=ThreeCropsTransform(
                image_train(), image_train_mocov2(), image_test()
            ),
        )

        test_dataset = dataset(
            root=root_dir,
            task=args.target,
            r=0,
            split="test",
            download=False,
            list_name=args.list_name,
            transform=image_test(),
        )
    else:
        if args.is_data_aug == True:
            train_source_dataset = dataset(
                root=root_dir,
                task=args.target,
                r=0,
                download=False,
                list_name=args.list_name,
                transform=TwoCropsTransform(
                    image_train_mocov2(), image_train()),
            )
        else:
            # for simple gjs
            train_source_dataset = dataset(
                root=root_dir,
                task=args.target,
                r=0,
                download=False,
                list_name=args.list_name,
                transform=TwoCropsTransform(image_train2(), image_train()),
            )
        pl_dataset = dataset(
            root=root_dir,
            task=args.target,
            r=0,
            download=False,
            list_name=args.list_name,
            transform=ThreeCropsTransform(
                image_train(), image_train_mocov2(), image_test()
            ),
        )
        test_dataset = dataset(
            root=root_dir,
            task=args.target,
            r=0,
            download=False,
            list_name=args.list_name,
            transform=image_test(),
        )
    
    train_source_loader = DataLoader(
        train_source_dataset,
        batch_size=train_bs,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=False,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=train_bs * 3,
        shuffle=False,
        num_workers=args.num_workers,
    )
    pl_loader = DataLoader(
        pl_dataset,
        batch_size=train_bs * 3,
        shuffle=False,
        num_workers=args.num_workers,
    )

    dset_loaders["target"] = train_source_loader
    dset_loaders["test"] = test_loader
    dset_loaders["pl"] = pl_loader


    num_classes = train_source_dataset.num_classes
    args.nb_classes = num_classes
    args.nb_samples = len(train_source_dataset)
    print("training samples size: ", args.nb_samples)
    args.out_file.write(
        f"Dataset: {args.dset}\nTask: {args.source} -> {args.target}; Training (target) sample size: {args.nb_samples}\n"
    )
    args.out_file.flush()

    return dset_loaders

