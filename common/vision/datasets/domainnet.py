"""
@author: Junguang Jiang
@contact: JiangJunguang1123@outlook.com
"""
import os
from typing import Optional
from .imagelist import ImageList
from ._util import download as download_data, check_exits


class DomainNet(ImageList):
    """`DomainNet <http://ai.bu.edu/M3SDA/#dataset>`_ (cleaned version, recommended)

    See `Moment Matching for Multi-Source Domain Adaptation <https://arxiv.org/abs/1812.01754>`_ for details.

    Args:
        root (str): Root directory of dataset
        task (str): The task (domain) to create dataset. Choices include ``'c'``:clipart, \
            ``'i'``: infograph, ``'p'``: painting, ``'q'``: quickdraw, ``'r'``: real, ``'s'``: sketch
        split (str, optional): The dataset split, supports ``train``, or ``test``.
        download (bool, optional): If true, downloads the dataset from the internet and puts it \
            in root directory. If dataset is already downloaded, it is not downloaded again.
        transform (callable, optional): A function/transform that  takes in an PIL image and returns a \
            transformed version. E.g, :class:`torchvision.transforms.RandomCrop`.
        target_transform (callable, optional): A function/transform that takes in the target and transforms it.

    .. note:: In `root`, there will exist following files after downloading.
        ::
            clipart/
            infograph/
            painting/
            quickdraw/
            real/
            sketch/
            image_list/
                clipart.txt
                ...
    """
    download_list = [
        ("image_list", "image_list.zip", "https://cloud.tsinghua.edu.cn/f/90ecb35bbd374e5e8c41/?dl=1"),
        ("clipart", "clipart.zip", "http://csr.bu.edu/ftp/visda/2019/multi-source/groundtruth/clipart.zip"),
        ("infograph", "infograph.zip", "http://csr.bu.edu/ftp/visda/2019/multi-source/infograph.zip"),
        ("painting", "painting.zip", "http://csr.bu.edu/ftp/visda/2019/multi-source/groundtruth/painting.zip"),
        ("quickdraw", "quickdraw.zip", "http://csr.bu.edu/ftp/visda/2019/multi-source/quickdraw.zip"),
        ("real", "real.zip", "http://csr.bu.edu/ftp/visda/2019/multi-source/real.zip"),
        ("sketch", "sketch.zip", "http://csr.bu.edu/ftp/visda/2019/multi-source/sketch.zip"),
    ]
    image_list = {
        "c": "clipart",
        "i": "infograph",
        "p": "painting",
        "q": "quickdraw",
        "r": "real",
        "s": "sketch",
    }
    CLASSES = ['tree', 'golf_club', 'squirrel', 'dog', 'whale', 'spreadsheet', 'snowman', 'tiger',
               'table', 'shoe', 'windmill', 'submarine', 'feather', 'bird', 'spider', 'strawberry',
               'nail', 'beard', 'bread', 'train', 'watermelon', 'zebra', 'sheep', 'elephant',
               'teapot', 'eye', 'mushroom', 'sea_turtle', 'sword', 'streetlight', 'lighthouse', 'owl',
               'horse', 'penguin', 'pond', 'sock', 'snorkel', 'helicopter', 'snake', 'butterfly']

    def __init__(self, root: str, task: str, split: Optional[str] = 'train', download: Optional[float] = False, list_name = "image_list",
                 **kwargs):
        assert task in self.image_list
        assert split in ['train', 'test']
        data_list_file = os.path.join(root, "image_list", "{}_{}.txt".format(self.image_list[task], split))
        print("loading {}".format(data_list_file))

        # if download:
        #     list(map(lambda args: download_data(root, *args), self.download_list))
        # else:
        #     list(map(lambda args: check_exits(root, args[0]), self.download_list))

        super(DomainNet, self).__init__(root, DomainNet.CLASSES, data_list_file=data_list_file, **kwargs)

    @classmethod
    def domains(cls):
        return list(cls.image_list.keys())
