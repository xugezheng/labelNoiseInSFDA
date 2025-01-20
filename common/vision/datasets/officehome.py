"""
@author: Junguang Jiang
@contact: JiangJunguang1123@outlook.com
"""
import os
from typing import Optional
from .imagelist import ImageList
from ._util import download as download_data, check_exits


class OfficeHome(ImageList):
    """`OfficeHome <http://hemanthdv.org/OfficeHome-Dataset/>`_ Dataset.

    Args:
        root (str): Root directory of dataset
        task (str): The task (domain) to create dataset. Choices include ``'Ar'``: Art, \
            ``'Cl'``: Clipart, ``'Pr'``: Product and ``'Rw'``: Real_World.
        download (bool, optional): If true, downloads the dataset from the internet and puts it \
            in root directory. If dataset is already downloaded, it is not downloaded again.
        transform (callable, optional): A function/transform that  takes in an PIL image and returns a \
            transformed version. E.g, :class:`torchvision.transforms.RandomCrop`.
        target_transform (callable, optional): A function/transform that takes in the target and transforms it.

    .. note:: In `root`, there will exist following files after downloading.
        ::
            Art/
                Alarm_Clock/*.jpg
                ...
            Clipart/
            Product/
            Real_World/
            image_list/
                Art.txt
                Clipart.txt
                Product.txt
                Real_World.txt
    """

    download_list = [
        (
            "image_list",
            "image_list.zip",
            "https://cloud.tsinghua.edu.cn/f/ca3a3b6a8d554905b4cd/?dl=1",
        ),
        (
            "Art",
            "Art.tgz",
            "https://cloud.tsinghua.edu.cn/f/4691878067d04755beab/?dl=1",
        ),
        (
            "Clipart",
            "Clipart.tgz",
            "https://cloud.tsinghua.edu.cn/f/0d41e7da4558408ea5aa/?dl=1",
        ),
        (
            "Product",
            "Product.tgz",
            "https://cloud.tsinghua.edu.cn/f/76186deacd7c4fa0a679/?dl=1",
        ),
        (
            "Real_World",
            "Real_World.tgz",
            "https://cloud.tsinghua.edu.cn/f/dee961894cc64b1da1d7/?dl=1",
        ),
    ]
    image_list_nrc = {
        "Ar": "image_list_nrc/Art.txt",
        "Cl": "image_list_nrc/Clipart.txt",
        "Pr": "image_list_nrc/Product.txt",
        "Rw": "image_list_nrc/Real_World.txt",
    }
    image_list_partial = {
        "Ar": "image_list_partial/Art.txt",
        "Cl": "image_list_partial/Clipart.txt",
        "Pr": "image_list_partial/Product.txt",
        "Rw": "image_list_partial/Real_World.txt",
    }
    list_dict = {
                 "image_list_nrc": image_list_nrc,
                 "image_list": image_list_nrc,
                 "image_list_partial": image_list_partial, # shot class order
                 }
    
    # CLASSES = ['Drill', 'Exit_Sign', 'Bottle', 'Glasses', 'Computer', 'File_Cabinet', 'Shelf', 'Toys', 'Sink',
    #            'Laptop', 'Kettle', 'Folder', 'Keyboard', 'Flipflops', 'Pencil', 'Bed', 'Hammer', 'ToothBrush', 'Couch',
    #            'Bike', 'Postit_Notes', 'Mug', 'Webcam', 'Desk_Lamp', 'Telephone', 'Helmet', 'Mouse', 'Pen', 'Monitor',
    #            'Mop', 'Sneakers', 'Notebook', 'Backpack', 'Alarm_Clock', 'Push_Pin', 'Paper_Clip', 'Batteries', 'Radio',
    #            'Fan', 'Ruler', 'Pan', 'Screwdriver', 'Trash_Can', 'Printer', 'Speaker', 'Eraser', 'Bucket', 'Chair',
    #            'Calendar', 'Calculator', 'Flowers', 'Lamp_Shade', 'Spoon', 'Candles', 'Clipboards', 'Scissors', 'TV',
    #            'Curtains', 'Fork', 'Soda', 'Table', 'Knives', 'Oven', 'Refrigerator', 'Marker']
    # sorted as SHOT list target number
    CLASSES = [
        "Alarm_Clock",
        "Backpack",
        "Batteries",
        "Bed",
        "Bike",
        "Bottle",
        "Bucket",
        "Calculator",
        "Calendar",
        "Candles",
        "Chair",
        "Clipboards",
        "Computer",
        "Couch",
        "Curtains",
        "Desk_Lamp",
        "Drill",
        "Eraser",
        "Exit_Sign",
        "Fan",
        "File_Cabinet",
        "Flipflops",
        "Flowers",
        "Folder",
        "Fork",
        "Glasses",
        "Hammer",
        "Helmet",
        "Kettle",
        "Keyboard",
        "Knives",
        "Lamp_Shade",
        "Laptop",
        "Marker",
        "Monitor",
        "Mop",
        "Mouse",
        "Mug",
        "Notebook",
        "Oven",
        "Pan",
        "Paper_Clip",
        "Pen",
        "Pencil",
        "Postit_Notes",
        "Printer",
        "Push_Pin",
        "Radio",
        "Refrigerator",
        "Ruler",
        "Scissors",
        "Screwdriver",
        "Shelf",
        "Sink",
        "Sneakers",
        "Soda",
        "Speaker",
        "Spoon",
        "TV",
        "Table",
        "Telephone",
        "ToothBrush",
        "Toys",
        "Trash_Can",
        "Webcam",
    ]

    def __init__(
        self, root: str, task: str, r: float, download: Optional[bool] = False, list_name = "image_list", **kwargs
    ):
        self.selected_list = self.list_dict[list_name]
        assert task in self.selected_list
        data_list_file = os.path.join(root, self.selected_list[task])

        # if download:
        #     list(map(lambda args: download_data(root, *args), self.download_list))
        # else:
        #     list(map(lambda file_name: check_exits(root, file_name), self.download_list))

        super(OfficeHome, self).__init__(
            root, OfficeHome.CLASSES, data_list_file=data_list_file, r=r, **kwargs
        )

    @classmethod
    def domains(cls):
        return list(cls.image_list.keys())
