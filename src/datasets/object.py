from torch.utils.data import Subset
from PIL import Image
from torchvision.datasets import ImageFolder
from base.torchvision_dataset import TorchvisionDataset
from .preprocessing import get_target_label_idx, global_contrast_normalization
import numpy as np
import torchvision.transforms as transforms
import torch

class OBJECT_Dataset(TorchvisionDataset):

    def __init__(self, root: str, normal_class):
        super().__init__(root)
        Mvtec_list = ['carpet', 'grid', 'leather', 'tile', 'wood', 'bottle', 'cable', 'capsule', 'hazelnut', 'metal_nut',
                     'pill', 'screw', 'toothbrush', 'transistor', 'zipper']

        self.n_classes = 2  # 0: normal, 1: outlier
        self.normal_classes = normal_class
        # for i in range(len(normal_class)):
        #     if not (normal_class[i] == '[' or normal_class[i] == ']' or normal_class[i] == ','):
        #         self.normal_classes.append(int(normal_class[i]))
        self.root = self.root+Mvtec_list[self.normal_classes]

        mean = [(0.36607818518032215, 0.3528722483374472, 0.3585191239764038),#0
                (0.4487305946663354, 0.4487305946663354, 0.4487305946663354),#1
                (0.3923340318128373, 0.26295472525674995, 0.22025334692657814),#2
                (0.4536255693657713, 0.4682865838881645, 0.4452575836280415),#3
                (0.672454086143443, 0.4779993567370712, 0.35007702036667776),#4
                (0.5352967021800805, 0.5314880132137422, 0.547828897157147),#5
                (0.3267409463643222, 0.41484389522093523, 0.46695618025405883),#6
                (0.6926364358307354, 0.662149771557822, 0.6490556404776292),#7
                (0.24011281595607017, 0.1769201147939173, 0.17123964257174726),#8
                (0.21251877631977975, 0.23440739849813622, 0.2363959074824541),#9
                (0.3025230547246622, 0.30300693821061303, 0.32466943588225744),#10
                (0.7214971293922232, 0.7214971293922232, 0.7214971293922232 ),#11
                (0.20453672401964704, 0.19061953742573437, 0.1973630989492544),#12
                (0.38709726938081024, 0.27680750921869235, 0.24161576675737736),#13
                (0.39719792798156195, 0.39719792798156195, 0.39719792798156195),#14
                ]
        std =  [(0.1334089197933497, 0.13091438558839882, 0.11854704285817017),  # 0
                (0.16192189716258867, 0.16192189716258867, 0.16192189716258867),  # 1
                (0.0527090063203568, 0.035927180158353854, 0.026535684323885065),  # 2
                (0.11774565267141425, 0.13039328961987165, 0.12533147519872007),  # 3
                (0.07714836895006975, 0.06278302787607731, 0.04349760909698915),  # 4
                (0.36582285759516936, 0.3661720233895615, 0.34943018535446296),   # 5
                (0.14995070226373788, 0.2117666336616603, 0.23554648659289779),   # 6
                (0.23612927993223184, 0.25644744015075704, 0.25718179933681784),  # 7
                (0.168789697373752, 0.07563237349131141, 0.043146545992581754),   # 8
                (0.15779873915363898, 0.18099161937329614, 0.15159372072430388),  # 9
                (0.15720102988319967, 0.1803989691876269, 0.15113407058442763),     # 10
                (0.13265686578689692, 0.13265686578689692, 0.13265686578689692),  # 11
                (0.2316392849251032, 0.21810285502082638, 0.19743939091294657),  # 12
                (0.20497542590257026, 0.14190994609091834, 0.11531548927488476),  # 13
                (0.3185215984033291, 0.3185215984033291, 0.3185215984033291),  # 14
                ]

        transform_train = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize(mean[self.normal_classes], std[self.normal_classes]),
                                        ])

        transform_test = transforms.Compose([transforms.Resize(256),
                                              transforms.ToTensor(),
                                              transforms.Normalize(mean[self.normal_classes], std[self.normal_classes])])
        self.train_set = ImageFolder(root=self.root +'/train',transform =transform_train )
        self.test_set = ImageFolder(root=self.root + '/test', transform=transform_test)
        self.test_data = []
        for i in range(len(self.test_set.imgs)):
            self.test_data.append(self.test_set[i][0].numpy())
        self.test_data = np.array(self.test_data)
        self.test_data = torch.from_numpy(self.test_data)


    def compute_mean_and_std(self):

            import os
            import cv2
            import numpy as np
            from torch.utils.data import Dataset
            from PIL import Image


            # 输入PyTorch的dataset，输出均值和标准差
            mean_r = 0
            mean_g = 0
            mean_b = 0

            for img, _ in self.train_set:
                img = np.asarray(img)  # change PIL Image to numpy array
                mean_b += np.mean(img[:, :, 0])
                mean_g += np.mean(img[:, :, 1])
                mean_r += np.mean(img[:, :, 2])

            mean_b /= len(self.train_set)
            mean_g /= len(self.train_set)
            mean_r /= len(self.train_set)

            diff_r = 0
            diff_g = 0
            diff_b = 0

            N = 0

            for img, _ in self.train_set:
                img = np.asarray(img)

                diff_b += np.sum(np.power(img[:, :, 0] - mean_b, 2))
                diff_g += np.sum(np.power(img[:, :, 1] - mean_g, 2))
                diff_r += np.sum(np.power(img[:, :, 2] - mean_r, 2))

                N += np.prod(img[:, :, 0].shape)

            std_b = np.sqrt(diff_b / N)
            std_g = np.sqrt(diff_g / N)
            std_r = np.sqrt(diff_r / N)

            mean = (mean_b.item() / 255.0, mean_g.item() / 255.0, mean_r.item() / 255.0)
            std = (std_b.item() / 255.0, std_g.item() / 255.0, std_r.item() / 255.0)
            return mean, std


class MyOBJECT(ImageFolder):
    """Torchvision MNIST class with patch of __getitem__ method to also return the index of a data sample."""

    def __init__(self, *args, **kwargs):
        super(MyOBJECT, self).__init__(*args, **kwargs)

    def __getitem__(self, index):
        """Override the original method of the MNIST class.
        Args:
            index (int): Index
        Returns:
            triple: (image, target, index) where target is index of the target class.
        """

        img, target = self.imgs[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img.numpy(), mode='L')

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, index  # only line changed

