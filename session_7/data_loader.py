import torch
import torchvision

import torchvision.transforms as transforms


class Loader(object):
    def __init__(self, net, train_transforms, test_transforms, **kwargs):
        super(Loader, self).__init__()
        self.cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.cuda else "cpu")

        self.model = net().to(self.device)

        train_transforms = transforms.Compose(train_transforms)

        test_transforms = transforms.Compose(test_transforms)

        train_set = torchvision.datasets.CIFAR10(root='./data', train=True,
                                                 download=True,
                                                 transform=train_transforms)
        self.train_loader = torch.utils.data.DataLoader(train_set,
                                                        batch_size=64,
                                                        shuffle=True,
                                                        num_workers=4)

        test_set = torchvision.datasets.CIFAR10(root='./data', train=False,
                                                download=True,
                                                transform=test_transforms)
        self.test_loader = torch.utils.data.DataLoader(test_set, batch_size=64,
                                                       shuffle=False,
                                                       num_workers=4)
