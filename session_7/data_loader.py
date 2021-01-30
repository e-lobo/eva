import torch
import torchvision

import torchvision.transforms as transforms


class Loader(object):
    def __init__(self, net, custom_transforms):
        super(Loader, self).__init__()
        self.cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.cuda else "cpu")

        self.model = net().to(self.device)

        transform = transforms.Compose(custom_transforms)

        train_set = torchvision.datasets.CIFAR10(root='./data', train=True,
                                                 download=True,
                                                 transform=transform)
        self.train_loader = torch.utils.data.DataLoader(train_set,
                                                        batch_size=64,
                                                        shuffle=True,
                                                        num_workers=4)

        test_set = torchvision.datasets.CIFAR10(root='./data', train=False,
                                                download=True,
                                                transform=transform)
        self.test_loader = torch.utils.data.DataLoader(test_set, batch_size=64,
                                                       shuffle=False,
                                                       num_workers=4)
