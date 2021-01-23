import torch
import torchvision

from session_7.model import Net
import torchvision.transforms as transforms


class Loader(object):
    def __init__(self, data_mean, data_std_dev):
        super(Loader, self).__init__()
        self.cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.cuda else "cpu")

        self.model = Net().to(self.device)

        transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize(data_mean, data_std_dev)])

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
