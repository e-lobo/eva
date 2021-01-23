import matplotlib.pyplot as plt
import torchvision
import numpy as np
from torchsummary import summary

DATA_MEAN = (0.4914, 0.4822, 0.4465)
DATA_STD_DEV = (0.2470, 0.2435, 0.2616)
LEARNING_RATE = 0.05
MOMENTUM = 0.9
EPOCH = 40

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


def display_image(train_loader):
    def show_image(img):
        img = img / 2 + 0.5
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))

    # get some random training images
    data_iter = iter(train_loader)
    images, labels = data_iter.next()

    # show images
    show_image(torchvision.utils.make_grid(images))
    # print labels
    print(' '.join('%5s' % classes[labels[j]] for j in range(64)))


def print_sum(model):
    summary(model, input_size=(3, 32, 32))
