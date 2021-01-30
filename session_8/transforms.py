import torchvision.transforms as transforms


def custom_transforms(data_mean, data_std_dev):
    return [transforms.RandomRotation(15),
            transforms.RandomResizedCrop(32, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(data_mean, data_std_dev)]
