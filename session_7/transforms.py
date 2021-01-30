import torchvision.transforms as transforms


def custom_transforms(data_mean, data_std_dev):
    return [transforms.ToTensor(),
            transforms.Normalize(data_mean, data_std_dev)]
