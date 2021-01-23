import torch
from tqdm import tqdm
import torch.nn.functional as F


class TestCifar10(object):
    def __init__(self, device, model, test_loader):
        self.test_acc = []
        self.test_losses = []
        self.device = device
        self.test_loader = test_loader
        self.model = model

    def test(self):
        self.model.eval()
        test_loss = 0
        correct = 0
        p_bar = tqdm(self.test_loader)
        with torch.no_grad():
            for batch_idx, (data, labels) in enumerate(p_bar):
                data, labels = data.to(self.device), labels.to(self.device)
                output = self.model(data)
                test_loss += F.nll_loss(output, labels,
                                        reduction='sum').item()  # sum up batch loss
                pred = output.argmax(dim=1,
                                     keepdim=True)  # get the index of the max log-probability
                correct += pred.eq(labels.view_as(pred)).sum().item()

        test_loss /= len(self.test_loader.dataset)
        self.test_losses.append(test_loss)
        print(
            '\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
                test_loss, correct, len(self.test_loader.dataset),
                100. * correct / len(self.test_loader.dataset)))
        self.test_acc.append(100. * correct / len(self.test_loader.dataset))
