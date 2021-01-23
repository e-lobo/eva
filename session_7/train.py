import torch.optim as optim
from tqdm import tqdm
import torch.nn.functional as F


class TrainCifar10(object):
    def __init__(self, device, model, train_loader):
        self.device = device
        self.train_loader = train_loader
        self.model = model
        self.train_acc = []
        self.train_losses = []

    def train(self, limit, test_cifar, learning_rate=0.01, momentum=0.9):
        optimizer = optim.SGD(self.model.parameters(), lr=learning_rate,
                              momentum=momentum)

        for epoch in range(limit):
            self.model.train()
            p_bar = tqdm(self.train_loader)
            correct = 0
            processed = 0

            print("EPOCH:", epoch+1)

            for batch_idx, (data, labels) in enumerate(p_bar):
                data, labels = data.to(self.device), labels.to(self.device)

                # Init
                optimizer.zero_grad()
                # In PyTorch, we need to set the gradients to zero before starting to do backpropagation because PyTorch accumulates the gradients on subsequent backward passes.
                # Because of this, when you start your training loop, ideally you should zero out the gradients so that you do the parameter update correctly.

                # Predict
                y_pred = self.model(data)

                # Calculate loss
                loss = F.nll_loss(y_pred, labels)
                self.train_losses.append(loss)

                # Backpropagation
                loss.backward()
                optimizer.step()

                # Update p_bar-tqdm
                pred = y_pred.argmax(dim=1,
                                     keepdim=True)  # get the index of the max log-probability
                correct += pred.eq(labels.view_as(pred)).sum().item()
                processed += len(data)

                p_bar.set_description(
                    desc=f'Loss={loss.item()} Batch_id={batch_idx} Accuracy={100 * correct / processed:0.2f}')
                self.train_acc.append(100 * correct / processed)
            test_cifar.test()
