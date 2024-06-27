import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision.models import vit_b_16
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import numpy as np

def cutmix(data, targets, alpha=1.0):
    indices = torch.randperm(data.size(0))
    shuffled_data = data[indices]
    shuffled_targets = targets[indices]

    lam = np.random.beta(alpha, alpha)
    image_h, image_w = data.shape[2:]
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(image_w * cut_rat)
    cut_h = int(image_h * cut_rat)

    cx = np.random.randint(image_w)
    cy = np.random.randint(image_h)

    bbx1 = np.clip(cx - cut_w // 2, 0, image_w)
    bby1 = np.clip(cy - cut_h // 2, 0, image_h)
    bbx2 = np.clip(cx + cut_w // 2, 0, image_w)
    bby2 = np.clip(cy + cut_h // 2, 0, image_h)

    data[:, :, bby1:bby2, bbx1:bbx2] = shuffled_data[:, :, bby1:bby2, bbx1:bbx2]

    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (data.size()[-1] * data.size()[-2]))
    targets = (targets, shuffled_targets, lam)

    return data, targets

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    trainset = torchvision.datasets.CIFAR100(root='./data', train=True,
                                             download=True, transform=transform)
    trainloader = DataLoader(trainset, batch_size=64, shuffle=True, num_workers=4)

    testset = torchvision.datasets.CIFAR100(root='./data', train=False,
                                            download=True, transform=transform)
    testloader = DataLoader(testset, batch_size=64, shuffle=False, num_workers=4)

    checkpoint_path = "vit2_cifar100_checkpoint.pth"
    start_epoch = 1

    writer = SummaryWriter('runs/cifar100_vit2')

    model = vit_b_16(pretrained=False)
    pretrained_weights_path = './vit_base_patch16_224_miil_21k.pth'
    model.load_state_dict(torch.load(pretrained_weights_path), strict=False)

    model.heads = nn.Sequential(
        nn.Linear(model.heads.head.in_features, 100)
    )

    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001,weight_decay=0.0001)

    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1

    epochs = 25
    for epoch in range(start_epoch, epochs + 1):
        model.train()
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            if np.random.rand() < 0.5:
                inputs, targets = cutmix(inputs, labels, alpha=1.0)
                outputs = model(inputs)
                loss = criterion(outputs, targets[0]) * targets[2] + criterion(outputs, targets[1]) * (1. - targets[2])
            else:
                outputs = model(inputs)
                loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 200 == 199:
                print(f'[{epoch}, {i + 1:5d}] loss: {running_loss / 200:.3f}')
                writer.add_scalar('training loss', running_loss / 200, epoch * len(trainloader) + i)
                running_loss = 0.0

        model.eval()
        test_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for data in testloader:
                images, labels = data
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                test_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        test_loss /= len(testloader)
        accuracy = correct / total

        print(f'\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{total} ({100. * accuracy:.2f}%)\n')
        writer.add_scalar('validation loss', test_loss, epoch)
        writer.add_scalar('validation accuracy', accuracy, epoch)

        if epoch % 5 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, checkpoint_path)

    torch.save(model.state_dict(), "vit2_cifar100_final.pth")
    writer.close()
    print('Finished Training')