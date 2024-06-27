import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet101
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
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    trainset = torchvision.datasets.CIFAR100(root='./data', train=True,
                                             download=True, transform=transform)
    trainloader = DataLoader(trainset, batch_size=64, shuffle=True, num_workers=4)

    testset = torchvision.datasets.CIFAR100(root='./data', train=False,
                                            download=True, transform=transform)
    testloader = DataLoader(testset, batch_size=64, shuffle=False, num_workers=4)

    model = resnet101(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 100)  # Adjusting the classifier for CIFAR-100

    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.0001)

    writer = SummaryWriter('runs/cifar_100_resnet101')

    checkpoint_path = "resnet101_cifar100_checkpoint.pth"
    start_epoch = 1

    # Load checkpoint if exists
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"Checkpoint loaded, starting from epoch {start_epoch}")

    epochs = 45
    for epoch in range(start_epoch, epochs + 1):
        model.train()
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(trainloader):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            if np.random.rand() < 0.5:  # Applying CutMix with 50% probability
                inputs, targets = cutmix(inputs, labels, alpha=1.0)
                outputs = model(inputs)
                loss = criterion(outputs, targets[0]) * targets[2] + criterion(outputs, targets[1]) * (1. - targets[2])
            else:
                outputs = model(inputs)
                loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 200 == 199:  # Logging the running loss
                writer.add_scalar('training loss',
                                  running_loss / 200,
                                  epoch * len(trainloader) + i)
                print(f'[{epoch}/{epochs}, {i + 1}] loss: {running_loss / 200:.3f}')
                running_loss = 0.0

        if epoch % 5 == 0: # Save checkpoint every 5 epochs
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': running_loss,
            }, checkpoint_path)
            print(f"Checkpoint saved at epoch {epoch}")

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in testloader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        writer.add_scalar('Accuracy', accuracy, epoch)
        print(f'\nTest set: Accuracy: {correct}/{total} ({accuracy:.2f}%)\n')

    writer.close()
    print('Finished Training')
    torch.save(model.state_dict(), "resnet101_cifar100_final.pth")