import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


transform = transforms.Compose([
    transforms.Resize((40, 40)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

batch_size = 4

testset = torchvision.datasets.ImageFolder(root='./datasets/CAPTCHA/test', transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=True, num_workers=2)

classes = [str(i) for i in range(0, 10)] + [chr(i) for i in range(ord('a'), ord('z') + 1)]
classes.remove("0")
classes.remove("1")
classes.remove("i")
classes.remove("l")
classes.remove("o")
print(len(classes))

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 7 * 7, 392)
        self.fc2 = nn.Linear(392, 196)
        self.fc3 = nn.Linear(196, 31)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

if __name__ == '__main__':
    model = Net()
    model.load_state_dict(torch.load('./cifar_net.pth'))

    model.eval()

    # get some random training images
    dataiter = iter(testloader)
    images, labels = next(dataiter)
    x, y = images[0], labels[0]
    testloader
    with torch.no_grad():
        # 添加批次维度
        x_with_batch = x.unsqueeze(0)
        pred = model(x_with_batch)
        predicted, actual = classes[pred[0].argmax(0)], classes[y]
        print(f'Predicted: "{predicted}", Actual: "{actual}"')
