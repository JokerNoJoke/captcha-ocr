import requests
from io import BytesIO
from PIL import Image, ImageFile
import torchvision.transforms as transforms
import torch
import torch.nn as nn
import torch.nn.functional as F


def getImage():
    response = requests.get("http://127.0.0.1:8080/captcha")
    captcha = response.headers["CAPTCHA"]
    pil_image = Image.open(BytesIO(response.content))
    return captcha, pil_image

def horizontally_split_image(captcha: str, pil_image: ImageFile.ImageFile):
    width, height = pil_image.size
    piece_width = width // 5
    labels = []
    pil_images = []
    for i in range(5):
        labels.append(captcha[i])
        left = i * piece_width
        upper = 0
        right = left + piece_width
        lower = height
        pil_image2 = pil_image.crop((left, upper, right, lower))
        pil_images.append(pil_image2)
    return labels, pil_images

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
    
classes = [str(i) for i in range(0, 10)] + [chr(i) for i in range(ord('a'), ord('z') + 1)]
classes.remove("0")
classes.remove("1")
classes.remove("i")
classes.remove("l")
classes.remove("o")
# print(len(classes))

captcha, pil_image = getImage()
print("Actual: " + captcha)
labels, pil_images = horizontally_split_image(captcha, pil_image)

transform = transforms.Compose([
    transforms.Resize((40, 40)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

tensor_images = []
for pil_image in pil_images:
    tensor_image = transform(pil_image)
    tensor_images.append(tensor_image)

inputs = torch.stack(tensor_images)

net = Net()
net.load_state_dict(torch.load('./cifar_net.pth'))

outputs = net(inputs)

_, predicted = torch.max(outputs, 1)

print('Predicted: ', ' '.join(f'{classes[predicted[j]]:5s}' for j in range(len(outputs))))