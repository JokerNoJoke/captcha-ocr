import os
import requests
from io import BytesIO
from PIL import Image, ImageFile
from datetime import date

def init_folders():
    numbers = [str(i) for i in range(0, 10)]
    letters = [chr(i) for i in range(ord('a'), ord('z') + 1)]

    if not os.path.exists(base_path):
        os.makedirs(base_path)

    labels_count = {}
    for item in numbers + letters:
        folder_path = os.path.join(base_path, item)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        labels_count[item] = len(os.listdir(folder_path))
    return labels_count

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

def generate(labels_count, label_max_count, loop):
    today = date.today()
    for _ in range(loop):
        captcha, pil_image = getImage()
        labels, pil_images = horizontally_split_image(captcha, pil_image)
        for i in range(5):
            label = labels[i].lower()
            pil_image = pil_images[i]
            num = labels_count[label]
            if num >= label_max_count:
                continue
            pil_image.save(f'{base_path}/{label}/{today}-{num}.png')
            labels_count[label] = num + 1

base_path = './datasets/CAPTCHA/test'
labels_count = init_folders()
label_max_count = 8
generate(labels_count, label_max_count, 1000)
for lable, count in labels_count.items():
    if count < label_max_count:
        print(lable, count)
