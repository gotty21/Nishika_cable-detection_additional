import cv2
import numpy as np

from PIL import Image


def pil2cv(img):
    new_img = np.array(img, dtype=np.uint8)

    if new_img.ndim == 2:       # モノクロ
        pass
    if new_img.shape[2] == 3:   # カラー
        new_img = cv2.cvtColor(new_img, cv2.COLOR_RGB2BGR)
    if new_img.shape[2] == 4:   # 透過
        new_img = cv2.cvt.Color(new_img, cv2.COLOR_RGBA2BGRA)

    return img

def resize(img, size=512):
    if type(img) == 'PIL':
        img = pil2cv(img)

    # min-max normalize
    img = img - np.min(img)
    img = img / np.max(img)

    # 0-255
    img = (img * 255).astype(np.uint8)

    # resize
    img = cv2.resize(img, (size, size))

    return img


class Preprosessing():
    def __init__(self):
        pass

    def __call__(self, img):
        img = pil2cv(img)
        img = resize(img)
        return img


if __name__ == '__main__':
    img = Image.open('./img/test_cable_img.jpg')
    p = Preprosessing()
    img = p(img)

    print(type(img))
    print(img.shape)





