import os
import random

from PIL import Image


def resize_image(input_image_path, shape = (128, 128)):
    original_image = Image.open(input_image_path)
    width, height = original_image.size
    resized_image = None
    try:
        resized_image = original_image.resize(shape, Image.ANTIALIAS)
        resized_image.save(input_image_path)
    except OSError:
        print(input_image_path)
    except ValueError:
        print(input_image_path)


def resize_all_image_in_path(path):
    cnt = 0
    for name in os.listdir(path):
        resize_image(path + name)
        cnt += 1
        if cnt % 50 == 0:
            print(cnt, "image resized")


def shuf_images(path):
    cnt = 0
    lst = []
    lst = [i for i in range(1426)]
    shuff_lst = sorted(lst, key=lambda A: random.random())
    new_path = path + 'new/'
    for name in os.listdir(path):
        if name == '1' or name == 'path':
            continue
        image = Image.open(path + name)
        if cnt % 50 == 0:
            print(cnt, "image renamed")
        try:
            image.save(os.path.join(new_path, str(shuff_lst[cnt])) + '.JPG', 'JPEG')
        except Exception:
            print(name)
        # os.remove(name)
        cnt += 1

