import os
import numpy as np
import random

from skimage import io, color, util
from skimage.draw import random_shapes
from skimage.transform import resize
from skimage.util import img_as_ubyte, img_as_float64

def generateMasks(path, amount):
    for i in range(amount):
        mask, _ = random_shapes((512, 512), 30, 20, 2, 60, channel_axis=None, shape="ellipse", intensity_range=(0, 0))
        mask = util.invert(mask)
        io.imsave(os.path.join(path, "mask{}.png".format(i + 1)), mask, check_contrast=False)

def generateDefectivePhotos(path):

    photo_list = sorted(os.listdir(os.path.join(path, "raw")), key=lambda f: int("".join(filter(str.isdigit, f))))

    mask_list = sorted(os.listdir(os.path.join(path, "mask")), key=lambda f: int("".join(filter(str.isdigit, f))))
    handmade_mask_list = [item for item in mask_list if "handmade" in item]
    for item in handmade_mask_list:
        mask_list.remove(item)

    for i in range(len(photo_list)):
        print("Start processing image {}".format(i + 1))
        img = io.imread(os.path.join(path, "raw", photo_list[i]))

        # Gray to RGB conversion
        if len(img.shape) < 3:
            img = color.gray2rgb(img)
        # RGBA to RGB conversion
        elif img.shape[2] > 3:
            img = color.rgba2rgb(img)
        
        img = img_as_float64(img)

        if random.random() < 0.1:
            index = random.randint(0, len(handmade_mask_list) - 1)
            mask = io.imread(os.path.join(path, "mask", handmade_mask_list[index]))
        else:
            index = random.randint(0, len(mask_list) - 1)
            mask = io.imread(os.path.join(path, "mask", mask_list[index]))

        # Gray to RGB conversion
        if len(mask.shape) < 3:
            mask = color.gray2rgb(mask)
        # RGBA to RGB conversion
        elif mask.shape[2] > 3:
            mask = color.rgba2rgb(mask)

        if img.shape != mask.shape:
            mask = resize(mask, (img.shape[0], img.shape[1]))

        mask = img_as_ubyte(mask)
        
        bool_mask = mask > 220

        np.putmask(img, bool_mask, 1)

        img = img_as_ubyte(img)

        io.imsave(os.path.join(path, "defective", photo_list[i]), img, check_contrast=False)
        io.imsave(os.path.join(path, "used_mask", photo_list[i]), mask, check_contrast=False)



def main():
    generateMasks("Generate/mask", 20)
    generateDefectivePhotos("Generate")

if __name__ == "__main__":
    main()