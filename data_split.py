import numpy as np
import glob
import openslide
import matplotlib.pyplot as plt
import random

import histomicstk as htk

import xml.etree.ElementTree as ET
import cv2
import skimage.measure
from skimage.transform import resize
from scipy import misc

from skimage import io
import os


img_size = (920, 1400)


def normalize_img(img):

    v = img.reshape(img.shape[0] * img.shape[1], )
    v = np.sort(v)

    min_val = v[int(v.shape[0] * 0.05)]
    max_val = v[int(v.shape[0] * 0.95)]
    img = np.clip(img, min_val, max_val)

    img = (img - min_val) * (1 / (max_val - min_val))

    return img



def convert_to_dab(img):
    # import pdb; pdb.set_trace()
    stainColorMap = {
        'hematoxylin': [0.65, 0.70, 0.29],
        'eosin': [0.07, 0.99, 0.11],
        'dab': [0.27, 0.57, 0.78],
        'null': [0.0, 0.0, 0.0]
    }

    stain_1 = 'hematoxylin'  # nuclei stain
    stain_2 = 'eosin'  # cytoplasm stain
    stain_3 = 'null'

    weights = np.array([stainColorMap[stain_1],
              stainColorMap[stain_2],
              stainColorMap[stain_3]]).T


    img = htk.preprocessing.color_deconvolution.color_deconvolution(img, weights).Stains
    img = img[:, :, 0]
    con = np.concatenate([normalize_img(img) * 255, img], axis = 1)
    plt.imshow(con)

    return img



if __name__ == '__main__':
    count = 0
    for ind in range(2, 5):


        source_image_path = 'qupath_like_exmaples/q00%d_extracted_image.png' % ind

        img = misc.imread(source_image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img = convert_to_dab(img)

        for i in range(8):
            for j in range(12):
                count += 1
                x = i * 128
                y = j * 128

                if (x + 128 > img_size[0]):
                    x = img_size[0] - 128

                if (y + 128 > img_size[1]):
                    y = img_size[1] - 128

                block = img[x:x+128, y:y+128]

                block = cv2.resize(block, (256, 256))

                cv2.imwrite('compare_data/%d.png' % count, block)



