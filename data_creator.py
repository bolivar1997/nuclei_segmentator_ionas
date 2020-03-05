import numpy as np
import histomicstk as htk
import xml.etree.ElementTree as ET
import cv2
import os


train_image_shape = (1000, 1000)

def parse_xml(image_path):
    msk = np.zeros(train_image_shape, dtype=np.uint8)

    tree = ET.parse(image_path)
    root = tree.getroot()

    for annot in root.iter('Vertices'):
        pts = []

        for coord in annot.iter('Vertex'):
            x = coord.get('X')
            y = coord.get('Y')

            x = int(x.split('.')[0])
            y = int(y.split('.')[0])

            pts.append([x, y])

        pts = np.array(pts, np.int32)
        pts = pts.reshape((1, -1, 2))

        cv2.fillPoly(msk, pts, color=1)

    return msk


def rotate(image, angle):
    center = (image.shape[0] / 2, image.shape[1] / 2)
    m = cv2.getRotationMatrix2D(center, angle, 1)
    rotated = cv2.warpAffine(image, m, (image.shape[0], image.shape[1]))
    return rotated


def normalize_img(img):

    v = img.reshape(img.shape[0] * img.shape[1], )
    v = np.sort(v)

    min_val = v[int(v.shape[0] * 0.05)]
    max_val = v[int(v.shape[0] * 0.95)]
    img = np.clip(img, min_val, max_val)

    img = (img - min_val) * (1 / (max_val - min_val))

    return img


def gen_data():
    stain_color_map = {
        'hematoxylin': [0.65, 0.70, 0.29],
        'eosin': [0.07, 0.99, 0.11],
        'null': [0.0, 0.0, 0.0]
    }

    stain_1 = 'hematoxylin'  # nuclei stain
    stain_2 = 'eosin'  # cytoplasm stain
    stain_3 = 'null'

    weights = np.array([stain_color_map[stain_1],
                        stain_color_map[stain_2],
                        stain_color_map[stain_3]]).T

    count = 0

    for path_img, path_annot in zip(sorted(os.listdir('./data/Tissue Images')),
                                    sorted(os.listdir('./data/Annotations/'))):

        msk = parse_xml('./data/Annotations/' + path_annot)

        img = cv2.imread('./data/Tissue Images/' + path_img)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img = htk.preprocessing.color_deconvolution.color_deconvolution(img, weights).Stains
        img = img[:, :, 0]

        for i in range(0, 4):
            for j in range(0, 4):

                ox = img[i * 250: i * 250 + 250, j * 250: j * 250 + 250]
                oy = msk[i * 250: i * 250 + 250, j * 250: j * 250 + 250]

                for degree in [0, 90, 180, 270]:
                    new_ox = rotate(ox, degree)
                    new_oy = rotate(oy, degree)

                    for flip_axis in range(0, 2):
                        flipped_ox = cv2.flip(new_ox, flip_axis)
                        flipped_oy = cv2.flip(new_oy, flip_axis)

                        flipped_ox = cv2.resize(flipped_ox, (256, 256), interpolation=cv2.INTER_CUBIC)
                        flipped_oy = cv2.resize(flipped_oy, (256, 256), interpolation=cv2.INTER_CUBIC)

                        flipped_ox = normalize_img(flipped_ox)

                        res = np.concatenate([(flipped_ox * 255), flipped_oy * 255], axis=1)
                        count += 1

                        if count < 3000:
                            cv2.imwrite('data/train/%d.png' % count, res)
                        else:
                            cv2.imwrite('data/validate/%d.png' % count, flipped_ox * 255)
                            cv2.imwrite('data/validation_masks/%d.png' % count, 255 * flipped_oy)


if __name__ == '__main__':
    gen_data()
