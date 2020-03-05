import cv2
from scipy import misc
from copy import deepcopy


img_size = (920, 1400)


if __name__ == '__main__':
    count = 0
    for ind in range(2, 5):
        source_image_path = 'qupath_like_exmaples/q00%d_extracted_image.png' % ind

        source_img = misc.imread(source_image_path)
        source_img = cv2.cvtColor(source_img, cv2.COLOR_BGR2RGB)

        img = deepcopy(source_img)

        for i in range(8):
            for j in range(12):
                count += 1

                msk = misc.imread('images/%d_fake_B.png' % count)
                msk = cv2.resize(msk, (128, 128))

                x = i * 128
                y = j * 128

                if (x + 128 > img_size[0]):
                    x = img_size[0] - 128

                if (y + 128 > img_size[1]):
                    y = img_size[1] - 128

                img[x:x+128, y : y + 128] = msk


        source_img[img > 0] = 255

        cv2.imwrite('msk%d.png' % ind, img)

