import cv2
import config
import numpy as np
import os


def apply_mask(img, mask):
    res = cv2.bitwise_and(img, img, mask=mask)
    return res


def hair_removal(img):
    grayScale = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)   # chuyen ve anh xam
    kernel = cv2.getStructuringElement(1, (17, 17))     # khoi tao cua so truot
    # su khac nhau giu dilation+erosion voi anh goc
    blackhat = cv2.morphologyEx(grayScale, cv2.MORPH_BLACKHAT, kernel)
    ret, thresh2 = cv2.threshold(
        blackhat, 10, 255, cv2.THRESH_BINARY)   # mask de remove hair
    # ghep mask voi anh goc
    dst = cv2.inpaint(img, thresh2, 1, cv2.INPAINT_TELEA)
    return dst


def resize(img, size):
    return cv2.resize(img, (size, size))


def remove_ink(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_purple = np.array([40, 70, 70])
    upper_purple = np.array([180, 255, 255])
    mask = cv2.inRange(hsv, lower_purple, upper_purple)
    res = cv2.bitwise_and(img, img, mask=mask)
    ret, thresh2 = cv2.threshold(res, 100, 255, cv2.THRESH_BINARY)
    thresh2 = cv2.cvtColor(thresh2, cv2.COLOR_BGR2GRAY)
    dst = cv2.bitwise_not(img, img, mask=thresh2)
    return dst


def ROI(img):
    gr = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mean, std = cv2.meanStdDev(gr)
    ret, thresh2 = cv2.threshold(gr, int(mean-std), 255, cv2.THRESH_BINARY_INV)
    return thresh2


def shade_removal(img):
    dilated_img = cv2.dilate(img, np.ones((7, 7), np.uint8))
    bg_img = cv2.medianBlur(dilated_img, 21)
    diff_img = 255 - cv2.absdiff(img, bg_img)
    norm_img = diff_img.copy()
    cv2.normalize(diff_img, norm_img, alpha=0, beta=255,
                  norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
    _, thr_img = cv2.threshold(norm_img, 230, 0, cv2.THRESH_TRUNC)
    cv2.normalize(thr_img, thr_img, alpha=0, beta=255,
                  norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
    return thr_img


def preprocess(img):
    img = hair_removal(img)
    # img = remove_ink(img)
    # mask = ROI(img)
    # res = apply_mask(img, mask)
    return img


if __name__ == '__main__':
    input_folder = 'data/train'
    output_folder = 'data/train_preprocessed'
    img_size = config.img_size
    for image_file in os.listdir(input_folder):
        output_image_file = ''
        try:
            input_image_file = os.path.join(input_folder, image_file)
            input_image = cv2.imread(input_image_file)
            resize_image = resize(input_image, img_size)
            preprocessed_image = preprocess(resize_image)
            output_image_file = os.path.join(output_folder, image_file)
            cv2.imwrite(output_image_file, preprocessed_image)
        except:
            print(f"Cannot write {output_image_file}")

