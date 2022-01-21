import os
import sys

import numpy as np
import cv2

IMAGE_THRESHOLD = 178
MATCH_THRESHOLD = 0.30

TEMPLATE_FILENAME = './puzzle.png'

def grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def threshold(image):
    return cv2.threshold(image, IMAGE_THRESHOLD, 255, cv2.THRESH_BINARY)[1]

def find_template(image, template) -> dict:
    """Returns dict:
    {
        'conf': float,
        'left': int,
        'top': int,
        'right': int,
        'bottom': int,
    }

    If no template found returns None.
    """
    h, w = template.shape[:2]

    res = cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)
    cv2.imshow('Matching Probability', res)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

    if max_val < MATCH_THRESHOLD:
        return None

    return {
        'conf': max_val,
        'left': max_loc[0],
        'top': max_loc[1],
        'right': max_loc[0] + w,
        'bottom': max_loc[1] + h,
    }

def main():
    if (len(sys.argv) < 2) or (not os.path.exists(sys.argv[1])):
        print('Необходимо указать файл изображения в командной строке.')
        return
    # image = cv2.imread('./canvas/canvas1.png')
    image = cv2.imread(sys.argv[1])
    image_thresh = threshold(grayscale(image))
    cv2.imshow('Threshold Filter', image_thresh)
    template = grayscale(cv2.imread(TEMPLATE_FILENAME))

    result = find_template(image_thresh, template)
    if result:
        print('Результат поиска пазла:')
        print(f"- координаты: ({result['left']}, {result['top']})"
              f" - ({result['right']}, {result['bottom']})")
        print(f"- точность обнаружения: {round(result['conf'] * 100)}%")

        cv2.rectangle(image, (result['left'], result['top']),
                      (result['right'], result['bottom']), (0, 255, 0), 2)
        cv2.imshow('Puzzle Template Matching', image)
    else:
        print('Не удалось найти пазл.')

    cv2.waitKey()

if __name__ == '__main__':
    main()
