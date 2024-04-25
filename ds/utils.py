import albumentations as A
import cv2
import pickle
import numpy as np

def imgaug():
    return [
        A.OneOf([
            A.GaussNoise(var_limit=(5.0, 10.0)),
            A.MultiplicativeNoise(),
            A.RandomRain(),
        ], p=0.2),
        A.OneOf([
            A.MotionBlur(blur_limit=3, p=0.3),
            A.MedianBlur(blur_limit=3, p=0.3),
            A.Blur(blur_limit=3, p=0.3),
        ], p=0.2),
        A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.12, rotate_limit=15, p=0.5,
                          border_mode = cv2.BORDER_CONSTANT),
        A.OneOf([
            A.OpticalDistortion(p=0.3),
            A.GridDistortion(p=0.3),
            A.PiecewiseAffine(p=0.3),
        ], p=0.2),
        A.OneOf([
            A.CLAHE(clip_limit=2),
            A.Sharpen(),
            A.Emboss(),
            A.RandomBrightnessContrast(),   
            A.Downscale(interpolation = {
                "downscale": cv2.INTER_NEAREST,
                "upscale": cv2.INTER_NEAREST
            }),
        ], p=0.3),
        A.OneOf([
            A.HueSaturationValue(p=0.3),
            A.ColorJitter(p=0.3),
        ], p= 0.3),
        A.RGBShift(p=0.3),
        A.RandomShadow(p=0.2)
    ]

def read_pickle(path):
    with open(path, 'rb') as  file:
        dct = pickle.load(file=file)
    file.close()
    return dct

def save_pickle(dct, path):
    with open(path, 'wb') as  file:
        pickle.dump(obj=dct, file=file, protocol=pickle.HIGHEST_PROTOCOL)

def draw_point(img, data):

    center = (int(data[1]), int(data[2]))
    wh = (int(data[3]), int(data[4]))
    angle = data[-1]

    img = cv2.circle(img, center, 10, (255, 0, 0), -1)
    
    box = cv2.boxPoints((center, wh, angle))
    box = np.array(box).astype(np.uint)
    img = cv2.drawContours(img,[box],0,(0,255,0),2)

    return img