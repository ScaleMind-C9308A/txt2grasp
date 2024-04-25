import albumentations as A
import cv2

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