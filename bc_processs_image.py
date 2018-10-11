import cv2
import random
import numpy as np
from random import shuffle

import bc_const

class BcProcesssImage():

    def __init__(self):
        self.counter=0
        total = getattr(self, "total", 0)
        if total ==0 :
            self.total = 0
        # print("BcProcesssImage " + total)

    def process_img(self, idx, record):

        X, y  = [],  []
        images, angles = [], []

        #existing: from 3 cameras
        images, angles = self.load_orig(record, images, angles)
        X += images
        y += angles

        #new images: augment_flip
        images, angles = self.augment_flip(record, images, angles)
        X += images
        y += angles


        #new images: augment_brightness
        if bc_const.ENABLE_AUGMENTATION:
            images, angles = self.augment_brightness(record, images, angles)
            X += images
            y += angles


        self.counter = len(X)
        self.total += self.counter

        # print("BcProcesssImage " , idx, " images=" ,   self.counter, " total=", self.total)
        return X, y

    def load_orig(self, record, images, angles ):

        angle = float(record[bc_const.COL_ANGLE])
        images.append(self._normalize(record[bc_const.COL_IMG_CENTER]))
        angles.append(angle)

        if bc_const.ENABLE_MULTIPLE_CAMERAS :
            if angle < - 0.5 and  angle >- 9.0:
                images.append(self._normalize(record[bc_const.COL_IMG_RIGHT]))
                angles.append(angle - bc_const.CORRECTION_FACTOR)

            if angle > 0.5 and  angle < 9.0:
                images.append(self._normalize(record[bc_const.COL_IMG_LEFT]))
                angles.append(angle + bc_const.CORRECTION_FACTOR)

        return  images, angles

    def _normalize(self, url):
        # print("_normalize ", url )
        img = cv2.imread(url)
        b,g,r = cv2.split(img)
        img = cv2.merge([r,g,b])
        #print("_normalize ", url, img.shape)
        return img

    def _augment_flip(self, img, angle):
        s_img = cv2.flip(img, 1)
        s_angle = - angle
        return s_img, s_angle

    def augment_flip(self, record, images, angles):
        f_images,f_angles=[],[]
        for img, angle in zip(images, angles):
            s_img , s_angle = self._augment_flip(img, angle)
            f_images.append(s_img)
            f_angles.append(s_angle)
        # flip
        images += f_images
        angles += f_angles
        return images, angles

    def augment_brightness(self, record, images, angles):
        b_images, b_angles = [], []
        for img, angle in zip(images, angles):
             hsv_img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
             h, s, v = cv2.split(hsv_img)
             v += np.clip(v + random.randint(-5, 15), 0, 255).astype('uint8')
             hsv_img = cv2.merge((h, s, v))
             s_img = cv2.cvtColor(hsv_img, cv2.COLOR_HSV2RGB)
             b_images.append(s_img)
             b_angles.append(angle)
        # brightness
        images += b_images
        angles += b_angles
        return images, angles

    ########### images for visualization ######
    def visualization_imgs(self, idx, record):
        images,data = {},{}
        data["idx"] = idx
        data["record"] = record
        images = self._visual_img(idx, record, bc_const.COL_IMG_CENTER, images)
        images = self._visual_img(idx, record, bc_const.COL_IMG_LEFT, images)
        images = self._visual_img(idx, record, bc_const.COL_IMG_RIGHT, images)
        return images, data
    def _visual_img(self,idx, record, key, images={}):
        file_path = record[key]
        angle = float(record[bc_const.COL_ANGLE])
        img =   cv2.imread(file_path)
        images[key] = img
        images[key + ".norm"] = self._normalize(file_path)
        f_image, _ = self._augment_flip(img, angle)
        images[key + ".flip"] = f_image
        return images