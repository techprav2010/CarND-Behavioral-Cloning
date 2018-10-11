
import os
import sys
import matplotlib.image as mpimg
import cv2
import csv
import numpy as np
from keras.models import Sequential
from keras.layers import Flatten,Dense,Lambda,Convolution2D,MaxPooling2D,Cropping2D
import matplotlib.pyplot as plt

import bc_const

class BcHelper:
    # def __init__(self):
    #     print("Find all subfolder  with training data in :", bc_const.RECORDING_ROOT_DIR)


    def find_recorded_subfolders(self, rootdir='', recorded_folders = []):
        for root, subFolders, files in os.walk(rootdir):
            if "driving_log.csv" in files:
                recorded_folders.append(root)
            else:
                for folder in subFolders:
                    self.find_recorded_subfolders(folder, recorded_folders)
        return recorded_folders

    def draw_data(y):
        plt.hist(y, 29, facecolor='g', alpha=0.75)
        plt.title('Steering angles')
        plt.grid(True)
        plt.show()

if __name__ == '__main__':
    helper= BcHelper()
    folders = helper.find_recorded_subfolders(bc_const.RECORDING_ROOT_DIR)


    print("folders = ", folders)