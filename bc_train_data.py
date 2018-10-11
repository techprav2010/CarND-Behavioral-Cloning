import os
import os.path
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from random import randint

#project
from bc_helper import BcHelper
import bc_const
from bc_processs_image import BcProcesssImage


class BcTrainData:

    def __init__(self, recording_dir = bc_const.RECORDING_ROOT_DIR , recording_moved_from_dir = bc_const.ORIGINAL_RECORDING_ROOT_DIR):
        self.bc_helper = BcHelper()
        self.bc_processs_image = BcProcesssImage()
        bc_const.print_config()
        self.recording_dirs = self._get_sample_dirs(recording_dir, recording_moved_from_dir)

    def _get_sample_dirs(self, recording_root_dir , original_recording_root_dir):
        self.recording_root_dir = recording_root_dir
        self.original_recording_root_dir = original_recording_root_dir
        self.recording_dirs = self.bc_helper.find_recorded_subfolders(recording_root_dir)
        # for recording_dir in self.recording_dirs:
        #     print("Found subfolder with training data in  "+ recording_dir)
        return self.recording_dirs

    ########### load train data ######
    def load_csv_list(self, debug=False):
        print("==================== recoding summary ================================ start")
        cols_dtypes ={"c_camera_img": str , "l_camera_img": str,  "r_camera_img": str,  "steering_angle": np.float64, "throttle": np.float64,  "break": np.float64, "speed": np.float64}
        cols =[*cols_dtypes] #['c_camera_img', 'l_camera_img', 'r_camera_img', 'steering_angle', 'throttle', 'break', 'speed']
        dfs_array = []
        for recording_dir in self.recording_dirs:
            csv_file = recording_dir + '/driving_log.csv'
            if  os.path.isfile(csv_file) :
                # with open(csv_file) as csvfile:
                # dfs_array += pd.read_csv(csvfile, sep=",", header=None, names=cols, decimal=".", skip_blank_lines=True,  dtype=cols_dtypes)
                df_loop = pd.read_csv(csv_file, sep=",", header=None, names=cols, decimal=".", skip_blank_lines=True,  dtype=cols_dtypes) # pd.read_csv(csv_file, sep=",").dropna()
                df_loop = df_loop.dropna().sample(frac=1).reset_index(drop=True)
                dfs_array.append(df_loop)
                print("== records ==",  df_loop.shape, " in ", csv_file)
                # print(df_loop.head(n=2))
                # print("== read_csv ================================ end")
                print("------------------------------------------")
        if len(dfs_array) ==0:
            print("No recording folders found inside the root folder", self.recording_root_dir ,":",self.recording_dirs)
        # print("dfs_array=",len(dfs_array), self.recording_dirs)
        df =  pd.concat(dfs_array)
        # for index , df_elm in enumerate(dfs_array):
        #     print("df recording ", index, df_elm.shape)

        # print("*** head: original image path =",df.head())
        #replace directory name in image path if were recorded in some other directory
        if not self.original_recording_root_dir is None:
            df=  df.replace([self.recording_root_dir],[self.original_recording_root_dir])
            print("*** after image path changeh=", self.original_recording_root_dir,df.head())

        df = df.sample(frac=1).reset_index(drop=True)
        df = df.sample(frac=1).reset_index(drop=True)
        print("*** head: after reindex and shuffle=", df.head())
        self.df = df
        print("==================== recodings summary ================================ end")
        print()
        print()
        return df
    ########### images for visualization ######
    def visualization_imgs(self):
        try :
            idx = randint(5, 2000)
            record =  self.train_samples.iloc[idx]
        except :
            idx = randint(5, 2000)
            record = self.train_samples.iloc[idx]
        images, data = self.bc_processs_image.visualization_imgs(idx, record)
        return images, data
    ########### util methods ######
    def train_steps_per_epoch(self, batch_size=bc_const.BATCH_SIZE):
        return len(self.train_samples)/batch_size
    def validation_steps(self, batch_size=bc_const.BATCH_SIZE):
        return len(self.validation_samples)/batch_size

    ########### create_generators ######
    def create_generators(self , batch_size=bc_const.BATCH_SIZE, test_size=bc_const.TRAIN_TEST_SPLIT_SIZE ):
        self.train_samples, self.validation_samples = train_test_split(self.df, test_size=test_size)
        # print("==================== create_generators summary ================================ start")
        print("self.train_samples=", self.train_samples.shape)
        print("self.validation_samples=", self.validation_samples.shape)
        self.train_generator = self._generator(self.train_samples, batch_size=batch_size)
        self.validation_generator = self._generator(self.validation_samples, batch_size=batch_size)
        # print("==================== create_generators summary ================================ end")
        return self.train_generator, self.validation_generator
    def _generator(self, df_gen, batch_size=bc_const.BATCH_SIZE, do_shuffle=True):
        df_gen = df_gen.sample(frac=1).reset_index(drop=True)
        #we have 3 images from cameras + one flip
        #we will than to normalize, and augment
        print("_generator shape:", df_gen.shape)
        total_rows= df_gen.shape[0]
        batch_number = 1
        row_index = 0
        X, y, counter = [], [], 0
        while 1:
            # since we are going to create many samples out of one sample
            # loop until you reach end of the df_gen
            # print("inside " , df_gen.ix[row_index])
            images, angles = self.bc_processs_image.process_img( row_index, df_gen.ix[row_index])
            for img, angle in zip(images, angles):
                if counter == batch_size:
                    yield np.array(X), np.array(y)
                    X, y, counter = [], [], 0
                X.append(img)
                y.append(angle)
                counter += 1

            #loop start all over if we come to end
            row_index += 1
            if row_index >= total_rows:
                row_index = 0
                if do_shuffle:
                    df_gen = df_gen.sample(frac=1).reset_index(drop=True)

