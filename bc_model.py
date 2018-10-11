
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras import optimizers
from keras.layers import Flatten, Dense, Lambda, Activation, Dropout
from keras.layers import Conv2D, MaxPooling2D, Cropping2D, Reshape
from keras.callbacks import History

#project
from bc_helper import BcHelper
import bc_const
from bc_processs_image import BcProcesssImage
from bc_train_data import BcTrainData

class BcModel:

    def __init__(self):
        self.bc_train_data = BcTrainData()

        self.bc_train_data.load_csv_list()
        print("================== generator   =================================== start ")
        print("recording total = ", self.bc_train_data.df.shape)
        self.train_generator, self.validation_generator = self.bc_train_data.create_generators()
        # samples_batch = bc_train_data.generator(bc_train_data.train_samples)
        # train_generator = bc_train_data.train_generator
        # validation_generator = bc_train_data.validation_generator\
        print(self.data_info())
        print("================== generator  =================================== end ")


    def data_info(self):
        X_train_batch, y_train_batch = next(self.train_generator)
        print(X_train_batch.shape, y_train_batch.shape)

        X_valid_batch, y_valid_batch = next(self.validation_generator)
        print(X_valid_batch.shape, y_valid_batch.shape)

    def histogram_angle(self):
        plt.hist(self.bc_train_data.train_samples[bc_const.COL_ANGLE], 29, facecolor='g', alpha=0.75)
        plt.title('Histogram of steering angles')
        plt.grid(True)
        plt.savefig('./out_images/histogram.png')
        plt.show()

    def camera_images(self):
        X_train_batch, y_train_batch = next(self.train_generator)
        print(X_train_batch.shape, y_train_batch.shape)

        X_valid_batch, y_valid_batch = next(self.validation_generator)
        print(X_valid_batch.shape, y_valid_batch.shape)

    def visualization_imgs(self):
        return  self.bc_train_data.visualization_imgs()
    #################  training and ploting ############################

    def train(self, model, model_name="model"):
        #set optimizer
        model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
        #train the given model
        self.history = model.fit_generator(
              self.train_generator
            , self.bc_train_data.train_steps_per_epoch()
            , validation_data=self.validation_generator
            , validation_steps=self.bc_train_data.validation_steps()
            , epochs=bc_const.EPOCHS)
        # save the trained model
        model_save = model_name+'.h5'
        model.save(model_save)
        print(model_save + ' trained')

    def plot_history(self, name="nvida"):
        print()
        print(self.history.history.keys())
        plt.plot(self.history.history['loss'])
        plt.plot(self.history.history['val_loss'])
        plt.title('Loss')
        plt.ylabel('MSE Loss')
        plt.xlabel('epoch')
        plt.legend(['training set', 'validation set'], loc='upper right')
        plt.savefig('./out_images/plot_history'+name+'.png')
        plt.show()
    #################  training and ploting ############################

    #################  keras models ############################

    def model_nvidia(self):
        # 5 Conv2D layers
        # 3 Conv2D layers kernel= 5x5
        # 2 Conv2D layers kernel= 5x5
        # flatten
        # 3 fully connected layers

        #nvidia style
        model = Sequential()
        model.add(Lambda(lambda x: x / 127.5 - 1.0, input_shape=(160, 320, 3)))
        model.add(Cropping2D(cropping=((70, 26), (60, 60))))

        # 5 Conv2D
        model.add(Conv2D(24, (5, 5), strides=(2, 2), activation="relu"))
        model.add(Conv2D(36, (5, 5), strides=(2, 2), activation="relu"))
        model.add(Conv2D(48, (5, 5), strides=(2, 2), activation="relu"))
        model.add(Conv2D(64, (3, 3), activation="relu"))
        model.add(Conv2D(64, (3, 3), activation="relu"))

        # flatten
        model.add(Flatten())
        model.add(Dropout(0.25))

        # fully connected layers
        model.add(Dense(100))
        model.add(Activation('relu'))
        model.add(Dropout(0.25))


        model.add(Dense(50))
        model.add(Activation('relu'))
        model.add(Dropout(0.25))

        model.add(Dense(10))
        model.add(Dense(1))

        # model.summary()
        return model

    def model_lenet(self):
        model = Sequential()
        model.add(Lambda(lambda x: x / 127.5 - 1.0, input_shape=(160, 320, 3)))
        model.add(Cropping2D(cropping=((70, 26), (60, 60))))

        model.add(Conv2D(6, (5, 5), strides=(2, 2), activation="relu"))
        model.add(MaxPooling2D((2, 2)))

        model.add(Conv2D(16,(5, 5), strides=(2, 2), activation="relu"))
        model.add(MaxPooling2D((2, 2)))

        model.add(Flatten())
        model.add(Dense(120))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))

        model.add(Dense(84))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))

        model.add(Dense(1))

        # model.summary()
        return model

    def model_single_layer(self):

        model = Sequential()
        model.add(Lambda(lambda x: x / 127.5 - 1.0, input_shape=(160, 320, 3)))
        model.add(Cropping2D(cropping=((70, 26), (60, 60))))

        model.add(Conv2D(32, (3, 3), strides=(2, 2), activation="relu"))
        model.add(MaxPooling2D((2, 2)))

        model.add(Dropout(0.5))
        model.add(Flatten())
        model.add(Dense(1))

        # model.summary()
        return model

     #################  keras models ############################



# hist = plt.hist(steerings, bins=100)
if __name__ == '__main__':
    bc_model= BcModel()
    bc_model.histogram_angle()
    images, data = bc_model.visualization_imgs()
    bc_model.model_lenet().summary()
    bc_model.model_single_layer().summary()
    bc_model.model_nvidia().summary()
    bc_model.train(bc_model.model_lenet(), 'model_lenet')
    bc_model.plot_history(name='model_lenet')
    bc_model.train(bc_model.model_single_layer(), 'model_single_layer')
    bc_model.plot_history(name='model_single_layer')
    # bc_model.train(bc_model.model_nvidia(), 'model_nvidia')
    # bc_model.plot_history(name='model_nvidia')