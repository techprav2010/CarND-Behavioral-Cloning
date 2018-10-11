 ## Behavioral Cloning Project  
 
 [//]: # (Image References) 

[nvidia_net]: ./out_images/nvidia_net.png
[histogram]: ./out_images/histogram.png
[model_nvidia]: ./out_images/model_nvidia.png
[model_lenet]: ./out_images/model_lenet.png
[model_single_layer]: ./out_images/model_single_layer.png
[camera_images_and_augmentation]: ./out_images/camera_images_and_augmentation.png
 

### Goal statement
In this project train a CNN model to clone driving behavior using camera images. Use Udacity provided simulator.

### To meet specifications, the project will require submitting five files:

- model.py (script used to create and train the model)
- drive.py (script to drive the car - feel free to modify this file)
- model_*.h5 (a trained Keras model)
- a report writeup file (either markdown or pdf)
- video.mp4 (a video recording of your vehicle driving autonomously around the track for at least one full lap)
 

## Rubric Points
### [rubric points](https://review.udacity.com/#!/rubrics/1968/view) 

---
### Are all required files submitted?

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

#### Following files are part of submission:
* Classes and files
    * bc_const = holds important constants 
        * RECORDING_ROOT_DIR = "/home/lab/sim_data/my_capture"
        * BATCH_SIZE = 512
        * EPOCHS = 20
    * BcProcesssImage = is used to load, process images.
    * BcTrainData = is responsible read traing data and  create 'generators'
    * BcModel = Is the glue class. Does the training using various CNN keras models. 
    * model.h5 = saved model
    * video.mp4 = recorded video using model_nvidia.h5 in autonomous mode 
    * CarND-Behavioral-Cloning-Visual.ipynb + html = train and visualization notebook.
     
 
#### 2. Submission includes functional code
Tested the model with Udacity simulator. and recored the video
```python drive.py model.h5
   # capture images autonomous mode
   python drive.py model.h5 ~/sim_data/video
   # create video from  captured image
   python video.py ~/sim_data/video
```


#### 3. Submission code is usable and readable

The working [CarND-Behavioral-Cloning-Visual.ipynb](CarND-Behavioral-Cloning-Visual.ipynb)  notebook, can be used to test the code end to end.\
- Change bc_const='TO_YOUR_RECORDING_ROOT_FOLDER' appropriately. And any other settings in bc_const.
- Step by step this notebook evolves into 'Behavioral Cloning' model for autonomous car model.h5.
 
 

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

The model I settled is similar to  nVidia's  [https://devblogs.nvidia.com/deep-learning-self-driving-cars/](https://devblogs.nvidia.com/deep-learning-self-driving-cars/)
 
![][nvidia_net]
#### 2. Attempts to reduce overfitting in the model

* I have collected enough training data to make model robust. e.g driving laps, clock_wise, anti_clock_wise driving. 
* Added dropouts after layers flatten_1, and next two fully_connected layers.
* Normalizing images is the most important for neural network to train better.
* Cropped un-useful area from the images, reduce the noise.
* All images has been mirrored to generalize the steering angle.

 

#### 3. Model parameter tuning

* Uses adam optimizer. 
    * model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
* TRAIN_TEST_SPLIT_SIZE = 0.2
* BATCH_SIZE = 512
* EPOCHS = 20

#### 4. Appropriate training data
* I learned how the model is behaving after every training, I created new training data to test if the car stops gogin off-track

 
 ![][camera_images_and_augmentation]

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The main goal is to avoid car driving of the road. It was essential to teach network how to drive in the center of the road. 
Also it has to learn how to drive back towards center of the road when it start approaching the side lines.

Started the code with smaller model to see how it behaves in autonomous mode e.g. single layer, network similar to lenet. 
Have to use more complex network with dropouts to avoid under-fitting.
Finally the  model architecture close to nvidia level of complexcity has successfully dive car without going off-road.

* models
    * model_single_layer
    
         ![][model_single_layer] 
         
    * model_lenet
    
        ![][model_lenet] 
        

#### 2. Final Model Architecture

The final model:
Layer (type)                 Output Shape              Param #   
=================================================================
lambda_1 (Lambda)            (None, 160, 320, 3)       0         
_________________________________________________________________
cropping2d_1 (Cropping2D)    (None, 64, 200, 3)        0         
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 30, 98, 24)        1824      
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 13, 47, 36)        21636     
_________________________________________________________________
conv2d_3 (Conv2D)            (None, 5, 22, 48)         43248     
_________________________________________________________________
conv2d_4 (Conv2D)            (None, 3, 20, 64)         27712     
_________________________________________________________________
conv2d_5 (Conv2D)            (None, 1, 18, 64)         36928     
_________________________________________________________________
flatten_1 (Flatten)          (None, 1152)              0         
_________________________________________________________________
dropout_1 (Dropout)          (None, 1152)              0         
_________________________________________________________________
dense_1 (Dense)              (None, 100)               115300    
_________________________________________________________________
activation_1 (Activation)    (None, 100)               0         
_________________________________________________________________
dropout_2 (Dropout)          (None, 100)               0         
_________________________________________________________________
dense_2 (Dense)              (None, 50)                5050      
_________________________________________________________________
activation_2 (Activation)    (None, 50)                0         
_________________________________________________________________
dropout_3 (Dropout)          (None, 50)                0         
_________________________________________________________________
dense_3 (Dense)              (None, 10)                510       
_________________________________________________________________
dense_4 (Dense)              (None, 1)                 11        
=================================================================
Total params: 252,219
Trainable params: 252,219
Non-trainable params: 0 

Here is a visualization of the architecture:
![][model_nvidia] 

#### 3. Creation of the Training Set & Training Process

* Neural network need to see a situation to learn / memorise to drive safely in that particular situation. 
Hence I keep adding more and more training data to counter the car going off-road in particular driving location
e.g on bridge or on turns etc
    * (4180, 7)  in  /home/lab/sim_data/my_capture/center_anti_clock_wise/driving_log.csv
    * (5760, 7)  in  /home/lab/sim_data/my_capture/center_clock_wise/driving_log.csv
    * (3511, 7)  in  /home/lab/sim_data/my_capture/recovery_laps2/driving_log.csv
    * (1863, 7)  in  /home/lab/sim_data/my_capture/bridge/driving_log.csv
    * (525, 7)   in  /home/lab/sim_data/my_capture/recovery_laps3/driving_log.csv
    * (2163, 7)  in  /home/lab/sim_data/my_capture/recovery_laps4/driving_log.csv
    * (2329, 7)  in  /home/lab/sim_data/my_capture/clock_wise/driving_log.csv 
* Also recorded two track and left track and right track. 
  The did the training a separate model with thse training data.

