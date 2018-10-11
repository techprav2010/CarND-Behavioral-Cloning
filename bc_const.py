
#load train data
# RECORDING_ROOT_DIR = "/home/lab/sim_data"
RECORDING_ROOT_DIR = "/home/lab/sim_data/my_capture"
ORIGINAL_RECORDING_ROOT_DIR = None

#augment
ENABLE_AUGMENTATION = False
ENABLE_MULTIPLE_CAMERAS = False
CORRECTION_FACTOR = 0.2

#fit
TRAIN_TEST_SPLIT_SIZE = 0.2
BATCH_SIZE = 512
EPOCHS = 20

# ['c_camera_img', 'l_camera_img', 'r_camera_img', 'steering_angle', 'throttle', 'break', 'speed']
COL_IMG_CENTER='c_camera_img'
COL_IMG_LEFT='l_camera_img'
COL_IMG_RIGHT='r_camera_img'
COL_ANGLE='steering_angle'
COL_THROTTLE='throttle'
COL_BREAK='break'
COL_SPEED='speed'

def print_config():
    print("==================== config summary ================================ start")
    print("recording_root_dir:", RECORDING_ROOT_DIR)
    print("original_recording_root_dir:", ORIGINAL_RECORDING_ROOT_DIR)
    print("enable augmentation:", ENABLE_AUGMENTATION)
    print("enable multiple cameras:", ENABLE_MULTIPLE_CAMERAS)
    print("correction_factor:", CORRECTION_FACTOR)
    print("train_test_split_size:", TRAIN_TEST_SPLIT_SIZE)
    print("batch_size:", BATCH_SIZE)
    print("epochs:", EPOCHS)
    print("Find all subfolder  with training data in :", RECORDING_ROOT_DIR)
    print("==================== config summary ================================ end")