from os import path

import numpy as np
import SimpleITK as sitk

from helpers import *

#############################
##         General         ##
#############################
PROJECT_NAME = "stroke_cta"
IMAGE_DIR = "/data/path_to_images/stroke-auto/data/"
PROJECT_DIR = f"/scratch/path_to_project_root/{PROJECT_NAME}/"

#############################
##      Deep learning      ##
#############################
DL_SPACING = [1., 1., 1.]       # Voxel spacing used for deep learning
WINDOW_SIZE = (256, 256, 64)    # Crop size in voxels
BATCH_SIZE = 5                  # Training batch size
LEARNING_RATE = 1e-4            # Learning rate
VALID_NUM_CROPS = 50            # Number of crops used for validation during training
REPLACEMENT_VALUE = -1000       # Value used to mask unwanted organs
MAX_EPOCHS = 5000                # Maximum number of training epochs
EARLY_STOPPING = 50             # Automatically stop training after X epochs
                                # (None to disable)
MODEL_SEL_VAR = "val_loss"      # Variable to track for determining best model
MODEL_SEL_MODE = "min"          # Whether to minimize or maximize the score

# Calculate the field of view of a single crop in millimeters
FIELD_OF_VIEW = [sp * sz for sp, sz in zip(DL_SPACING, WINDOW_SIZE)]

############################
##         Classes        ##
############################
CLASSES = ["background", "external_artery", "brain"]
NUM_CLASSES = len(CLASSES)
#############################
##     Synchronization     ##
#############################

DBX_TOKEN = "s1.YOUR_DROPBOX_TOKEN"
DBX_PUSH_PATH = "/stroke_cta/"
DBX_GENERATED_PUSH_PATH = path.join(DBX_PUSH_PATH, 'generated.zip')
DBX_PULL_LINK = r"YOUR_DROPBOX_DOWNLOAD_LINK?dl=1"

#############################
## Preprocessing function ##
#############################
def preprocess(img, seg = None, 
    dtype="sitk", 
    resample_spacing=None,
    resample_min_shape=None,
    normalization="znorm"):

    if resample_spacing is not None:
        img = resample(img, 
            min_shape=resample_min_shape, 
            method=sitk.sitkLinear,
            new_spacing=resample_spacing)
        if seg is not None:
            seg = resample_to_reference(seg, img)
    img_n = sitk.GetArrayFromImage(img).T
    
    if seg is not None:
        seg_n = sitk.GetArrayFromImage(seg).T

    if normalization == "znorm":
        img_n = img_n - np.mean(img_n)
        img_n = img_n / np.std(img_n)
    else:
        img_n = img_n / 1000. + 1.

    if dtype == "numpy":
        if seg is not None:
            return img_n, seg_n
        return img_n, None

    # Restore SITK parameters
    img_s = sitk.GetImageFromArray(img_n.T)
    img_s.CopyInformation(img)

    if seg is not None:
        seg_s = sitk.GetImageFromArray(seg_n.T)
        seg_s.CopyInformation(seg)

        return img_s, seg_s
    return img_s, None

def get_mask(segmentation):
    """Function to create the mask used to blind the image, based on the 
    generated segmentation.
    """
    mask = (segmentation > 0) * 1.
    return mask
