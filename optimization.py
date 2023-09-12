from glob import glob
from os import path
import os
from time import sleep
import numpy as np

from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.utils import get_custom_objects
from tensorflow.keras.models import load_model

from config import *
from dbx import download_and_extract
from helpers import categorical_focal_loss, dice_coef_multilabel, print_
from predict import predict

from train import train

_WINDOW_XY_LOW, _WINDOW_XY_HIGH, _WINDOW_XY_STEP = 160, 256, 32
_WINDOW_Z_LOW, _WINDOW_Z_HIGH, _WINDOW_Z_STEP = 16, 64, 16

_RESAMPLE_SPACINGS = [None, 0.5, 1.]
_NORMALIZATION = ['znorm', 'divide']
_BATCH_MIN, _BATCH_MAX = 1, 15
_LR_MIN, _LR_MAX = 1e-4, 1e-3
_L2_MIN, _L2_MAX = 1e-4, 1e-3
_OPTIMIZERS = 'adam', 'rmsprop'

_LOSSES = [
    # 'focal_loss_5',
    'dice_loss'
]

_UNET_TYPES = ['dual_attention', 'simple']

def get_optimizer(opt_str, learning_rate):
    if opt_str == "adam":
        return Adam(learning_rate)
    else: 
        return RMSprop(learning_rate)

def get_loss(loss_str, num_classes):
    if loss_str == "focal_loss_5":
        return categorical_focal_loss([0.25] + [5.] + [0.25])
    if loss_str == "dice_loss":
        return dice_coef_multilabel
    return loss_str

###############################################

fold_num = 0
num_folds = 7
random_id = np.random.randint(111111,999999)

optimization_dir = path.join(PROJECT_DIR, "optimization")
optuna_db = path.join(optimization_dir, "optuna.db")

# Fetch all available segmentations
print_("> Downloading available segmentation files..")
s_dir = path.join(optimization_dir, "segmentations/")
m_dir = path.join(optimization_dir, f"segmentations/manual/{random_id}")
os.makedirs(m_dir, exist_ok=True)

# Clean the directory to be sure
for filename in glob(path.join(m_dir, "*.nii.gz")):
    os.remove(filename)

download_and_extract(
    url=DBX_PULL_LINK, 
    target_dir=m_dir, 
    tmp_name=path.join(s_dir, f"manual_tmp_fold{fold_num}.zip"))
    
# Match images and segmentations
all_images = glob(path.join(IMAGE_DIR, "*.nii.gz"))
all_segmentations = glob(path.join(m_dir, "*.nii.gz"))

# Find all segmentations and images with matching segmentations
img_files, seg_files = [], []
for bn in sorted([path.basename(s) for s in all_segmentations]):
    img_path = path.join(IMAGE_DIR, bn)
    seg_path = path.join(m_dir, bn)

    if path.exists(img_path) and path.exists(seg_path):
        img_files.append(img_path)
        seg_files.append(seg_path)
        print_("> Found matching image and segmentation for:", bn)

test_imgs = [sitk.ReadImage(img_files.pop()) for _ in range(10)]
test_segs = [sitk.ReadImage(seg_files.pop(), sitk.sitkUInt8) for _ in range(10)]

def fake_loss(*args, **kwargs):
        return 0.

get_custom_objects().update({"loss": fake_loss,
    "categorical_focal_loss_fixed": fake_loss,
    "external_artery": fake_loss,
    "brain": fake_loss,
    "dice_coef_multilabel": fake_loss,
    "mean_dice": fake_loss})
    
###############################################

import optuna

def objective(trial):
    print_("Trial number:", trial.number)
    trial_dir = path.join(optimization_dir, f"trial_{trial.number}")
    trial_seg_dir = path.join(trial_dir, "segmentations")
    trial_generated_dir = path.join(trial_seg_dir, "generated")
    trial_masked_dir = path.join(trial_seg_dir, "masked")

    window_xy  = trial.suggest_int('window_xy', _WINDOW_XY_LOW, _WINDOW_XY_HIGH, _WINDOW_XY_STEP)
    window_z   = trial.suggest_int('window_z', _WINDOW_Z_LOW, _WINDOW_Z_HIGH, _WINDOW_Z_STEP)
    window_size = (window_xy, window_xy, window_z)

    batch_size =  trial.suggest_int('batch_size', _BATCH_MIN, _BATCH_MAX)
    learning_rate =  trial.suggest_uniform('learning_rate', _LR_MIN, _LR_MAX)
    unet_type =  trial.suggest_categorical('unet_type', _UNET_TYPES)
    regularization = trial.suggest_uniform('reg', _L2_MIN, _L2_MAX)
    inst_norm =  trial.suggest_categorical('inst_norm', [True, False])
    
    normalization = trial.suggest_categorical('normalization', _NORMALIZATION)
    loss_str =  trial.suggest_categorical('loss', _LOSSES)
    opt_str =  trial.suggest_categorical('optimizer', _OPTIMIZERS)

    rotation_freq = trial.suggest_uniform("rotation_freq", 0., 0.5)
    tilt_freq = trial.suggest_uniform("tilt_freq", 0.0, 0.5)
    noise_freq = trial.suggest_uniform("noise_freq", 0.0, 0.7)
    noise_mult = trial.suggest_uniform("noise_mult", 1e-4, 1e-2)

    resample_spacing =  trial.suggest_categorical('spacing', _RESAMPLE_SPACINGS)
    if resample_spacing is not None:
        resample_spacing = (resample_spacing, resample_spacing, 3)

    optimizer = get_optimizer(opt_str, learning_rate)
    loss = get_loss(loss_str, 3)
    
    print_(f"Window: ({window_xy}, {window_xy}, {window_z})")
    print_(f"Resample:", resample_spacing)
    print_(f"Learning rate:", round(learning_rate, 5))
    print_(f"Regularization:", round(regularization, 5))
    print_(f"Optimizer:", opt_str)
    print_(f"Loss:", loss_str)

    try:
        train(
            img_paths=img_files,
            seg_paths=seg_files,
            work_dir=trial_dir,
            fold_num=fold_num,
            num_folds=num_folds,
            resample_spacing = resample_spacing,
            window_size = window_size,
            batch_size = batch_size,
            num_validation = 50,
            max_epochs = 50,
            early_stopping = 50,
            early_stopping_var = "val_loss",
            early_stopping_mode = "min",

            conditional_stopping_epochs=5,
            conditional_stopping_mode='max',
            conditional_stopping_threshold=0.4,
            conditional_stopping_var='val_mean_dice',

            loss = loss,
            optimizer = optimizer,

            rotation_freq = rotation_freq,
            tilt_freq = tilt_freq,
            noise_freq = noise_freq, 
            noise_mult = noise_mult,

            unet_type = unet_type,
            l2_regularization = regularization,
            normalization = normalization,
            instance_norm=inst_norm,

            class_names = CLASSES,

            num_samples = 10
        )
    except Exception as err:
        print_("Incompatible configuration.")
        print_(err)
        return 2.
    try:
        print_("Reading best model")
        models = [load_model(m) for m in glob(
            path.join(trial_dir, "models", "best_loss*.h5"))]
        
        print_("Making predictions")
        dice_scores = []
        for test_idx in range(len(test_imgs)):
            test_img = test_imgs[test_idx]
            test_seg = test_segs[test_idx]
            masked_image_s, segmentation_s, mask_s, dice = predict(
                input_s = test_img,
                label_s = test_seg,
                models = models, 
                window_size = window_size,
                resample_spacing = resample_spacing,
                resample_min_shape = window_size,
                normalization = normalization
            )
            dice_scores.append(dice)
            sitk.WriteImage(segmentation_s, path.join(
                    trial_generated_dir, f"test_{test_idx}.nii.gz"))
            sitk.WriteImage(masked_image_s, path.join(
                    trial_masked_dir, f"test_{test_idx}.nii.gz"))
        return 1-(sum(dice_scores) / len(dice_scores))
        
    except Exception as err:
        print_("Failed during evaluation.")
        raise(err)

    return 2.

# Wait a random number of seconds to desync parallel jobs
sleep(np.random.randint(0, 60))

from optuna.samplers import TPESampler

sampler = TPESampler(n_startup_trials=10)

if path.exists(optuna_db):
    study = optuna.load_study(
        study_name="ct_stroke_opt",
        storage="sqlite:///" + optuna_db,
        sampler=sampler)
else:
    study = optuna.create_study(
        study_name="ct_stroke_opt",
        storage="sqlite:///" + optuna_db,
        sampler=sampler)

study.optimize(objective, n_trials=100)

print_(study.best_params)
