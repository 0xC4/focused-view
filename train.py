import os
import sys
from os import path
from glob import glob

import numpy as np
import SimpleITK as sitk

from sklearn.model_selection import KFold

from helpers import *
from dual_attention_unet import build_dual_attention_unet
from config import *
from dbx import download_and_extract

def train(
    img_paths,
    seg_paths,
    work_dir,
    
    fold_num = 0,
    num_folds = 10,
    
    resample_spacing = None,
    window_size = (256, 256, 64),
    batch_size = 5,
    num_validation = 50,
    max_epochs = 5000,
    early_stopping = None,
    early_stopping_var = "val_loss",
    early_stopping_mode = "min",

    conditional_stopping_threshold = None,
    conditional_stopping_epochs = None,
    conditional_stopping_var = None,
    conditional_stopping_mode = None,


    loss = "binary_crossentropy",
    optimizer = "adam",

    rotation_freq = 0.1,
    tilt_freq = 0.1,
    noise_freq = 0.3, 
    noise_mult = 1e-3,

    unet_type = "dual_attention", # or simple
    l2_regularization = 0.0001,
    normalization = 'divide',
    instance_norm=True,

    class_names = ["background", "target"],

    num_samples = 10
    ):

    num_classes = len(class_names)

    print_("> Creating folder structure")
    segmentation_dir = path.join(work_dir, "segmentations/")
    manual_dir = path.join(segmentation_dir, "manual/")
    output_dir = path.join(work_dir, "output")
    sample_dir = path.join(work_dir, "samples")
    log_dir    = path.join(work_dir, "logs")
    model_dir  = path.join(work_dir, "models")
    current_manual_dir = path.join(manual_dir, f"fold{fold_num}")
    print_("> Work dir:", work_dir)

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(sample_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(manual_dir, exist_ok=True)
    os.makedirs(current_manual_dir, exist_ok=True)
    os.makedirs(path.join(segmentation_dir, "generated"), exist_ok=True)
    os.makedirs(path.join(segmentation_dir, "masked"), exist_ok=True)

    print_("> Fold number:", fold_num, "of", num_folds)

    # Read and preprocess the images and segmentations
    print_("> Preprocessing images and segmentations")    
    imgs, segs = [], []
    for img_path, seg_path in tqdm(zip(img_paths, seg_paths)):
        img, seg = preprocess(
            img=sitk.ReadImage(img_path), 
            seg=sitk.ReadImage(seg_path, sitk.sitkUInt8),
            resample_spacing=resample_spacing,
            resample_min_shape=window_size,
            normalization=normalization,
            dtype="numpy")
        imgs.append(img)
        segs.append(seg)

    # Get the number of observations
    num_obs = len(imgs)

    kfold = KFold(num_folds, shuffle=True, random_state=123)
    train_idxs, valid_idxs = list(kfold.split(segs))[fold_num]
    train_idxs = list(train_idxs)
    valid_idxs = list(valid_idxs)

    print_(f"Dataset division:\n- Train:", len(train_idxs),
        "- Valid:", len(valid_idxs))
    print_("Valid indexes:", valid_idxs)

    # print("> Exporting crop samples..")
    # for i in range(min(num_samples, num_obs)):
    #     x, y = random_crop(imgs[i], segs[i], shape = window_size)
    #     x, y = augment(x, y)
    #     x_s = sitk.GetImageFromArray(x.T)
    #     y_s = sitk.GetImageFromArray(y.T)
    #     sitk.WriteImage(x_s, 
    #         path.join(sample_dir, f"fold_{fold_num}_img_{i:02d}.nii.gz"))
    #     sitk.WriteImage(y_s, 
    #         path.join(sample_dir, f"fold_{fold_num}_seg_{i:02d}.nii.gz"))

    # Create a data generator that the model can train on
    train_generator = get_generator(
        batch_size=batch_size,
        shape=window_size,
        input_images=imgs,
        output_images=segs,
        num_classes=num_classes,
        indexes=train_idxs,
        shuffle=True,
        augmentation=True,
        rotation_freq = rotation_freq,
        tilt_freq = tilt_freq,
        noise_freq = noise_freq, 
        noise_mult = noise_mult
        )

    validation_generator = get_generator(
        batch_size=num_validation,
        shape=window_size,
        input_images=imgs,
        output_images=segs,
        num_classes=num_classes,
        indexes=valid_idxs,
        shuffle=True,
        augmentation=False)
    validation_set = next(validation_generator)
    print(f"Validation set: {validation_set[0].shape}")

    # Add a dice coefficient for each non-background class
    from dice_score import categorical_dice_coefficient
    dice_metrics = []
    for class_idx in range(1, num_classes):
        dice_metrics+=[categorical_dice_coefficient(class_idx, class_names[class_idx])] 

    # Add a mean dice score over all non-background classes
    def mean_dice(y_true, y_pred):
        return sum([m(y_true, y_pred) for m in dice_metrics])/len(dice_metrics)
    
    # Create the model and show summary
    if unet_type == "simple":
        dnn = build_unet(
            window_size=window_size + (1,), 
            num_classes=num_classes,
            l2_regularization=l2_regularization,
            instance_norm=instance_norm)
    else:
        dnn = build_dual_attention_unet(
            input_shape=window_size + (1,), 
            num_classes=num_classes,
            l2_regularization=l2_regularization,
            instance_norm=instance_norm)
    dnn.summary(line_length=160)
    dnn.compile(
        optimizer   = optimizer,
        loss        = loss,
        metrics     = dice_metrics + [mean_dice])

    from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, EarlyStopping
    callbacks = []

    # This callback predicts on a number of images for the test set after each epoch
    # and shows a few slices in PNG files in the "output/" folder
    callbacks += [
        IntermediateImages(
        validation_set  = validation_set, 
        prefix          = path.join(output_dir, f"train-fold{fold_num}"), 
        num_images      = 10)
    ]

    # This callback produces a log file of the training and validation metrics at 
    # each epoch
    callbacks += [CSVLogger(path.join(log_dir, f"train_fold{fold_num}.csv"))]

    # After every epoch store the model with the best validation performance for
    # each metric that we record
    for metric_name in [m.__name__ for m in dice_metrics] + ["loss", "mean_dice"]:
        callbacks += [
            ModelCheckpoint(
            path.join(model_dir, f"best_{metric_name}_fold{fold_num}.h5"),
            monitor         = f"val_{metric_name}", 
            save_best_only  = True, 
            mode            = 'min' if "loss" in metric_name else "max",
            verbose         = 1)
        ]

    if early_stopping:
        # Stop training after X epochs without improvement
        callbacks += [
            EarlyStopping(
            patience        = early_stopping,
            monitor         = early_stopping_var,
            mode            = early_stopping_mode,
            verbose         = 1)
        ]
    
    if conditional_stopping_epochs:
        callbacks += [
            ConditionalStopping(
                monitor=conditional_stopping_var,
                threshold=conditional_stopping_threshold,
                after_epochs=conditional_stopping_epochs,
                mode=conditional_stopping_mode
            )]

    # Train the model we created
    dnn.fit(train_generator,
        validation_data    = validation_set,
        steps_per_epoch    = len(train_idxs) // batch_size * 10, 
        epochs             = max_epochs,
        callbacks          = callbacks,
        verbose            = 1)

    print_("[I] Completed.")


if __name__ == "__main__":
    fold_num = int(sys.argv[1])
    num_folds = int(sys.argv[2])

    # Fetch all available segmentations
    print_("> Downloading available segmentation files..")
    s_dir = path.join(PROJECT_DIR, "segmentations/")
    m_dir = path.join(PROJECT_DIR, "segmentations/manual/")

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

    import optuna
    
    #opt_study = optuna.load_study(
    #   study_name="ct_stroke_opt", 
    #    storage="sqlite:////scratch/p286425/stroke_cta/optimization/optuna.db")
    #bp = opt_study.best_params
    
    bp = {
        "window_xy": 160,
        "window_z": 48,
        "unet_type": "dual_attention",
        "spacing": (0.7, 0.7, 3.),
        "batch_size": 13,
        "learning_rate": 4e-4,
        "normalization": "znorm",
        "reg": 2e-4,
        "inst_norm": True,
        "rotation_freq": 0.25,
        "tilt_freq": 0.04,
        "noise_freq": 0.6,
        "noise_mult": 0.003
    }
    
    print("Loading best settings:", bp)

    best_window = (bp["window_xy"], bp['window_xy'], bp['window_z'])

    from tensorflow.keras.optimizers import RMSprop
    train(
        img_paths=img_files,
        seg_paths=seg_files,
        work_dir=PROJECT_DIR,
        fold_num=fold_num,
        num_folds=num_folds,
        resample_spacing = bp["spacing"],
        window_size = best_window,
        batch_size = bp["batch_size"],
        num_validation = 50,
        max_epochs = 5000,
        early_stopping = 50,
        early_stopping_var = "val_loss",
        early_stopping_mode = "min",
        loss = dice_coef_multilabel,
        optimizer = RMSprop(bp["learning_rate"]),

        unet_type = bp["unet_type"],
        l2_regularization = bp["reg"],

        normalization=bp["normalization"],
        noise_freq=bp["noise_freq"],
        noise_mult=bp["noise_mult"],
        rotation_freq=bp["rotation_freq"],
        tilt_freq=bp["tilt_freq"],
        instance_norm=bp['inst_norm'],

        class_names = CLASSES,

        num_samples = 10
    )
    
