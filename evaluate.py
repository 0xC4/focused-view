import os
from glob import glob
import argparse
from itertools import product
from datetime import datetime

import numpy as np
import SimpleITK as sitk

from helpers import *
from dbx import *
import wetransfer

from sklearn.model_selection import KFold

from tensorflow.keras.utils import get_custom_objects
from tensorflow.keras.models import load_model

from config import *
from predict import predict

def get_dices(y_true, y_pred, num_classes):
    dices = []
    for c in range(1, num_classes):
        y_class_true = (y_true == c) * 1.
        y_class_pred = (y_pred == c) * 1.

        nom = np.sum(y_class_pred * y_class_true)
        den = np.sum(y_class_pred) + np.sum(y_class_true)

        dc = (2 * nom + 1e-7) / (den + 1e-7)
        dices.append(dc)
    return dices

#def predict(input_s, 
#    models: list, 
#    window_size: list,
#    label_s = None,
#    resample_spacing=None,
#    resample_min_shape=None,
#    normalization="znorm"):
#    # Calculate dice score
#    if label_s is not None:
#        return masked_image_s, segmentation_s, mask_s, mean_dice_np(segmentation_n, resampled_label_n, 3)
#    return masked_image_s, segmentation_s, mask_s

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Predict heatmap on SITK images.')
    parser.add_argument('--input', '-i', metavar='[path]',
                        help='path to text file containing image filepaths. (.nii.gz recommended)')
    parser.add_argument('--models', '-m', metavar='[path]', nargs='+',
                        help='path to h5 models. If multiple then prediction is done by ensemble.')
    args = parser.parse_args()

    random_id = np.random.randint(111111,999999)

    # Fetch all available segmentations
    print_("> Downloading available segmentation files..")
    s_dir = path.join(PROJECT_DIR, "segmentations/")
    m_dir = path.join(PROJECT_DIR, f"segmentations/manual/{random_id}")
    os.makedirs(m_dir, exist_ok=True)
    
    # Clean the directory to be sure
    for filename in glob(path.join(m_dir, "*.nii.gz")):
        os.remove(filename)
    
    download_and_extract(
        url=DBX_PULL_LINK, 
        target_dir=m_dir, 
        tmp_name=path.join(s_dir, f"manual_tmp_{random_id}.zip"))
        
    # Match images and segmentations
    all_images = glob(path.join(IMAGE_DIR, "*.nii.gz"))
    all_segmentations = glob(path.join(m_dir, "*.nii.gz"))
    
    # Find all segmentations and images with matching segmentations
    img_paths, seg_paths = [], []
    for bn in sorted([path.basename(s) for s in all_segmentations]):
        img_path = path.join(IMAGE_DIR, bn)
        seg_path = path.join(m_dir, bn)
    
        if path.exists(img_path) and path.exists(seg_path):
            img_paths.append(img_path)
            seg_paths.append(seg_path)
            print_("> Found matching image and segmentation for:", bn)
    
    # Read and preprocess the images and segmentations
    print_("> Preprocessing images and segmentations")    
    imgs, segs = [], []
    for img_path, seg_path in tqdm(zip(img_paths, seg_paths)):
        img = sitk.ReadImage(img_path, sitk.sitkFloat32)
        seg = sitk.ReadImage(seg_path, sitk.sitkUInt8)
        imgs.append(img)
        segs.append(seg)
    
    test_imgs = [imgs.pop() for _ in range(10)]
    test_segs = [segs.pop() for _ in range(10)]
    
    # Initialize the model
    def fake_loss(*args, **kwargs):
        return 0.

    get_custom_objects().update({"loss": fake_loss,
        "categorical_focal_loss_fixed": fake_loss,
        "external_artery": fake_loss,
        "brain": fake_loss,
        "mean_dice": fake_loss,
        "dice_coef_multilabel": fake_loss})
    
    models = []
    for m_idx, m in enumerate(args.models): 
        print(f"> Reading model #{m_idx+1}:", m)
        models.append(load_model(m))
    
    num_folds = 7
    scores = {}
    for i in range(num_folds):
        print_(f"Getting performance of model #{i+1}")
        
        scores[f"model_{i}"] = {
            "brain_val": [],
            "artery_val": [],
            "brain_test": [],
            "artery_test": []
        }
    
        # Split the data
        kfold = KFold(num_folds, shuffle=True, random_state=123)
        _, valid_idxs = list(kfold.split(segs))[i]
        valid_idxs = list(valid_idxs)
        
        for valid_idx in valid_idxs:
            valid_img = imgs[valid_idx]
            valid_seg = segs[valid_idx]
            
            _, pred_s, _ = predict(
                models = [models[i]],
                input_s = valid_img,
                window_size = (160, 160, 48),
                normalization="znorm")
            
            label_n = sitk.GetArrayFromImage(valid_seg).T            
            pred_n = sitk.GetArrayFromImage(pred_s).T
            
            dice_scores = get_dices(label_n, pred_n, num_classes = 3)
            print("DICES:", dice_scores)
            scores[f"model_{i}"]["artery_val"].append(dice_scores[0])
            scores[f"model_{i}"]["brain_val"].append(dice_scores[1])
        
        for test_img, test_seg in zip(test_imgs, test_segs):
            _, pred_s, _ = predict(
                models = [models[i]],
                input_s = test_img,
                window_size = (160, 160, 48),
                normalization="znorm")
            
            label_n = sitk.GetArrayFromImage(test_seg).T            
            pred_n = sitk.GetArrayFromImage(pred_s).T
            
            dice_scores = get_dices(label_n, pred_n, num_classes = 3)
            scores[f"model_{i}"]["artery_test"].append(dice_scores[0])
            scores[f"model_{i}"]["brain_test"].append(dice_scores[1])
        
    # Predict the test data with the full ensemble
    scores["ensemble"] = {
        "artery_test" : [],
        "brain_test" : []
    }
    
    for test_img, test_seg in zip(test_imgs, test_segs):
        masked_image, pred_s, _ = predict(
            models = models,
            input_s = test_img,
            window_size = (160, 160, 48),
            normalization="znorm")
        
        label_n = sitk.GetArrayFromImage(test_seg).T            
        pred_n = sitk.GetArrayFromImage(pred_s).T
        
        dice_scores = get_dices(label_n, pred_n, num_classes = 3)
        scores["ensemble"]["artery_test"].append(dice_scores[0])
        scores["ensemble"]["brain_test"].append(dice_scores[1])

    # TODO: Fix this
    # TODO: Melted output
    
    # Write everything to file
    print_("> Writing images to file...")
    bn = path.basename(filename)
    sitk.WriteImage(pred_s, path.join(generated_dir, bn))
    sitk.WriteImage(masked_image, path.join(masked_image_dir, bn))
    print_("> Done.")

    print_("> Zipping results..")
    create_zip(generated_dir, path.join(segmentation_dir, "generated.zip"))
    create_zip(masked_image_dir, path.join(segmentation_dir, "masked.zip"))

    # Upload the masked scans and segmentations to wetransfer
    print_("> Uploading segmentations and masks")
    wetransfer_link_segs = wetransfer.upload([
        path.join(segmentation_dir, "generated.zip")], 
        message=f"Segmentations ({PROJECT_NAME})",
        sender='x.xxxxxxxxx@xxx.xx')

    wetransfer_link_masked = wetransfer.upload([
        path.join(segmentation_dir, "masked.zip")], 
        message=f"Masked scans ({PROJECT_NAME})",
        sender='x.xxxxxxxxx@xxx.xx')
    
    print_("> Wetransfer link (segmentations):", wetransfer_link_segs)
    print_("> Wetransfer link (masked scans): ", wetransfer_link_masked)

    print_("> Done")
