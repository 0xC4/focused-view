import os
from glob import glob
import argparse
from itertools import product
from datetime import datetime

import numpy as np
import SimpleITK as sitk
from tqdm import tqdm
from dice_score import mean_dice_np

from helpers import *
from dbx import *
import wetransfer

from tensorflow.keras.utils import get_custom_objects
from tensorflow.keras.models import load_model

from config import *

def resample_class_response_maps(prediction, reference_image, target_reference_image):
    """
    Resample the predicted segmentation using linear interpolation on the class
        response map to realise a more accurate and less blocky resampled
        segmentation than with nearest neighbour interpolation.
    """
    resampled_crms = []
    for class_num in range(prediction.shape[-1]):
        # Create an SITK image for the CRM
        crm_s = sitk.GetImageFromArray(prediction[..., class_num].T)
        
        # Copy the physical parameters from the reference image
        crm_s.CopyInformation(reference_image)

        # Resample it to the desired spacing
        crm_s = sitk.Resample(crm_s, target_reference_image, 
            sitk.Transform(), sitk.sitkLinear, 0, crm_s.GetPixelID())

        resampled_crms.append(sitk.GetArrayFromImage(crm_s).T)
    resampled_crms = np.stack(resampled_crms, axis=-1)
    return resampled_crms

def get_centroid(arr):
    x, y, z = np.where(arr == 1.)

    return np.round(x.mean()), np.round(y.mean()), np.round(z.mean())

def get_sphere_mask(center, diameter, shape, spacing):
    X, Y, Z = np.ogrid[:shape[0], :shape[1], :shape[2]]
    dist_from_center = np.sqrt(
        ((X - center[0]) * spacing[0])**2 + 
        ((Y - center[1]) * spacing[1])**2 + 
        ((Z - center[2]) * spacing[2])**2)
    sphere_mask = (dist_from_center <= diameter / 2) * 1.
    return sphere_mask

def predict(input_s, 
    models: list, 
    window_size: list,
    label_s = None,
    resample_spacing=None,
    resample_min_shape=None,
    normalization="znorm"):

    original_s = input_s
    input_n, _ = preprocess(input_s, 
        resample_spacing=resample_spacing,
        resample_min_shape=resample_min_shape,
        normalization=normalization,
        dtype="numpy")
    input_s, _ = preprocess(input_s, 
        resample_spacing=resample_spacing,
        resample_min_shape=resample_min_shape,
        normalization=normalization,
        dtype="sitk")

    # Calculate the possible crop origins to cover the full image with overlap
    possible_offsets = [
        a_len - c_len for a_len, c_len in zip(input_n.shape, window_size)]

    x_list, y_list, z_list = [], [], []
    x, y, z = 0,0,0

    while x <= possible_offsets[0]:
        x_list += [x]
        x += max(min(window_size[0]//4, possible_offsets[0]-x), 1)
    while y <= possible_offsets[1]:
        y_list += [y]
        y += max(min(window_size[1]//4, possible_offsets[1]-y), 1)
    while z <= possible_offsets[2]:
        z_list += [z]
        z += max(min(window_size[2]//4, possible_offsets[2]-z), 1)
    
    all_crop_origins = list(product(x_list, y_list, z_list))

    # Sliding window prediction
    full_prediction = np.zeros(input_n.shape + (NUM_CLASSES,))
    for x, y, z in tqdm(all_crop_origins, desc="Collecting predictions..."):
        img_crop = input_n[
            np.newaxis,
            x:x+window_size[0],
            y:y+window_size[1],
            z:z+window_size[2],
            np.newaxis]

        # Softmax predictions are collected for each of the CV models
        crop_pred = sum([m.predict(img_crop) for m in models]).squeeze()

        full_prediction[
            x:x+window_size[0],
            y:y+window_size[1],
            z:z+window_size[2],
            :] += crop_pred


    # Divide to obtain the average softmax predction across models
    full_prediction = full_prediction / np.sum(
        full_prediction, axis=-1)[..., np.newaxis]

    full_prediction[..., 1] *= 1.5

    # Resample the segmentation to the target spacing by applying linear 
    #  interpolation on the CRMs.
    full_prediction = resample_class_response_maps(
        prediction=full_prediction, 
        reference_image=input_s, 
        target_reference_image=original_s)

    # Collapse to a singular prediction
    segmentation_n = np.argmax(full_prediction, axis=-1)
    segmentation_n = segmentation_n.astype(np.float32).squeeze()

    # Create an image for the segmentation itself
    segmentation_s = sitk.GetImageFromArray(segmentation_n.T)
    segmentation_s.CopyInformation(original_s)

    # Mask organs
    mask = get_mask(segmentation_n)

    # Create SITK image for the mask
    mask_s = sitk.GetImageFromArray(mask.T)
    mask_s.CopyInformation(original_s)

    # Create SITK image for the masked image
    masked_image = sitk.GetArrayFromImage(original_s).T
    masked_image[mask == 0] = REPLACEMENT_VALUE
    masked_image_s = sitk.GetImageFromArray(masked_image.T)
    masked_image_s.CopyInformation(original_s)

    # Calculate dice score
    if label_s is not None:
        resampled_label_s = resample_to_reference(label_s, segmentation_s)
        resampled_label_n = sitk.GetArrayFromImage(resampled_label_s).T

        return masked_image_s, segmentation_s, mask_s, mean_dice_np(segmentation_n, resampled_label_n, 3)


    # Return all created images
    return masked_image_s, segmentation_s, mask_s

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Predict heatmap on SITK images.')
    parser.add_argument('--input', '-i', metavar='[path]',
                        help='path to text file containing image filepaths. (.nii.gz recommended)')
    parser.add_argument('--models', '-m', metavar='[path]', nargs='+',
                        help='path to h5 models. If multiple then prediction is done by ensemble.')
    args = parser.parse_args()

    with open(args.input.strip(), 'r') as f:
        input_files = [fn.strip() for fn in f.readlines()]

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

    segmentation_dir = path.join(PROJECT_DIR, "segmentations_sens")
    generated_dir = path.join(segmentation_dir, "generated")
    masked_image_dir = path.join(segmentation_dir, "masked")

    os.makedirs(generated_dir, exist_ok=True)
    os.makedirs(masked_image_dir, exist_ok=True)

    #for filename in glob(path.join(generated_dir, "*.nii.gz")):
    #    os.remove(filename)
    #for filename in glob(path.join(masked_image_dir, "*.nii.gz")):
    #    os.remove(filename)

    # Read each image
    for i, filename in enumerate(input_files):
        print_(f"> Processing image #{i+1}: {filename}")
        input_s = sitk.ReadImage(filename, sitk.sitkFloat32)

        # Get the prediction
        masked_image, segmentation, mask = predict(
            input_s=input_s, 
            models=models, 
            window_size=(160, 160, 48),
            normalization="znorm")

        # Write everything to file
        print_("> Writing images to file...")
        bn = path.basename(filename)
        sitk.WriteImage(segmentation, path.join(generated_dir, bn))
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
        sender='XXXXXXXX@xxxxx.xxxxx')

    wetransfer_link_masked = wetransfer.upload([
        path.join(segmentation_dir, "masked.zip")], 
        message=f"Masked scans ({PROJECT_NAME})",
        sender='XXXXXXXX@xxxxx.xxxxx')
    
    print_("> Wetransfer link (segmentations):", wetransfer_link_segs)
    print_("> Wetransfer link (masked scans): ", wetransfer_link_masked)

    print_("> Done")
