import os
from os import path
import time
from typing import Callable, Dict, Iterable, Iterator, List, Optional, Tuple
from zipfile import ZipFile
from functools import partial
import multiprocessing

import numpy as np
import SimpleITK as sitk
import tensorflow as tf
tf.compat.v1.disable_eager_execution()
import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import Input, Conv3D, concatenate, Activation
from tensorflow.keras.layers import MaxPooling3D, UpSampling3D
from tensorflow_addons.layers import InstanceNormalization

from scipy import ndimage
from tqdm import tqdm

def print_(*args, **kwargs):
    print(*args, **kwargs, flush=True)

def resample( 
    image: sitk.Image, min_shape: List[int], method=sitk.sitkLinear, 
    new_spacing: List[float]=[1, 1, 3.6]
    ) -> sitk.Image:
    """Resamples an image to given target spacing and shape.

    Parameters:
    image: Input image (SITK).
    shape: Minimum output shape for the underlying array.
    method: SimpleITK interpolator to use for resampling. 
        (e.g. sitk.sitkNearestNeighbor, sitk.sitkLinear)
    new_spacing: The new spacing to resample to.

    Returns:
    int: Resampled image

   """

    # Extract size and spacing from the image
    size = image.GetSize()
    spacing = image.GetSpacing()

    # Determine how much larger the image will become with the new spacing
    factor = [sp / new_sp for sp, new_sp in zip(spacing, new_spacing)]

    # Determine the outcome size of the image for each dimension
    get_size = lambda size, factor, min_shape: max(int(size * factor), min_shape)
    new_size = [get_size(sz, f, sh) for sz, f, sh in zip(size, factor, min_shape)]

    # Resample the image 
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(image)
    resampler.SetOutputSpacing(new_spacing)
    resampler.SetSize(new_size)
    resampler.SetInterpolator(method)
    resampled_image = resampler.Execute(image)

    return resampled_image

def augment_noise(img, multiplier):
    noise = np.random.standard_normal(img.shape) * multiplier
    return img + noise

def augment_rotate(img, seg, angle):
    img = ndimage.rotate(img, angle, reshape=False, cval=-1)
    seg = ndimage.rotate(seg, angle, reshape=False, order=0)
    return img, seg

def augment_tilt(img, seg, angle):
    img = ndimage.rotate(img, angle, reshape=False, axes=(2,1), cval=-1)
    seg = ndimage.rotate(seg, angle, reshape=False, axes=(2,1), order=0)
    return img, seg

def augment_elastic(img, seg, alpha, sigma):
    shape = img.shape[0:3]
    dx = ndimage.gaussian_filter(
        input=(np.random.rand(*shape) * 2 - 1), 
        sigma=sigma, 
        mode="constant", 
        cval=0) * alpha
    dy = ndimage.gaussian_filter(
        input=(np.random.rand(*shape) * 2 - 1), 
        sigma=sigma, 
        mode="constant", 
        cval=0) * alpha
    dz = ndimage.gaussian_filter(
        input=(np.random.rand(*shape) * 2 - 1), 
        sigma=sigma, 
        mode="constant", 
        cval=0) * alpha
    x, y, z = np.meshgrid(
        np.arange(shape[0]), 
        np.arange(shape[1]), 
        np.arange(shape[2]), 
        indexing='ij')
    indices = (
        np.reshape(x + dx, (-1, 1)), 
        np.reshape(y + dy, (-1, 1)), 
        np.reshape(z + dz, (-1, 1)))

    img_elastic = ndimage.map_coordinates(
        input=img, 
         coordinates=indices
        ).reshape(shape)
    seg_elastic = ndimage.map_coordinates(
        input=seg, 
        coordinates=indices, 
        order=0
        ).reshape(shape)
    return img_elastic, seg_elastic

def augment(img, seg, 
    noise_chance = 0.3,
    noise_mult_max = 1e-3,
    rotate_chance = 0.1,
    rotate_max_angle = 30,
    tilt_chance = 0.1,
    tilt_max_angle = 30
    ):

    if np.random.uniform() < noise_chance:
        img = augment_noise(img, np.random.uniform(0., noise_mult_max))
        
    if np.random.uniform() < rotate_chance:
        img, seg = augment_rotate(img, seg, np.random.uniform(0-rotate_max_angle, rotate_max_angle))
    
    if np.random.uniform() < tilt_chance:
        img, seg = augment_tilt(img, seg, np.random.uniform(0-tilt_max_angle, tilt_max_angle))

    return img, seg

def random_crop(img, seg, shape):
    possible_offsets = [
        a_len - c_len for a_len, c_len in zip(img.shape, shape)]
    
    crop_offsets = [
        np.random.randint(0, offset) for offset in possible_offsets]
    
    z_max = possible_offsets[2]
    z = np.random.randint(0, z_max)
    crop_offsets[2] = z

    img_crop = img[
        crop_offsets[0]:crop_offsets[0] + shape[0],
        crop_offsets[1]:crop_offsets[1] + shape[1],
        crop_offsets[2]:crop_offsets[2] + shape[2],
        ]

    seg_crop = seg[
        crop_offsets[0]:crop_offsets[0] + shape[0],
        crop_offsets[1]:crop_offsets[1] + shape[1],
        crop_offsets[2]:crop_offsets[2] + shape[2],
        ]
    
    return img_crop, seg_crop

def get_generator(
    shape: Tuple,
    input_images: List[np.ndarray], 
    output_images: List[np.ndarray],
    num_classes: int,
    batch_size: Optional[int] = 5, 
    indexes: Optional[List[int]] = None, 
    shuffle: bool = False, 
    augmentation = True,
    rotation_freq = 0.1,
    tilt_freq = 0.1,
    noise_freq = 0.3, 
    noise_mult = 1e-3,
    ) -> Iterator[Tuple[dict, dict]]:
    """
    Returns a (training) generator for use with model.fit().

    Parameters:
    input_modalities: List of modalty names to include.
    output_modalities: Names of the target modalities.
    batch_size: Number of images per batch (default: all).
    indexes: Only use the specified image indexes.
    shuffle: Shuffle the lists of indexes once at the beginning.
    augmentation: Apply augmentation or not (bool).
    """

    num_rows = len(input_images)

    if indexes == None:
        indexes = list(range(num_rows))

    if type(indexes) == int:
        indexes = list(range(indexes))

    if batch_size == None:
        batch_size = len(indexes)  

    idx = 0

    # Prepare empty batch placeholder with named inputs and outputs
    input_batch = np.zeros((batch_size,) + shape + (1,))
    output_batch = np.zeros((batch_size,) + shape + (num_classes,))

    # Loop infinitely to keep generating batches
    while True:
        # Prepare each observation in a batch
        for batch_idx in range(batch_size):
            # Shuffle the order of images if all indexes have been seen
            if idx == 0 and shuffle:
                np.random.shuffle(indexes)

            current_index = indexes[idx]

            # Insert the augmented images into the input batch
            img_crop, seg_crop = random_crop(
                img=input_images[current_index], 
                seg=output_images[current_index],
                shape=shape)
            if augmentation:
                img_crop, seg_crop = augment(img_crop, seg_crop,
                noise_chance = noise_freq,
                noise_mult_max = noise_mult,
                rotate_chance = rotation_freq,
                tilt_chance = tilt_freq)
            input_batch[batch_idx] = img_crop[..., None]
            output_batch[batch_idx] = to_categorical(
                y=seg_crop, 
                num_classes=num_classes)

            # Increase the current index and modulo by the number of rows
            #  so that we stay within bounds
            idx = (idx + 1) % len(indexes)
                
        yield input_batch, output_batch

class IntermediateImages(Callback):
    def __init__(self, validation_set, prefix, 
        num_images=10):
        self.prefix = prefix
        self.num_images = num_images
        self.validation_set = (
            validation_set[0][:num_images, ...],
            validation_set[1][:num_images, ...]
            )
        
        # Export scan crops and targets once
        # they don't change during training so we export them only once
        for i in range(self.num_images):
            img_s = sitk.GetImageFromArray(self.validation_set[0][i].squeeze().T)
            seg_s = sitk.GetImageFromArray(
                np.argmax(self.validation_set[1][i], axis=-1).squeeze().astype(np.float32).T)
            sitk.WriteImage(img_s, f"{prefix}_{i:03d}_img.nii.gz")
            sitk.WriteImage(seg_s, f"{prefix}_{i:03d}_seg.nii.gz")

    def on_epoch_end(self, epoch, logs={}):
        # Predict on the validation_set
        predictions = self.model.predict(self.validation_set, batch_size=1)
        
        for i in range(self.num_images):
            prd_s = sitk.GetImageFromArray(
                np.argmax(predictions[i], axis=-1).astype(np.float32).squeeze().T)
            sitk.WriteImage(prd_s, f"{self.prefix}_{i:03d}_pred.nii.gz")

def categorical_focal_loss(alpha, gamma=2.):
    """
    Softmax version of focal loss.
    When there is a skew between different categories/labels in your data set, you can try to apply this function as a
    loss.
           m
      FL = ∑  -alpha * (1 - p_o,c)^gamma * y_o,c * log(p_o,c)
          c=1
      where m = number of classes, c = class and o = observation
    Parameters:
      alpha -- the same as weighing factor in balanced cross entropy. Alpha is used to specify the weight of different
      categories/labels, the size of the array needs to be consistent with the number of classes.
      gamma -- focusing parameter for modulating factor (1-p)
    Default value:
      gamma -- 2.0 as mentioned in the paper
      alpha -- 0.25 as mentioned in the paper
    References:
        Official paper: https://arxiv.org/pdf/1708.02002.pdf
        https://www.tensorflow.org/api_docs/python/tf/keras/backend/categorical_crossentropy
    Usage:
     model.compile(loss=[categorical_focal_loss(alpha=[[.25, .25, .25]], gamma=2)], metrics=["accuracy"], optimizer=adam)
    """

    alpha = np.array(alpha, dtype=np.float32)

    def categorical_focal_loss_fixed(y_true, y_pred):
        """
        :param y_true: A tensor of the same shape as `y_pred`
        :param y_pred: A tensor resulting from a softmax
        :return: Output tensor.
        """

        # Clip the prediction value to prevent NaN's and Inf's
        epsilon = K.epsilon()
        y_pred = K.clip(y_pred, epsilon, 1. - epsilon)

        # Calculate Cross Entropy
        cross_entropy = -y_true * K.log(y_pred)

        # Calculate Focal Loss
        loss = alpha * K.pow(1 - y_pred, gamma) * cross_entropy

        # Compute mean loss in mini_batch
        return K.mean(K.sum(loss, axis=-1))

    return categorical_focal_loss_fixed

def build_unet(window_size, num_classes, 
    l2_regularization=0.0001, 
    instance_norm=False):
    # Default parameters for conv layers
    c_defaults = {
        "kernel_size" : (3,3,3),
        "kernel_initializer" : 'he_normal',
        "padding" : 'same'
    }
    in_defaults = {
        "axis": -1,
        "center": True, 
        "scale": True,
        "beta_initializer": "random_uniform",
        "gamma_initializer": "random_uniform"
    }

    # Create NAMED input layers for each sequence
    ct_input  = Input(window_size)

    # Contraction path
    # he_normal defines initial weights - it is a truncated normal distribution (Gaussian dist.)
    # sets padding to same, meaning that input dimensions are the same as output dimensions
    c1 = Conv3D(16, kernel_regularizer = l2(l2_regularization), **c_defaults)(ct_input)
    c1 = Conv3D(16, kernel_regularizer = l2(l2_regularization), **c_defaults)(c1)
    if instance_norm:
        c1 = InstanceNormalization(**in_defaults)(c1)
    c1 = Activation('relu')(c1)
    p1 = MaxPooling3D((2, 2, 2))(c1)

    c2 = Conv3D(32, kernel_regularizer = l2(l2_regularization), **c_defaults)(p1)
    c2 = Conv3D(32, kernel_regularizer = l2(l2_regularization), **c_defaults)(c2)
    if instance_norm:
        c2 = InstanceNormalization(**in_defaults)(c2)
    c2 = Activation('relu')(c2)
    p2 = MaxPooling3D((2, 2, 2))(c2)

    c3 = Conv3D(64, kernel_regularizer = l2(l2_regularization), **c_defaults)(p2)
    c3 = Conv3D(64, kernel_regularizer = l2(l2_regularization), **c_defaults)(c3)
    if instance_norm:
        c3 = InstanceNormalization(**in_defaults)(c3)
    c3 = Activation('relu')(c3)
    p3 = MaxPooling3D((2, 2, 2))(c3)

    c4 = Conv3D(128, kernel_regularizer = l2(l2_regularization), **c_defaults)(p3)
    c4 = Conv3D(128, kernel_regularizer = l2(l2_regularization), **c_defaults)(c4)
    if instance_norm:
        c4 = InstanceNormalization(**in_defaults)(c4)
    c4 = Activation('relu')(c4)
    p4 = MaxPooling3D((2, 2, 2))(c4)

    c5 = Conv3D(256, kernel_regularizer = l2(l2_regularization), **c_defaults)(p4)
    c5 = Conv3D(256, kernel_regularizer = l2(l2_regularization), **c_defaults)(c5)
    if instance_norm:
        c5 = InstanceNormalization(**in_defaults)(c5)
    c5 = Activation('relu')(c5)

    # Upwards U part
    u6 = UpSampling3D((2, 2, 2))(c5)
    u6 = concatenate([u6, c4], axis=-1)
    c6 = Conv3D(128, kernel_regularizer = l2(l2_regularization), **c_defaults)(u6)
    c6 = Conv3D(128, kernel_regularizer = l2(l2_regularization), **c_defaults)(c6)
    if instance_norm:
        c6 = InstanceNormalization(**in_defaults)(c6)
    c6 = Activation('relu')(c6)

    u7 = UpSampling3D((2, 2, 2))(c6)
    u7 = concatenate([u7, c3], axis=-1)
    c7 = Conv3D(64, kernel_regularizer = l2(l2_regularization), **c_defaults)(u7)
    c7 = Conv3D(64, kernel_regularizer = l2(l2_regularization), **c_defaults)(c7)
    if instance_norm:
        c7 = InstanceNormalization(**in_defaults)(c7)
    c7 = Activation('relu')(c7)

    u8 = UpSampling3D((2, 2, 2))(c7)
    u8 = concatenate([u8, c2], axis=-1)
    c8 = Conv3D(32, kernel_regularizer = l2(l2_regularization), **c_defaults)(u8)
    c8 = Conv3D(32, kernel_regularizer = l2(l2_regularization), **c_defaults)(c8)
    if instance_norm:
        c8 = InstanceNormalization(**in_defaults)(c8)
    c8 = Activation('relu')(c8)

    u9 = UpSampling3D((2, 2, 2))(c8)
    u9 = concatenate([u9, c1], axis=-1)
    c9 = Conv3D(16, kernel_regularizer = l2(l2_regularization), **c_defaults)(u9)
    c9 = Conv3D(16, kernel_regularizer = l2(l2_regularization), **c_defaults)(c9)
    if instance_norm:
        c9 = InstanceNormalization(**in_defaults)(c9)
    c9 = Activation('relu')(c9)

    # Perform 1x1x1 convolution and reduce the feature maps to N channels.
    output_layer = Conv3D(num_classes, (1, 1, 1), 
        padding='same', 
        activation='softmax'
        )(c9)

    unet = Model(
        inputs=ct_input,
        outputs=output_layer
        )

    return unet

def create_zip(dirname, zip_path):
    with ZipFile(zip_path, 'w') as zipObj:
        # Iterate over all the files in directory
        for folderName, subfolders, filenames in os.walk(dirname):
            for filename in filenames:
                #create complete filepath of file in directory
                filePath = os.path.join(folderName, filename)
                # Add file to zip
                zipObj.write(filePath, path.basename(filePath))

def filter_components_by_size(segmentation, min_size = 0, max_size = None):
    out_mask = np.copy(segmentation)

    # Multiply by one in case argument is a boolean array
    labels, num_features = ndimage.label(segmentation * 1)

    for i in range(1, num_features):
        print(f"prrp {i} {num_features}", flush=True)
        # Extract the component 
        component_mask = (labels == i) * 1

        # Calculate the number of voxels in the component
        component_size = np.sum(component_mask)

        if component_size < min_size: 
            out_mask[component_mask] = 0
        if max_size and component_size > max_size: 
            out_mask[component_mask] = 0
    
    return out_mask

def load_quick(
    paths: List[str],
    preprocessing_function: Optional[Callable] = None,
    num_workers: Optional[int] = None,
    **kwargs):

    # Start a pool of workers.
    pool = multiprocessing.Pool(processes=num_workers)
    _partial = partial(preprocessing_function)

    # Apply the load and preprocess function for each file in the given paths
    data_list = pool.map(_partial, paths)
    
    # Aggregate the data in the first axis.
    data = np.stack(data_list, axis=0)

    return data

def resample_to_reference(image, ref_img, 
    interpolator=sitk.sitkNearestNeighbor, 
    default_pixel_value=0):
    resampled_img = sitk.Resample(image, ref_img, 
            sitk.Transform(), 
            interpolator, default_pixel_value, 
            ref_img.GetPixelID())
    return resampled_img

def _dice_coef(y_true, y_pred, smooth=1):
    """
    Dice = (2*|X & Y|)/ (|X|+ |Y|)
         =  2*sum(|A*B|)/(sum(A^2)+sum(B^2))
    ref: https://arxiv.org/pdf/1606.04797v1.pdf
    """
    intersection = K.sum(K.abs(y_true * y_pred))
    return (2. * intersection + smooth) / (K.sum(K.square(y_true)) + K.sum(K.square(y_pred)) + smooth)

def dice_coef_multilabel(y_true, y_pred, numLabels=3):
    dice=0
    for index in range(1, numLabels):
        dice -= _dice_coef(y_true[:,:,:,:,index], y_pred[:,:,:,:,index])
    return dice

class ConditionalStopping(Callback):
    def __init__(self,
               monitor='val_loss',
               threshold=0,
               after_epochs=0,
               verbose=0,
               mode='auto'):
        super(ConditionalStopping, self).__init__()

        self.monitor = monitor
        self.threshold = threshold
        self.after_epochs = after_epochs
        self.verbose = verbose
        self.stopped_epoch = 0
        if mode not in ['min', 'max']: 
            mode = 'min'
        self.mode = mode

    def on_train_begin(self, logs=None):
        # Allow instances to be re-used        
        self.sufficient = False

    def on_epoch_end(self, epoch, logs=None):
        if self.sufficient: 
            return

        current = self.get_monitor_value(logs)
        if current is None:
            return

        if self.mode == 'min' and current <= self.threshold:
            self.sufficient = True
        
        if self.mode == 'max' and current >= self.threshold:
            self.sufficient = True

        # Only check after the first epoch.
        if epoch == self.after_epochs and not self.sufficient:
            self.stopped_epoch = epoch
            self.model.stop_training = True

    def on_train_end(self, logs=None):
        if self.stopped_epoch > 0 and self.verbose > 0:
            print_(f'Epoch {self.stopped_epoch + 1}: conditional stopping')

    def get_monitor_value(self, logs):
        logs = logs or {}
        monitor_value = logs.get(self.monitor)
        if monitor_value is None:
            print_('Conditional stopping conditioned on metric `%s` '
                        'which is not available. Available metrics are: %s',
                        self.monitor, ','.join(list(logs.keys())))
        return monitor_value