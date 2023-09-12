import numpy as np
import tensorflow.keras.backend as K

def categorical_dice_coefficient(class_idx, name):
    def dice_coef(y_true, y_pred):
        yt_argmax = K.argmax(y_true)
        yp_argmax = K.argmax(y_pred)
        yt_class = K.cast(K.equal(yt_argmax, class_idx), K.floatx())
        yp_class = K.cast(K.equal(yp_argmax, class_idx), K.floatx())
        intersection = K.sum(yt_class * yp_class)
        return (2. * intersection + K.epsilon()) / (
            K.sum(yt_class) + K.sum(yp_class) + K.epsilon())
    metric = dice_coef
    metric.__name__ = name
    return metric

def mean_dice_np(y_true, y_pred, num_classes):
    dices = []
    for c in range(1, num_classes):
        y_class_true = (y_true == c) * 1.
        y_class_pred = (y_pred == c) * 1.

        nom = np.sum(y_class_pred * y_class_true)
        den = np.sum(y_class_pred) + np.sum(y_class_true)

        dc = (2 * nom + 1e-7) / (den + 1e-7)
        dices.append(dc)
    return sum(dices) / len(dices)
