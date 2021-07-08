from PIL import Image
from tensorflow.python.keras.losses import categorical_crossentropy
from utils import draw_boxes
import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.losses import binary_crossentropy, cosine_similarity, mean_squared_error

smooth = 1.

def interpretable_iou(y_true, y_pred, domain_mask):
    """
    y is the list of boxes (x1, y1, x2, y2)
    domain_mask is a binary mask of regions to consider in it (PIL object)
    """
    gt_mask = draw_boxes(Image.new('L', domain_mask.size), y_true, "#ffffff", fill="#ffffff")
    pred_mask = draw_boxes(Image.new('L', domain_mask.size), y_pred, "#ffffff", fill="#ffffff")
    gt_mask = (np.array(gt_mask) // 255) * (np.array(domain_mask) // 255)
    pred_mask = (np.array(pred_mask) // 255) * (np.array(domain_mask) // 255)
    intersection = np.sum(gt_mask * pred_mask) + 1.
    union = np.sum(np.clip(gt_mask+pred_mask, 0, 1)) + 1.
    return intersection / union

def SSIMLoss(y_true, y_pred):
    return 1 - tf.reduce_mean(tf.image.ssim(y_true, y_pred, 1))

def SSIM(y_true, y_pred):
    return tf.reduce_mean(tf.image.ssim(y_true, y_pred, 1))
    
def MSELoss(batch_size):
    def MSE(y_true, y_pred):
        y_true = tf.math.sigmoid(y_true)
        y_pred = tf.math.sigmoid(y_pred)
        y_true = tf.reshape(y_true, (batch_size, -1))
        y_pred = tf.reshape(y_pred, (batch_size, -1))
        return tf.math.reduce_mean(tf.math.squared_difference(y_true, y_pred))
    return MSE

def dice_coef(y_true, y_pred):
    y_true_c = tf.cast(y_true, tf.float32)
    y_true_f = tf.keras.layers.Flatten()(y_true_c)
    y_pred_f = tf.keras.layers.Flatten()(y_pred)
    intersection = tf.math.multiply(y_true_f, y_pred_f)
    intersection = tf.reduce_sum(intersection)
    return (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)

def dice_loss(factor):
    def d_loss(y_true, y_pred):
        return factor * (1.0 - dice_coef(y_true, y_pred))
    return d_loss

def bce_dice_loss(y_true, y_pred):
    return 0.2 * binary_crossentropy(y_true, y_pred) + 0.8 * dice_loss(y_true, y_pred)

def focal_loss(y_true, y_pred):
    # y_pred = y_pred[..., :5]
    alpha = 0.25
    gamma = 2
    y_true = tf.reshape(y_true[..., -1], [-1, y_pred.shape[1]*y_pred.shape[2]])
    y_pred = tf.reshape(y_pred[..., -1], [-1, y_pred.shape[1]*y_pred.shape[2]])

    def focal_loss_with_logits(logits, targets, alpha, gamma, y_pred):
        weight_a = alpha * (1 - y_pred) ** gamma * targets
        weight_b = (1 - alpha) * y_pred ** gamma * (1 - targets)
        return (tf.math.log1p(tf.exp(-tf.abs(logits))) + tf.nn.relu(-logits)) * (weight_a + weight_b) + logits * weight_b

    y_pred = tf.clip_by_value(y_pred, tf.keras.backend.epsilon(), 1 - tf.keras.backend.epsilon())
    logits = tf.math.log(y_pred / (1 - y_pred))
    loss = focal_loss_with_logits(logits=logits, targets=y_true, alpha=alpha, gamma=gamma, y_pred=y_pred)
    # or reduce_sum and/or axis=-1
    return tf.reduce_mean(loss)


# def crossentropy_loss(num_classes):
#     def loss_func(y_true, y_pred):
#         y_true = tf.cast(y_true, tf.float32)
#         y_pred = tf.cast(y_pred, tf.float32)
#         y_true = tf.reshape(y_true, [-1, num_classes])
#         y_pred = tf.reshape(y_pred, [-1, num_classes])
#         loss = categorical_crossentropy(y_true, y_pred)
#         return loss
#     return loss_func


def tversky(y_true, y_pred):
    y_true_f = tf.cast(y_true, tf.float32)
    y_pred_f = tf.cast(y_pred, tf.float32)
    y_true_pos = K.flatten(y_true_f)
    y_pred_pos = K.flatten(y_pred_f)
    true_pos = K.sum(y_true_pos * y_pred_pos)
    false_neg = K.sum(y_true_pos * (1-y_pred_pos))
    false_pos = K.sum((1-y_true_pos)*y_pred_pos)
    alpha = 0.7
    return (true_pos + smooth)/(true_pos + alpha*false_neg + (1-alpha)*false_pos + smooth)

def tversky_loss(y_true, y_pred):
    return 1 - tversky(y_true,y_pred)

def focal_tversky_loss(y_true,y_pred):
    ti = tversky(y_true, y_pred)
    gamma = 1.
    return K.pow((1-ti), gamma)


def multiclass_dice_coeff(y_true, y_pred):
    """
    both tensors are [b, h, w, classes]
    returns a tensor with dice coeff values for each class
    """
    y_true_c = tf.cast(y_true, tf.float32)
    y_true_shape = tf.shape(y_true_c)
    # [b, h*w, classes]
    y_true_f = tf.reshape(y_true_c, [-1, y_true_shape[1]*y_true_shape[2], y_true_shape[3]])
    y_pred_f = tf.reshape(y_pred, [-1, y_true_shape[1]*y_true_shape[2], y_true_shape[3]])
    
    intersection = tf.math.multiply(y_true_f, y_pred_f)
    # [1, classes]
    intersection = 2 * tf.reduce_sum(intersection, axis=[0, 1]) + smooth
    total = tf.reduce_sum(y_true_f, axis=[0, 1]) + tf.reduce_sum(y_pred_f, axis=[0, 1]) + smooth

    return tf.math.divide(intersection, total)


def multiclass_dice_loss(loss_scales):
    total = np.sum(np.array(loss_scales))
    loss_scales = tf.convert_to_tensor(loss_scales, dtype=tf.float32)
    def md_loss(y_true, y_pred):
        return (total - tf.math.reduce_sum(loss_scales * multiclass_dice_coeff(y_true, y_pred))) / total
    return md_loss

def ch_dice_coeff(channel):
    def dice(y_true, y_pred):
        y_true = tf.slice(y_true, [0, 0, 0, channel], [-1, -1, -1, 1])
        y_pred = tf.slice(y_pred, [0, 0, 0, channel], [-1, -1, -1, 1])
        return multiclass_dice_coeff(y_true, y_pred)
    return dice


def cosine_loss(y_true, y_pred):
    labels = y_true[..., -1]
    y_true = y_true[..., 0:4]
    y_pred = y_pred[..., 0:4]

    cos_loss = 1. + cosine_similarity(y_true, y_pred, axis=-1)
    cos_loss = tf.where(labels == 1, cos_loss, 0.)

    loss = tf.reduce_mean(cos_loss)
    return loss


def uniclass_dice_coeff_0(y_true, y_pred):
    y_true0 = y_true[..., 0:1]
    y_pred0 = y_pred[..., 0:1]
    return multiclass_dice_coeff(y_true0, y_pred0)

def uniclass_dice_coeff_1(y_true, y_pred):
    y_true1 = y_true[..., 1:2]
    y_pred1 = y_pred[..., 1:2]
    return multiclass_dice_coeff(y_true1, y_pred1)

def uniclass_dice_coeff_2(y_true, y_pred):
    y_true2 = y_true[..., 2:3]
    y_pred2 = y_pred[..., 2:3]
    return multiclass_dice_coeff(y_true2, y_pred2)

def uniclass_dice_coeff_3(y_true, y_pred):
    y_true3 = y_true[..., 3:4]
    y_pred3 = y_pred[..., 3:4]
    return multiclass_dice_coeff(y_true3, y_pred3)

def uniclass_dice_coeff_4(y_true, y_pred):
    y_true4 = y_true[..., 4:5]
    y_pred4 = y_pred[..., 4:5]
    return multiclass_dice_coeff(y_true4, y_pred4)


class CLCR_CL():
    def __init__(self):
        self.alpha = 0.1
        self.tou = 0.1
    
    def cl_loss_func(self, y_true, y_pred):
        """
        y_true is one hot class vector of patches (S, num_classes)
        y_pred = (S, dim)
        """
        y_pred_norm = tf.math.l2_normalize(y_pred, axis=1)
        # sim_mat[i][j] = cosine angle between ith and jth embed
        sim_mat = tf.matmul(y_pred_norm, y_pred_norm, transpose_b=True) / self.tou
        # class_eq[i][j] = (class[i] == class[j])
        class_eq = tf.matmul(y_true, y_true, transpose_b=True)
        # neg_sims = sum or exp(similarity) of ith patch with every negative patch
        neg_sim = tf.math.exp(sim_mat) * (1 - class_eq)
        # The hard negative sample z = a*zi + (1 - a)*zn. sim(z, zi) = a + (1 - a)*sim(zn, zi)
        self.alpha = K.random_uniform([1], 0, 0.4, dtype=tf.float32)
        hard_neg_sim = tf.math.exp(self.alpha + (1. - self.alpha) * sim_mat) * (1 - class_eq)
        neg_sim = tf.reduce_sum(neg_sim, axis=1)
        neg_sim = tf.reshape(neg_sim, [-1, 1])
        # contrast[i][j] = -log( exp(ziT.zj) / (exp(ziT.zj) + sum over zneg exp(ziT.zneg)) ) if j is positive for i else zero
        numerator = tf.math.exp(sim_mat)
        denominator = numerator + neg_sim + hard_neg_sim
        contrast = (numerator / denominator)
        contrast = -tf.math.log(contrast) * (class_eq)
        loss_per_query = tf.reshape(tf.reduce_sum(contrast, axis=1), [-1, 1])
        num_positive_keys = tf.reshape(tf.reduce_sum(class_eq, axis=1), [-1, 1])
        loss_per_query = loss_per_query / num_positive_keys
        cl_loss = tf.reduce_mean(loss_per_query)
        return cl_loss



def normalized_similarity_loss(vec1, vec2):
    """
    Computes dot product similarity of 2 tensors
    """
    vec1 = tf.reshape(vec1, [-1])
    vec1 = tf.math.l2_normalize(vec1)
    vec2 = tf.reshape(vec2, [-1])
    vec2 = tf.math.l2_normalize(vec2)
    dotp = tf.reduce_sum(tf.multiply(vec1, vec2))
    return 1.0 - dotp
    # return mean_squared_error(vec1, vec2)


def similarity_loss(num_inputs=4):
    num_combinations = num_inputs * (num_inputs - 1)
    def _similarity_loss(y_true, y_pred):
        total_channels = y_pred.shape[-1]
        num_channels = int(total_channels / num_inputs)
        # print(total_channels, num_channels)

        feats = []
        for i in range(num_inputs):
            feats.append(y_pred[..., i*num_channels:(i+1)*num_channels])
        
        loss = 0.0
        for i in range(num_inputs):
            for j in range(num_inputs):
                if i != j:
                    sim_loss = normalized_similarity_loss(feats[i], feats[j])
                    loss += sim_loss
        return loss / num_combinations
    return _similarity_loss

def mean_absolute_val(y_true, y_pred):
    return tf.reduce_mean(tf.math.abs(y_pred))


def silhouette(y_true, y_pred):
    # A modified version of silhoutte score where avg inter class distance is calculated sample wise
    y_pred_norm = tf.math.l2_normalize(y_pred, axis=1)
    # sim_mat[i][j] = cosine angle between ith and jth embed
    dist_mat = 1. - tf.matmul(y_pred_norm, y_pred_norm, transpose_b=True)
    # class_eq[i][j] = (class[i] == class[j])
    class_eq = tf.matmul(y_true, y_true, transpose_b=True)
    a = tf.reduce_sum(dist_mat * class_eq) / tf.reduce_sum(class_eq)
    b = tf.reduce_sum(dist_mat * (1 - class_eq)) / tf.reduce_sum(1 - class_eq)
    s = (b - a) / (tf.maximum(a, b) + 1e-8)
    s = tf.minimum(1., s)
    return s
