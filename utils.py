from PIL import Image, ImageDraw
import numpy as np
import json
import tensorflow as tf
import cv2

def load_image(img_path):
    """
    Load an image at the index.
    Returns PIL image
    """
    img = Image.open(img_path).convert('RGB')
    # img = np.array(img)
    # print("Loaded image: ", image_path)
    return img


def load_annotations(anns_path):
    """
    Load annotations for an image_index.
    """
    labels = json.load(open(anns_path))
    labels = labels["shapes"]
    return labels.copy()


def get_ground_truth(img_path, anns_path, target_classes):
    """
    Input: The instance id path
    Output: The image and the ground truth binary mask images (PIL)
    """
    img = load_image(img_path)
    anns = load_annotations(anns_path)
    image_size = img.size

    masks = {label: Image.new("L", image_size) for label in target_classes}
    draw_masks = {label: ImageDraw.Draw(masks[label]) for label in target_classes}
    # Get valid present annotations
    anns = [x for x in anns if type(x) == dict]

    # Gathering polygon points - A dict of points for each target class
    poly_pts = {label: [] for label in target_classes}
    for ann in anns:
        if ann['label'] in target_classes:
            poly_pts[ann['label']].append([np.rint(point) for point in ann['points']])

    # Drawing the masks
    for label in target_classes:
        polygons = poly_pts[label]
        for poly in polygons:
            coords = [(pt[0], pt[1]) for pt in poly]
            draw_masks[label].polygon(xy=coords, fill=255)
    
    return img, masks


def infer_image_with_anns(img_path, anns_path, image_size, target_classes, model, conf_threshold):
    """
    Inputs: the path to image, its annotations, (W,H), list of classes, preprocessing func, keras model object, confidence threshold
    Returns: The original image, the result mask image, combined 1 & 2, combined ground truth
    """
    raw_image, gt_masks = get_ground_truth(img_path, anns_path, target_classes)
    w, h = raw_image.size
    img = raw_image.resize(image_size)
    img = np.array(img)
    # Convert (H, W, C) to (W. H, C)
    img = np.transpose(img, (1, 0, 2))
    img = img.astype(np.float32)
    img = img/255.0
    # Make a batch of 1 image
    test_image = tf.convert_to_tensor(np.array([img]))

    result_mask = model.predict(test_image, verbose=1)
    if type(result_mask) == list:
        result_mask = result_mask[0]
    result_mask = (result_mask > conf_threshold) * 255
    result_mask = np.transpose(result_mask, (0, 2, 1, 3))
    if(len(target_classes) != 3):
        dummy_shape = result_mask.shape
        dummy_shape[-1] = 3 - dummy_shape[1]
        dummy = np.zeros(dummy_shape)
        result_mask = np.concatenate([result_mask, dummy], axis=-1)
    result_mask = Image.fromarray(np.uint8(result_mask[0]))

    result_image = np.array(raw_image.resize(image_size))
    result_image += np.uint8((np.array(result_mask) / 255) * 200)
    result_image = np.clip(result_image, 0, 255)
    result_image = Image.fromarray(result_image)

    gt_mask = []
    for label in target_classes:
        gt_mask.append(np.array(gt_masks[label].resize(image_size)))
    gt_mask = np.stack(gt_mask, axis=-1)
    gt_mask = Image.fromarray(gt_mask)

    gt_image = np.array(raw_image.resize(image_size))
    gt_image += np.uint8((np.array(gt_mask) / 255) * 255)
    gt_image = Image.fromarray(gt_image)

    return raw_image.resize(image_size), result_mask, result_image, gt_mask, gt_image


def infer_image(img_path, image_size, target_classes, model, conf_threshold):
    """
    Inputs: the path to image, (W,H), preprocessing func, keras model object, confidence threshold
    Returns: The original image, the result mask image, combined 1 & 2, combined ground truth
    """
    raw_image = load_image(img_path)
    w, h = raw_image.size
    img = raw_image.resize(image_size)
    img = np.array(img)
    # Convert (H, W, C) to (W. H, C)
    img = np.transpose(img, (1, 0, 2))
    img = img.astype(np.float32)
    img = img/255.0
    # Make a batch of 1 image
    test_image = tf.convert_to_tensor(np.array([img]))

    result_mask = model.predict(test_image, verbose=1)
    if type(result_mask) == list:
        result_mask = result_mask[0]
    result_mask = (result_mask > conf_threshold) * 255
    result_mask = np.transpose(result_mask, (0, 2, 1, 3))
    if(len(target_classes) != 3):
        dummy_shape = result_mask.shape
        dummy_shape[-1] = 3 - dummy_shape[1]
        dummy = np.zeros(dummy_shape)
        result_mask = np.concatenate([result_mask, dummy], axis=-1)
    result_mask = Image.fromarray(np.uint8(result_mask[0]))

    result_image = np.array(raw_image.resize(image_size))
    result_image += np.uint8((np.array(result_mask) / 255) * 200)
    result_image = np.clip(result_image, 0, 255)
    result_image = Image.fromarray(result_image)

    return raw_image.resize(image_size), result_mask, result_image


def draw_boxes(image, bboxes, color, fill=None):
    image_d = ImageDraw.Draw(image)
    for box in bboxes:
        x1, y1, x2, y2 = box
        image_d.rectangle([x1, y1, x2, y2], fill=fill, outline=color, width=2)
    return image


def output2image(result_mask):
    """
    Args: result_mask is image with float values in 0 and 1 (W, H, channels)
    Returns a pil image that can visualize the result_mask
    """
    result_mask_img = result_mask * 255.
    result_mask_img = np.transpose(result_mask_img, (1, 0, 2))
    if result_mask_img.shape[-1] == 4:
        result_mask_img[..., :3] += np.expand_dims(result_mask_img[..., -1], axis=-1) // 1.5
    if result_mask_img.shape[-1] == 5:
        result_mask_img[..., :3] += np.expand_dims(result_mask_img[..., -2], axis=-1) // 1.5  # Brunners gland
        result_mask_img[..., :2] += np.expand_dims(result_mask_img[..., -1], axis=-1)     # Circular crypts
    result_mask_img = np.clip(result_mask_img, 0., 255.)
    result_mask_img = Image.fromarray(np.uint8(result_mask_img[..., :3]))
    return result_mask_img


def histogram_equalization(img):
    R, G, B = cv2.split(img)
    B = cv2.equalizeHist(B.astype(np.uint8))
    G = cv2.equalizeHist(G.astype(np.uint8))
    R = cv2.equalizeHist(R.astype(np.uint8))
    img = cv2.merge((R, G, B))
    return img


class UpdateCRCallback(tf.keras.callbacks.Callback):
    def __init__(self, intensity_var):
        super(UpdateCRCallback, self).__init__()
        self.intensity = intensity_var
        
    def on_epoch_end(self, epoch, logs=None):
        tf.keras.backend.set_value(self.intensity, tf.constant((logs['emb_silhouette'] + 1) / 2))
