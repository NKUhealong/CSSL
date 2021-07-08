import os
#os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
# os.environ["CUDA_VISIBLE_DEVICES"]="0"
from data_generator import DataGenerator
from model import attention_unet_resnet50, attention_unet_refined
from metrics import *
import sys
import time

from PIL import Image, ImageEnhance
from utils import *
import numpy as np
import tensorflow as tf
from tqdm import tqdm


def segmentation_eval(gt_mask, pred_mask):
    """(W, H, ch) 0/1 values"""
    pred = np.array(pred_mask)
    gt = np.array(gt_mask)
    
    smooth = 1e-5
    intersection = np.sum(np.abs(pred*gt), axis=(0,1))
    union = np.sum(gt,(0,1)) + np.sum(pred,(0,1)) - intersection
    dice = (2*intersection+smooth)/(union+intersection+smooth)
    precision = (intersection + smooth) / (np.sum(pred, (0,1)) + smooth)
    recall = (intersection + smooth) / (np.sum(gt, (0,1)) + smooth)
    return dice, precision, recall


def IR_eval(y_true, y_pred, domain_mask):
    """
    y is the list of boxes (x1, y1, x2, y2)
    domain_mask is a binary mask of regions to consider in it (PIL object)
    """
    gt_mask = draw_boxes(Image.new('L', domain_mask.size), y_true, "#ffffff", fill="#ffffff")
    pred_mask = draw_boxes(Image.new('L', domain_mask.size), y_pred, "#ffffff", fill="#ffffff")
    gt_mask = (np.array(gt_mask) // 255.) * (np.array(domain_mask) // 255.)
    pred_mask = (np.array(pred_mask) // 255.) * (np.array(domain_mask) // 255.)
    smooth = 1e-5
    intersection = np.sum(gt_mask * pred_mask)
    union = np.sum(np.clip(gt_mask+pred_mask, 0, 1))

    ir_dice = (2*intersection + smooth) / (union + intersection + smooth)
    ir_precision = (intersection + smooth) / (np.sum(pred_mask) + smooth)
    ir_recall = (intersection + smooth) / (np.sum(gt_mask) + smooth)
    F1 = (2*ir_precision*ir_recall+smooth)/(ir_precision+ir_recall+smooth)
    # print(F1, ir_dice)
    return ir_dice, ir_precision, ir_recall


def prepare_input(img_path, img_size):
    img = Image.open(img_path).convert('RGB')
    w, h = img_size
    # Preprocessing images
    img_resized = img.resize(img_size)
    img = np.array(img_resized)
    img = np.transpose(img, (1, 0, 2))
    img = np.array([img], np.float32)
    img = img/255.0
    return img


def make_pseudolabel(thresh_mask):
        """
        Takes in the threholded predictions
        and gives out pil image pseudolabel
        """
        w, h, num_classes = thresh_mask.shape
        plabels = np.ones((w, h), dtype=np.uint8) * 255
        for cls in range(num_classes):
            plabels[thresh_mask[:, :, cls] == 1] = cls
        plabels = np.transpose(plabels, (1, 0))
        plabels = Image.fromarray(np.uint8(plabels))
        return plabels


def magic():
    
    def print_array(title, arr):
        print(title + ": ", end='')
        print('[', end='')
        for elem in arr:
            print("{:.3f}".format(elem), end='\t')
        print(']')

    def print_value(title, fval):
        print(title + ": ", end='')
        print("{:.3f}".format(fval))

    def decode_boxes(image, labels, regression, conf_thres, scale=16):
        img_w, img_h = labels.shape[-2]*scale, labels.shape[-1]*scale

        i = np.arange(labels.shape[0])
        j = np.arange(labels.shape[1])
        i, j = np.meshgrid(i, j, indexing='ij')

        cx, cy, bw, bh = regression[..., 0], regression[..., 1], regression[..., 2], regression[..., 3]
        cx += scale * i
        cy += scale * j
        
        x1 = np.clip(cx-bw/2, 0, img_w)
        x2 = np.clip(cx+bw/2, 0, img_w)
        y1 = np.clip(cy-bh/2, 0, img_h)
        y2 = np.clip(cy+bh/2, 0, img_h)

        labels = labels.reshape((-1, 1))
        regression = regression.reshape((-1, 4))
        assert(labels.shape[0] == regression.shape[0])

        valid_preds = labels[..., 0] >= conf_thres
        confs = labels[valid_preds]
        
        bboxes = np.stack([x1, y1, x2, y2], axis=-1)
        bboxes = bboxes.reshape((-1, 4))
        bboxes = bboxes[valid_preds]

        miniboxes = np.stack([i*scale, j*scale, (i+1)*scale - 1, (j+1)*scale - 1], axis=-1)
        miniboxes = miniboxes.reshape((-1, 4))
        miniboxes = miniboxes[labels.reshape(-1) >= conf_thres, :]
        return np.array(bboxes), np.array(miniboxes), np.array(confs)


    def bottom_up_grouping(bboxes_in, iou_thres):
        num_boxes = len(bboxes_in)
        if num_boxes == 0:
            return bboxes_in
        bboxes = np.array(bboxes_in)
        done = False
        while not done:
            iou_matrix = test_generator.compute_overlap(bboxes, bboxes)
            np.fill_diagonal(iou_matrix, 0.)
            ind = np.argmax(iou_matrix)
            i, j = ind//num_boxes, ind%num_boxes
            if iou_matrix[i][j] < iou_thres:
                done = True
                break
            # print("boxes ", bboxes[i], " and ", bboxes[j], " selected with overlap ", iou_matrix[i][j])
            x11, y11, x12, y12 = bboxes[i]
            x21, y21, x22, y22 = bboxes[j]
            # x1, y1, x2, y2 = min(x11, x21), min(y11, y21), max(x12, x22), max(y12, y22)
            x1, y1, x2, y2 = (x11 + x21)/2, (y11 + y21)/2, (x12 + x22)/2, (y12 + y22)/2
            bboxes = np.delete(bboxes, [i, j], 0)
            bboxes = np.append(bboxes, np.array([[x1, y1, x2, y2]]), axis=0)
            num_boxes -= 1
            # print("Added box ", np.array([[x1, x2, y1, y2]]))
        return bboxes


    def get_absurd_boxes(bboxes, mask):
        m1, m2, m3 = np.array(mask)[:, :, 0].T, np.array(mask)[:, :, 1].T, np.array(mask)[:, :, 2].T
        absurds = []
        for box in bboxes:
            x1, y1, x2, y2 = box.astype(int)
            if np.sum(m1[x1:x2, y1:y2]) == 0 or np.sum(m2[x1:x2, y1:y2]) == 0 or np.sum(m3[x1:x2, y1:y2]) == 0:
                absurds.append([x1, y1, x2, y2])
        return np.array(absurds)


    target_classes = ["Good Crypts", "Good Villi", "Epithelium", "Brunner's Gland"]
    test_image_size = (320, 256)
    # pred_model = attention_unet_resnet50(input_shape=(test_image_size[0], test_image_size[1], 3), 
    #                                     out_channels=len(target_classes), 
    #                                     freeze_encoder=True, 
    #                                     encoder_weights=None, 
    #                                     freeze_decoder=True, 
    #                                     dropout_rate=0.0)
    _, pred_model, _, _, _ = attention_unet_refined(test_image_size, 
                                                3, 
                                                len(target_classes), 
                                                multiplier=10, 
                                                freeze_encoder=True, 
                                                freeze_decoder=True, 
                                                use_constraints = False, 
                                                dropout_rate=0.0)
    pred_model.load_weights(sys.argv[1], by_name=True)
    test_folders = [sys.argv[2]]
    predictor = pred_model.predict_on_batch

    if float(sys.argv[3]) >= 0.:
        val = float(sys.argv[3])
        seg_thresholds = [np.array([val] * len(target_classes)).reshape((1, 1, -1))]
    else:
        seg_thresholds = [np.array([val] * len(target_classes)).reshape((1, 1, -1)) for val in np.linspace(0.0, 1.0, 101)]
    box_thresholds = [0.3]
    # box_thresholds = np.linspace(0.0, 1.0, 21)

    for test_folder in test_folders:

        # test_generator = DataGenerator(test_folder, test_image_size, 1, 'full', target_classes, augment=False)
        files = list(sorted(os.listdir(test_folder)))
        file_ids = list(set([f.replace('.jpg', '').replace('.json', '').replace('.xml', '') for f in files if 'pseudolabel' not in f]))
        images = [os.path.join(test_folder, fid+'.jpg') for fid in file_ids]
        print("Running Anti-Celiac on {} images".format(len(file_ids)))
        seg_dice = []
        seg_prec = []
        seg_rec = []
        IR_dice = []
        IR_prec = []
        IR_rec = []

        for i in tqdm(range(len(images)), position=0, leave=True):

            """Fetching instance from the generator"""
            start = time.time()
            input = prepare_input(images[i], test_image_size)
            img_id = file_ids[i]

            """Making predictions"""
            input = tf.convert_to_tensor(input)
            result_mask = predictor(input)

            for seg_threshold in seg_thresholds:
                """Processing the predictions for segmentation"""
                result_mask_thresh = (result_mask[0] > seg_threshold) * 1.
                # result_mask_thresh = result_mask[0]
                # print(result_mask_thresh)
                # result_mask_img = output2image(result_mask_thresh)
                # print("Pred", time.time()-start)
                # extra_preds = result_mask * (1. - gt_masks)
                # extra_preds_img = output2image(extra_preds)

            """Saving the thresholded pseudolabels - segmentation masks"""
            plabels = make_pseudolabel(result_mask_thresh)
            plabels.save(os.path.join(test_folder, img_id+'_pseudolabels.png'))

            """Making box predictions"""
            # IR_dice.append([])
            # IR_prec.append([])
            # IR_rec.append([])
            # for box_threshold in box_thresholds:
            #     """Processing prediction for box localization"""
            #     result_boxes, temp, result_confs = decode_boxes(raw_image, grid[0][..., -1], grid[0][..., 0:4], box_threshold, scale=16)
            #     # result_absurds = get_absurd_boxes(result_boxes, result_mask_img)
            #     # start = time.time()
            #     result_boxes = bottom_up_grouping(result_boxes, 0.2)
            #     raw_image = draw_boxes(raw_image, result_boxes, "#0000ff")
            #     raw_image = draw_boxes(raw_image, gt_boxes, "#00ff00")
            #     # raw_image = draw_boxes(raw_image, temp, "#ff0000")
            #     # raw_image = draw_boxes(raw_image, result_absurds, "#ff0000")

            #     """Calculating box localization metrics"""
            #     IR_dice_score, IR_prec_score, IR_rec_score = IR_eval(gt_boxes, result_boxes, pink_mass)
            #     IR_dice[-1].append(IR_dice_score)
            #     IR_prec[-1].append(IR_prec_score)
            #     IR_rec[-1].append(IR_rec_score)

magic()
