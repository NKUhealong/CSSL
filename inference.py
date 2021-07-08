import os
#os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
# os.environ["CUDA_VISIBLE_DEVICES"]="0"
from data_loader import DataGenerator
from attn_unet import attention_unet_refined
from metrics import *
import sys
import time

from PIL import Image, ImageEnhance
from utils import *
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.metrics import Precision, Recall, MeanIoU
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
    pred_model = attention_unet_refined(test_image_size, 
                                        3, 
                                        len(target_classes), 
                                        multiplier=10, 
                                        freeze_encoder=True, 
                                        freeze_decoder=True, 
                                        use_constraints = False, 
                                        dropout_rate=0.0)
    test_folders = [sys.argv[2]]
    pred_model.load_weights(sys.argv[1], by_name=True)
    predictor = pred_model.predict_on_batch

    if float(sys.argv[3]) >= 0.:
        val = float(sys.argv[3])
        seg_thresholds = [np.array([val] * len(target_classes)).reshape((1, 1, -1))]
    else:
        seg_thresholds = [np.array([val] * len(target_classes)).reshape((1, 1, -1)) for val in np.linspace(0.0, 1.0, 101)]
    box_thresholds = [0.3]
    # box_thresholds = np.linspace(0.0, 1.0, 21)

    for test_folder in test_folders:

        test_generator = DataGenerator(test_folder, None, (320, 256), 'seg', target_classes, 1, model=None, augment=False, S=12)
        files = os.listdir(test_folder)
        file_ids = list(set([f.replace('.jpg', '').replace('.json', '').replace('.xml', '') for f in files]))
        print("Running Anti-Celiac on {} images".format(len(file_ids)))
        seg_dice = []
        seg_prec = []
        seg_rec = []
        IR_dice = []
        IR_prec = []
        IR_rec = []

        for i in tqdm(range(test_generator.__len__()), position=0, leave=True):

            """Fetching instance from the generator"""
            start = time.time()
            [_, inputs], [_, gt_masks] = test_generator.__getitem__(i)
            # print("Fetching", time.time()-start)
            start = time.time()

            """Processing the input"""
            raw_image = np.transpose(inputs[0], (1, 0, 2))
            raw_image = (raw_image*255.0).astype(np.uint8)
            raw_image = Image.fromarray(raw_image)
            pink_mass = raw_image.copy()
            enhancer = ImageEnhance.Contrast(pink_mass)
            pink_mass = enhancer.enhance(4.0)
            pink_mass = np.array(pink_mass)
            pink_mass = 255 - ((pink_mass[:, :, 0] > 150) * (pink_mass[:, :, 1] > 150) * (pink_mass[:, :, 2] > 150)) * 255
            pink_mass = Image.fromarray(pink_mass.astype(np.uint8))

            """Processing the ground truth data"""
            gt_masks = gt_masks[0] * 1.
            gt_masks_img = output2image(gt_masks)
            # gt_boxes, _, _ = decode_boxes(raw_image, gt_grid[0][..., -1], gt_grid[0][..., 0:4], 0.99, scale=16)

            # print("GT", time.time()-start)
            # start = time.time()

            """Making predictions"""
            inputs = tf.convert_to_tensor(inputs)
            result_mask = predictor(inputs)

            seg_dice.append([])
            seg_prec.append([])
            seg_rec.append([])
            for seg_threshold in seg_thresholds:
                """Processing the predictions for segmentation"""
                result_mask_thresh = (result_mask[0] > seg_threshold) * 1.
                # result_mask_thresh = result_mask[0]
                # print(result_mask_thresh)
                result_mask_img = output2image(result_mask_thresh)
                # print("Pred", time.time()-start)
                # extra_preds = result_mask * (1. - gt_masks)
                # extra_preds_img = output2image(extra_preds)

                """Calculating segmentation metrics"""
                seg_dice_score, seg_precision, seg_recall = segmentation_eval(gt_masks, result_mask_thresh)
                seg_dice[-1].append(seg_dice_score)
                seg_prec[-1].append(seg_precision)
                seg_rec[-1].append(seg_recall)            

            IR_dice.append([[0.0]])
            IR_prec.append([[0.0]])
            IR_rec.append([[0.0]])
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

            """Combined visualization"""
            if float(sys.argv[3]) >= 0.:
                divider = Image.new('RGB', (10, result_mask_img.height), (255, 255, 255))
                combined_img = Image.new('RGB', (256*3 + 20, result_mask_img.height))
                combined_img.paste(raw_image.crop((32, 0, 288, 256)), (0, 0))
                combined_img.paste(divider, (256, 0))
                combined_img.paste(result_mask_img.crop((32, 0, 288, 256)), (256 + 10, 0))
                combined_img.paste(divider, (256*2 + 10, 0))
                combined_img.paste(gt_masks_img.crop((32, 0, 288, 256)), (256*2 + 20, 0))
                # combined_img.paste(divider, (256*3 + 20, 0))
                # combined_img.paste(extra_preds_img, (256*3 + 30, 0))
                # display(combined_img)
                combined_img.save('Results/celiac_test60/'+str(i)+'.png')

        seg_dice, seg_prec, seg_rec, IR_dice, IR_prec, IR_rec = np.array(seg_dice), np.array(seg_prec), np.array(seg_rec), np.array(IR_dice), np.array(IR_prec), np.array(IR_rec)
        dice_arr = np.mean(seg_dice, axis=(0, 2))
        seg_idx = np.argmax(dice_arr)
        box_idx = 0
        print("\nResults for", test_folder)
        print("Optimal threshold", seg_thresholds[seg_idx])
        print_array("Dice score", np.mean(seg_dice[:, seg_idx, :], axis=0))
        print_value("Average Dice score", np.mean(np.mean(seg_dice[:, seg_idx, :], axis=0)))
        print_array("Segmentation Precision", np.mean(seg_prec[:, seg_idx, :], axis=0))
        print_value("Average precision", np.mean(np.mean(seg_prec[:, seg_idx, :], axis=0)))
        print_array("Segmentation Recall", np.mean(seg_rec[:, seg_idx, :], axis=0))
        print_value("Average Recall", np.mean(np.mean(seg_rec[:, seg_idx, :], axis=0)))
        print_value("IR Dice", np.mean(IR_dice[:, box_idx]))
        print_value("IR Precision", np.mean(IR_prec[:, box_idx]))
        print_value("IR Recall", np.mean(IR_rec[:, box_idx]))

        """Precision Recall curve for varying threshold"""
        print("Precision-Recall Curves -")
        fig, axs = plt.subplots(nrows=1, ncols=4, sharex='all', sharey='all', figsize=(20, 4))
        axs[0].plot(np.mean(seg_rec, axis=(0))[:, 0], np.mean(seg_prec, axis=(0))[:, 0], 'b:')
        axs[0].set_title("Crypts", color='cyan')
        axs[1].plot(np.mean(seg_rec, axis=(0))[:, 1], np.mean(seg_prec, axis=(0))[:, 1], 'b:')
        axs[1].set_title("Good Villi", color='cyan')
        axs[2].plot(np.mean(seg_rec, axis=(0))[:, 2], np.mean(seg_prec, axis=(0))[:, 2], 'b:')
        axs[2].set_title("Epithelium", color='cyan')
        # axs[3].plot(np.mean(seg_rec, axis=(0))[:, 3], np.mean(seg_prec, axis=(0))[:, 3], 'b:')
        # axs[3].set_title("Brunner's Gland", color='cyan')
        plt.show()

        print('\n')


magic()
