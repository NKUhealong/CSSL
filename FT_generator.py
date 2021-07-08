from collections import OrderedDict
import json
import numpy as np
import os
import random
import time
from tqdm import tqdm

from PIL import Image, ImageDraw, ImageEnhance
from tensorflow.keras.utils import Sequence
from utils import *

# NOTE: IMAGE CONVENTION IS (W, H, C)

debug = False

class FT_DataGenerator(Sequence):
    def __init__(self, folder_path, image_size=(320,240), batch_size=4, mode='seg', target_classes=["Good Crypts"], filter_classes=[], augment=True):
        """
        target classes can be a list from Good Crypts / Good Villi / Interpretable Region / Epithelium / Muscularis Mucosa
        mode should be one of 'seg', 'loc' or 'full'
        """
        print("Initialising data generator")
        # Making the image ids list
        self.folder_path = folder_path
        image_paths = [f for f in os.listdir(folder_path) if f.endswith(".jpg")]
        self.image_ids = [f.replace('.jpg', '') for f in image_paths]
        self.orig_image_ids = self.image_ids.copy()
        self.filter_classes = filter_classes
        self.filter_data()

        self.image_size = image_size
        self.batch_size = batch_size
        self.mode = mode
        self.target_classes = target_classes
        self.augment = augment
        print("Image count in {} path: {}".format(self.folder_path,len(self.image_ids)))
        self.on_epoch_end()

    def on_epoch_end(self):
        if self.filter_classes != []:
            self.resample_filters()
        random.shuffle(self.image_ids)

    def __len__(self):
        """ Returns the number of batches per epoch """
        gen_len = len(self.image_ids) // self.batch_size
        if len(self.image_ids) % self.batch_size != 0:
            gen_len += 1
        return gen_len

    def load_image(self, index):
        """
        Load an image at the index.
        Returns PIL image
        """
        image_path = os.path.join(self.folder_path, self.image_ids[index] + '.jpg')
        img = Image.open(image_path).convert('RGB')
        if debug:
            print("Loaded image: ", image_path)
        return img

    def load_annotations(self, index):
        """
        Load annotations for an image_index.
        """
        anns_file = open(os.path.join(self.folder_path, self.image_ids[index] + '.json'))
        labels = json.load(anns_file)
        labels = labels["shapes"]
        anns_file.close()
        return labels.copy()

    
    def filter_data(self):
        """
        Keeps only those data instances which contain at least one class in filter_classes
        """
        if(self.filter_classes == []):
            return
        
        filtered_idx = []
        for id in range(len(self.image_ids)):
            anns = self.load_annotations(id)
            found = False
            for ann in anns:
                if ann['label'] in self.filter_classes:
                    found = True
                    break
            if found:
                filtered_idx.append(id)
        
        self.filtered_ids = [self.image_ids[id] for id in filtered_idx]
        # self.image_ids = self.filtered_ids
        print("Number of filtered instances:", len(self.filtered_ids))

    def resample_filters(self):
        """
        Manages the class imbalance. If filtered instances are way less than total data, 
        It will randomly resample from them and append to total instances
        """
        a = len(self.filtered_ids)
        b = len(self.orig_image_ids)
        imbalance_ratio = a / b
        if imbalance_ratio > 0.9:
            return
        num_req = b - 2 * a
        new_ids = ((num_req) // a) * self.filtered_ids
        if num_req % a != 0:
            some_more = random.sample(self.filtered_ids, k=(num_req % a))
            new_ids += some_more
        self.image_ids = self.orig_image_ids + new_ids
        print("Resampled total:", len(self.image_ids))


    def augment_instance(self, img, masks, bboxes, flip_hor=None, flip_ver=None, rotate_90=None, brightness_factor=None, contrast_factor=None):
        """
        Args:
            PIL img
            dict of masks {label: PIL mask}
            bboxes: np array (num_boxes, 4) of format (x1, x2, y1, y2)
        Takes in PIL image and its dict of PIL masks and performs augmentation randomly to return image and new masks
        Returns boxes of (x1, y1, x2, y2)
        """

        def adjust_boxes(boxes_arr):
            for i in range(len(boxes_arr)):
                temp = list(boxes_arr[i])
                boxes_arr[i][0] = min(temp[0], temp[2])
                boxes_arr[i][1] = min(temp[1], temp[3])
                boxes_arr[i][2] = max(temp[0], temp[2])
                boxes_arr[i][3] = max(temp[1], temp[3])

        no_boxes = False
        if len(bboxes) == 0:
            no_boxes = True
            bboxes = np.array([[0., 0., 0., 0.]])  # Dummy box
        assert(bboxes.shape[1] == 4)
        bboxes = np.array(bboxes, np.float)
        
        if flip_hor is None:
            flip_hor = np.random.randint(2)
            # flip_hor = 0
        if flip_ver is None:
            flip_ver = np.random.randint(2)
            # flip_ver = 0
        if rotate_90 is None:
            rotate_90 = np.random.randint(4)
            # rotate_90 = 1
        if brightness_factor is None:
            brightness_factor = 0.2 * random.random() + 0.9
        if contrast_factor is None:
            contrast_factor = 0.2 * random.random() + 0.9

        w, h = img.size

        # Flip left-right
        if flip_hor == 1:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            bboxes[:, 0] = w - bboxes[:, 0]
            bboxes[:, 2] = w - bboxes[:, 2]
            adjust_boxes(bboxes)
            for cl in masks:
                masks[cl] = masks[cl].transpose(Image.FLIP_LEFT_RIGHT)
        
        # Flip top-bottom
        if flip_ver == 1:
            img = img.transpose(Image.FLIP_TOP_BOTTOM)
            bboxes[:, 1] = h - bboxes[:, 1]
            bboxes[:, 3] = h - bboxes[:, 3]
            adjust_boxes(bboxes)
            for cl in masks:
                masks[cl] = masks[cl].transpose(Image.FLIP_TOP_BOTTOM)
        
        # rotate 90 degrees anticlock
        if rotate_90 >= 1:
            img = img.rotate(90, expand = True)
            xmins = np.array(bboxes[:, 0])
            xmaxs = np.array(bboxes[:, 2])
            bboxes[:, 0] = bboxes[:, 1]*1.
            bboxes[:, 2] = bboxes[:, 3]*1.
            bboxes[:, 1] = w - xmaxs
            bboxes[:, 3] = w - xmins
            for cl in masks:
                masks[cl] = masks[cl].rotate(90, expand = True)
            # Now image is in portrait shape, We need landscape window from it
            w_new, h_new = img.size
            w_crop = h
            h_crop = int(h * (w_crop / w))
            left = 0
            right = h
            upper = int(random.random() * (h_new - h))
            lower = upper + h_crop
            rotation_crop = (left, upper, right, lower)
            img = img.crop(rotation_crop)
            img = img.resize((w, h))
            for cl in masks:
                masks[cl] = masks[cl].crop(rotation_crop)
                masks[cl] = masks[cl].resize((w, h))
            # Modifying bboxes to adjust to the cropped image
            bboxes[:, 0] -= left
            bboxes[:, 1] -= upper
            bboxes[:, 2] -= left
            bboxes[:, 3] -= upper
            # Modifying bboxes for the scaled up cropped image
            bboxes[:, 0] *= w / w_crop
            bboxes[:, 1] *= h / h_crop
            bboxes[:, 2] *= w / w_crop
            bboxes[:, 3] *= h / h_crop
            # Sanity check
            bboxes[:, 0] = np.clip(bboxes[:, 0], 0., w)
            bboxes[:, 1] = np.clip(bboxes[:, 1], 0., h)
            bboxes[:, 2] = np.clip(bboxes[:, 2], 0., w)
            bboxes[:, 3] = np.clip(bboxes[:, 3], 0., h)
            adjust_boxes(bboxes)

        # random brightness and contrast   
        brighten = ImageEnhance.Brightness(img)
        img = brighten.enhance(brightness_factor)
        contrast = ImageEnhance.Contrast(img)
        img = contrast.enhance(contrast_factor)

        if no_boxes:
            bboxes = np.array([])
        
        return img, masks, bboxes
            
    def compute_overlap(self, boxes1, boxes2):
        [y_min1, x_min1, y_max1, x_max1] = np.split(boxes1, 4, axis=1)
        [y_min2, x_min2, y_max2, x_max2] = np.split(boxes2, 4, axis=1)

        all_pairs_min_ymax = np.minimum(y_max1, np.transpose(y_max2))
        all_pairs_max_ymin = np.maximum(y_min1, np.transpose(y_min2))
        intersect_heights = np.maximum(
            np.zeros(all_pairs_max_ymin.shape),
            all_pairs_min_ymax - all_pairs_max_ymin)
        all_pairs_min_xmax = np.minimum(x_max1, np.transpose(x_max2))
        all_pairs_max_xmin = np.maximum(x_min1, np.transpose(x_min2))
        intersect_widths = np.maximum(
            np.zeros(all_pairs_max_xmin.shape),
            all_pairs_min_xmax - all_pairs_max_xmin)
        intersect = intersect_heights * intersect_widths

        area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
        area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
        union = np.expand_dims(area1, axis=1) + np.expand_dims(area2, axis=0) - intersect

        return (intersect + 1.) / (union + 1.)

    def bbox_transform(self, anchors, gt_boxes):
        wa = anchors[:, 2] - anchors[:, 0]
        ha = anchors[:, 3] - anchors[:, 1]
        cxa = anchors[:, 0] + wa / 2.
        cya = anchors[:, 1] + ha / 2.

        w = gt_boxes[:, 2] - gt_boxes[:, 0]
        h = gt_boxes[:, 3] - gt_boxes[:, 1]
        cx = gt_boxes[:, 0] + w / 2.
        cy = gt_boxes[:, 1] + h / 2.
        # Avoid NaN in division and log below.
        ha += 1e-7
        wa += 1e-7
        h += 1e-7
        w += 1e-7
        tx = (cx - cxa) / wa
        ty = (cy - cya) / ha
        tw = np.log(w / wa)
        th = np.log(h / ha)
        targets = np.stack([ty, tx, th, tw], axis=1)
        return targets

    def grid_boxes(self, boxes, img_resized):
        """
        Inputs: Boxes procesed to the target image size
        Output: A grid with boxes assigned to it
        """
        scale = 16
        grid_size = (self.image_size[0]//scale, self.image_size[1]//scale)
        box_set = np.zeros((grid_size[0]*grid_size[1], 5))
        if len(boxes) == 0:
            return box_set.reshape((grid_size[0], grid_size[1], 5))

        img_resized = img_resized.resize((int(img_resized.size[0]//scale), int(img_resized.size[1]//scale)))
        enhancer = ImageEnhance.Contrast(img_resized)
        img = enhancer.enhance(4.0)
        img = np.array(img)
        threshold = ((img[:, :, 0] > 150) * (img[:, :, 1] > 150) * (img[:, :, 2] > 150))
        threshold = (1 - 1*threshold).T
        # # Make anchor boxs and compute overlap based indices
        # anchor_boxes = np.zeros((grid_size[0], grid_size[1], 4))
        # ax1 = np.arange(grid_size[0])
        # ay1 = np.arange(grid_size[1])
        # ay1, ax1 = np.meshgrid(ay1, ax1)
        # # print(ax1)
        # # print(ay1)
        # anchor_boxes[:, :, 0] = ax1 * scale
        # anchor_boxes[:, :, 1] = ay1 * scale
        # anchor_boxes[:, :, 2] = ((ax1 + 1) * scale) - 1
        # anchor_boxes[:, :, 3] = ((ay1 + 1) * scale) - 1
        # anchor_boxes = anchor_boxes.reshape((-1, 4))
        # print(anchor_boxes)
        # print(boxes)
        # overlaps = self.compute_overlap(anchor_boxes, boxes)
        # argmax_overlaps_inds = np.argmax(overlaps, axis=1)
        # max_overlaps = overlaps[np.arange(overlaps.shape[0]), argmax_overlaps_inds]
        # print("Overlaps", max_overlaps)
        # print(np.sum(max_overlaps))
        # positive_indices = max_overlaps >= 0.5

        # # Assign the relevant values
        # if debug:
        #     print(positive_indices)
        #     print(np.sum(positive_indices.astype(int)))
        # box_set[positive_indices, -1] = 1.
        # box_set[positive_indices, 0:4] = self.bbox_transform(anchor_boxes, boxes[argmax_overlaps_inds, :])[positive_indices, :]
        
        box_set = box_set.reshape((grid_size[0], grid_size[1], 5))

        for box in boxes:
            x1, y1, x2, y2 = box
            w, h = abs(x2 - x1), abs(y2 - y1)
            img_w, img_h = self.image_size
            cx, cy = x1 + w//2, y1 + h//2

            # i, j = int(cx//scale), int(cy//scale)
            for i in range(int(x1//scale), int(x2//scale)):
                for j in range(int(y1//scale), int(y2//scale)):
                    # box_set[i][j] = np.array([(cx % scale)/scale, (cy % scale)/scale, w/img_w, h/img_h, 1.])
                    # check if the minibox contains some pink mass
                    if threshold[i][j] > 0:
                        box_set[i][j] = np.array([(cx - scale*i), (cy - scale*j), w, h, 1.])

        box_set = np.array(box_set)
        return box_set

    def preprocess_instance(self, image, masks, bboxes):
        """
        Args:
            PIL image
            dict of PIL binary masks
            bboxes (num_boxes, 4) of format (x1, y1, x2, y2)
        """
        w, h = image.size

        # Preprocessing images
        img_resized = image
        # image.show()
        img = np.array(img_resized)
        # Convert (H, W, C) to (W. H, C)
        img = np.transpose(img, (1, 0, 2))
        # img = histogram_equalization(img)
        # img = np.clip(img - np.median(img)+127, 0, 255)
        img = img.astype(np.float32)
        # img = (img - np.min(img)) / (np.max(img) - np.min(img))
        img = img/255.0

        # Preprocssing masks
        msks = OrderedDict(sorted(masks.items()))
        msks = list(masks.values())
        # (W, H, num_classes)
        proc_masks = []
        for mask in msks:
            # mask.show()
            # msk = mask.resize(self.image_size)
            msk = mask
            msk = np.array(msk, dtype=np.uint8)
            msk = np.expand_dims(msk, axis=-1)
            # Convert (H, W, C) to (W, H, C)
            msk = np.transpose(msk, (1, 0, 2))
            msk = msk // 255
            proc_masks.append(msk)
        proc_masks = np.stack(proc_masks, axis=-1)
        proc_masks = np.squeeze(proc_masks, axis=-2)
        # print(proc_masks.shape)
        # temp = Image.fromarray(proc_masks*255, 'RGB')
        # temp.show()

        # Preprocessing boxes
        if len(bboxes) > 0:
            bboxes = np.array(bboxes, np.float)
            wscale = self.image_size[0] / w
            hscale = self.image_size[1] / h
            scale_arr = np.array([wscale, hscale, wscale, hscale])
            bboxes *= scale_arr
            num_boxes = len(bboxes)
            for i in range(num_boxes-1, -1, -1):
                x1, y1, x2, y2 = bboxes[i]
                if x2 <= x1 or y2 <= y1:
                    bboxes = np.delete(bboxes, i, 0)
        box_grid = self.grid_boxes(bboxes, img_resized)

        return img, proc_masks, box_grid

    def points(self, image):
        """
        Takes in image: An image array that contains segmentation mask in a target region
        Returns extern box around the total masked blobs in format xmin, ymin, xmax, ymax
        """
        img = np.squeeze(image)
        activations = img > 0
        activations = activations.astype(int)
        nz = np.nonzero(activations)  # A tuple of arrays indicating non zero element indices per axis
        xmin = np.min(nz[1])
        xmax = np.max(nz[1])
        ymin = np.min(nz[0])
        ymax = np.max(nz[0])
        return xmin, ymin, xmax, ymax

    def get_instance(self, index):
        """
        index is the index of the sample in the main array of indices
        returns the PIL image, a dict of label: masks with bboxes of IRs in format (x, y, w, h) where x, y are top left coords
        """
        # start = time.time()
        # Load the source image and its annotations
        img = self.load_image(index)
        w_img, h_img = img.size
        w, h = self.image_size
        anns = self.load_annotations(index)
        # print("Loading time = {:.5f}s".format(time.time() - start))
        # start = time.time()
        img = img.resize((w, h))

        # Initialize blank masks for each target class
        masks = {label: Image.new("L", (w, h)) for label in self.target_classes}
        draw_masks = {label: ImageDraw.Draw(masks[label]) for label in self.target_classes}
        combined_mask = Image.new("L", (w, h))
        combined_draw_mask = ImageDraw.Draw(combined_mask)

        # Get valid present annotations
        anns = [x for x in anns if type(x) == dict]
        
        # Combine some classes
        for i in range(len(anns)):
            anns[i] = {k.replace('Circular Crypts', 'Good Crypts'): v for k, v in anns[i].items()}

        # Gathering polygon points - A dict of points for each target class
        poly_pts = {label: [] for label in self.target_classes}
        for ann in anns:
            if ann['label'] in self.target_classes:
                pts = np.array(ann['points'])
                pts[:, 0] = pts[:, 0] * (w*1. / w_img)
                pts[:, 1] = pts[:, 1] * (h*1. / h_img)
                poly_pts[ann['label']].append(pts)

        # Drawing the masks
        for label in self.target_classes:
            if label != "Interpretable Region":
                polygons = poly_pts[label]
                for poly in polygons:
                    coords = [(pt[0], pt[1]) for pt in poly]
                    draw_masks[label].polygon(xy=coords, fill=255)
                    combined_draw_mask.polygon(coords, 255)
        
        # Making the Interpretable Region mask
        if (self.mode == 'seg' or self.mode == 'full') and 'Interpretable Region' in self.target_classes:
            pink_mass = img.copy()
            enhancer = ImageEnhance.Contrast(pink_mass)
            pink_mass = enhancer.enhance(4.0)
            pink_mass = np.array(pink_mass)
            pink_mass = 255 - ((pink_mass[:, :, 0] > 150) * (pink_mass[:, :, 1] > 150) * (pink_mass[:, :, 2] > 150)) * 255
            IRs = poly_pts["Interpretable Region"]
            IR_image = Image.new('L', (w, h))
            IR_draw_image = ImageDraw.Draw(IR_image)
            for i in range(len(IRs)):
                IR = IRs[i]
                IR = [(p[0], p[1]) for p in IR]
                IR_draw_image.polygon(IR, 255)
            pink_mass *= np.array(IR_image) // 255
            pink_mass = Image.fromarray(pink_mass.astype(np.uint8))
            masks['Interpretable Region'] = pink_mass

        # Gathering IR boxes in format x, y, w, h where x, y are top left coords
        bboxes = []
        if (self.mode == 'loc' or self.mode == 'full') and "Interpretable Region" in self.target_classes:
            IRs = poly_pts["Interpretable Region"]
            for i in range(len(IRs)):
                IR = IRs[i]
                IR = [(p[0], p[1]) for p in IR]
                IR_image = Image.new('L', (w, h))
                IR_draw_image = ImageDraw.Draw(IR_image)
                IR_draw_image.polygon(IR, 255)
                mul_img_arr = (np.array(IR_image)/255.) * (np.array(combined_mask)/255.)
                mul_img_arr = np.array(mul_img_arr*255, np.uint8)
                xmin, ymin, xmax, ymax = self.points(mul_img_arr)
                bboxes.append([xmin, ymin, xmax, ymax])
        bboxes = np.array(bboxes)

        # print("Parsing time = {:.5f}s".format(time.time() - start))
        # start = time.time()

        # Visualizing the final data sample mask
        if debug:
            img.show()
            # debug_image = Image.new('RGB', img.size)
            debug_image = np.zeros((img.size[1], img.size[0], 3), np.uint8)
            debug_image[:, :, 0] = np.array(masks['Good Crypts'])
            debug_image[:, :, 1] = np.array(masks['Good Villi'])
            debug_image[:, :, 2] = np.array(masks['Epithelium'])
            if "Brunner's Gland" in masks:
                debug_image += np.expand_dims((np.array(masks["Brunner's Gland"])//2), -1)
            if "Interpretable Region" in masks:
                debug_image[:, :, :2] += np.expand_dims((np.array(masks["Interpretable Region"])), -1)
            debug_image = Image.fromarray(debug_image)
            debug_draw = ImageDraw.Draw(debug_image)
            for IR in poly_pts['Interpretable Region']:
                IR = [(p[0], p[1]) for p in IR]
                debug_draw.polygon(IR, fill=None, outline="#ffffff")
            for bbox in bboxes:
                print(bbox)
                debug_draw.rectangle([bbox[0], bbox[1], bbox[2], bbox[3]], outline="#ff00ff", width=4)
            debug_image.show()

        # Perform random augmentations
        if self.augment:
            img, masks, bboxes = self.augment_instance(img, masks, bboxes)

        # Visualizing the final data sample mask
        if debug:
            # debug_image = Image.new('RGB', img.size)
            debug_image = np.zeros((img.size[1], img.size[0], 3), np.uint8)
            debug_image[:, :, 0] = np.array(masks['Good Crypts'])
            debug_image[:, :, 1] = np.array(masks['Good Villi'])
            debug_image[:, :, 2] = np.array(masks['Epithelium'])
            if "Brunner's Gland" in masks:
                debug_image += np.expand_dims((np.array(masks["Brunner's Gland"])//2), -1)
            if "Interpretable Region" in masks:
                debug_image[:, :, :2] += np.expand_dims((np.array(masks["Interpretable Region"])), -1)
            debug_image = Image.fromarray(debug_image)
            debug_draw = ImageDraw.Draw(debug_image)
            for bbox in bboxes:
                print(bbox)
                debug_draw.rectangle([bbox[0], bbox[1], bbox[2], bbox[3]], outline="#ff00ff", width=4)
            debug_image.show()

        # Preprocess the image, masks and bboxes
        img, masks, box_grid = self.preprocess_instance(img, masks, bboxes)
        
        # print("Aug and preprocess time = {:.5f}s".format(time.time() - start))
        # start = time.time()

        return img, masks, box_grid

    def getitem(self, index):
        """
        index is the index of batch here
        """
        # start = time.time()

        batch_indices = [i for i in range(index*self.batch_size, (index+1)*self.batch_size)]
        batch_indices = [i % len(self.image_ids) for i in batch_indices]
        input_imgs = []
        input_masks = []
        target_masks = []
        target_box_grids = []
        for ind in batch_indices:
            # istart = time.time()
            img, masks, box_grid = self.get_instance(ind)
            # print("Instance time = {:.5f}s".format(time.time() - istart))
            if debug:
                print(np.sum(box_grid[:, :, 0]))
            input_imgs.append(img)
            input_masks.append(masks)
            target_masks.append(masks)
            target_box_grids.append(box_grid)  # x in cell, y in cell, w//scale, h//scale
        
        input_imgs = np.array(input_imgs)   # (B, w, h, 3)
        input_masks = np.array(input_masks)  # (B, w, h, channels)
        target_masks = np.array(target_masks)  # (B, w, h, channels)
        target_box_grids = np.array(target_box_grids)   # (B, w/scale, h/scale, 5)
        
        if self.mode == 'seg':
            inputs = input_imgs
            targets = target_masks
        elif self.mode == 'loc':
            inputs = [input_imgs, input_masks]
            targets = target_box_grids
        elif self.mode == 'full':
            inputs = [input_imgs, input_masks]
            targets = [target_masks, target_box_grids]
        else:
            raise ValueError("Invalid mode given: Options are 'seg'/'loc'/'full'")

        # print("Batch generation time = {:.5f}s".format(time.time() - start))
        return inputs, targets


    def __getitem__(self, index):
        return self.getitem(index)
        # return self.instances[index]



if __name__ == "__main__":
    
    np.random.seed(0)
    random.seed(0)
    debug = True
    # train_path = "/media/nisarg/DATA/Master/Study/Semester7/BTP1/Magic/Data/Train"
    # val_path = "/media/nisarg/DATA/Master/Study/Semester7/BTP1/Magic/Data/Valid"
    train_path = "D:/Master/Study/Semester7/BTP1/Data/Train"
    val_path = "D:/Master/Study/Semester7/BTP1/Data/Valid"

    img_size = (640, 512)
    # img_size = (2448, 1920)
    batch_size = 3
    # target_classes = ["Good Crypts"  , "Interpretable Region"]
    target_classes = ["Good Crypts", "Good Villi", "Epithelium", "Brunner's Gland", "Interpretable Region"]

    train_gen = DataGenerator(train_path, img_size, batch_size, mode='seg', target_classes=target_classes
                            , filter_classes=["Brunner's Gland"], augment=True)
    # val_gen = DataGenerator(val_path, img_size, batch_size, target_classes, augment=False)
    inps, tgts = train_gen.__getitem__(1)
    # print(inps)
    # print(inps.shape)
    # print(tgts)
    # print(tgts[0].shape)
    # print(tgts[1].shape)
