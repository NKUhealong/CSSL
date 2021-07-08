from PIL import Image, ImageDraw, ImageEnhance, ImageOps
from tensorflow.keras.utils import Sequence, to_categorical
import os
import json
import random
import numpy as np
import cv2
import skimage
import matplotlib.pyplot as plt
from sklearn.feature_extraction import image as patcher
import time


class DataGenerator(Sequence):
    def __init__(self, folder_path, labelled_path, image_size, target_classes, batch_size=4, model=None, augment=True, S=24, aux_class=False):
        """
        target classes can be a list from Good Crypts / Good Villi / Interpretable Region / Epithelium / Muscularis Mucosa
        mode should be one of 'seg', 'loc' or 'full'
        S is the numbr of total patches to be made from a single image
        """
        print("Initialising data generator")
        # Making the image ids list
        self.model = model
        self.folder_path = folder_path
        self.labelled_path = labelled_path
        self.image_size = image_size
        self.target_classes = target_classes
        image_paths = [f for f in os.listdir(folder_path) if f.endswith(".jpg")]
        if labelled_path is not None:
            image_paths_labelled = [f for f in os.listdir(labelled_path) if f.endswith(".jpg")]
        else:
            image_paths_labelled = []
        image_paths.extend(image_paths_labelled)
        self.image_ids = np.array([f.replace('.jpg', '') for f in image_paths])
        self.is_labelled = np.array([False] * (len(self.image_ids) - len(image_paths_labelled)) + [True] * (len(image_paths_labelled)))
        self.batch_size = batch_size
        self.augment = augment
        self.S = S
        self.aux_class = aux_class
        print("Image count in {} path: {}".format(self.folder_path,len(self.image_ids)))
        self.on_epoch_end()

    def on_epoch_end(self):
        idx = np.arange(len(self.image_ids))
        np.random.shuffle(idx)
        self.image_ids = self.image_ids[idx]
        self.is_labelled = self.is_labelled[idx]

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
        # print(self.is_labelled[index])
        if self.is_labelled[index]:
            image_path = os.path.join(self.labelled_path, self.image_ids[index] + '.jpg')
        else:
            image_path = os.path.join(self.folder_path, self.image_ids[index] + '.jpg')
        img = Image.open(image_path).convert('RGB')
        w, h = img.size
        self.dataset_img_size = (w, h)
        if w != self.image_size[0] and h != self.image_size[1]:
            img = img.resize((self.image_size[0], self.image_size[1]))
        return img

    def load_annotations(self, index):
        anns_file = open(os.path.join(self.labelled_path, self.image_ids[index] + '.json'))
        labels = json.load(anns_file)
        anns = labels["shapes"]
        anns_file.close()
        # Initialize blank masks for each target class
        w, h = self.image_size
        w_img, h_img = self.dataset_img_size
        masks = {label: Image.new("L", (w, h)) for label in self.target_classes}
        draw_masks = {label: ImageDraw.Draw(masks[label]) for label in self.target_classes}
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
        masks = list(masks.values())
        masks = [np.array(m).T / 255. for m in masks]
        masks = np.stack(masks, axis=-1)

        return masks

        
    # GENERATES PATCHES FROM IMAGE GIVEN MASK (POLYGONS) FOR  WHICH PATCHES ARE REQUIRED
    def generate_patches(self, polygons, image, num_patches):
        mask = np.zeros((image.shape[0], image.shape[1], 3))
        image = np.concatenate([image, np.expand_dims(polygons, -1)], axis=-1)
        patches = []
        # Make patches and filter them
        blotches = patcher.extract_patches_2d(image, (64, 64), max_patches=500)
        for i in range(blotches.shape[0]):
            if blotches[i][:, :, -1].sum() > 25:
                patches.append(blotches[i][:, :, :3])
        # Count patches
        if len(patches) < num_patches and len(patches) > 0: # If there are not M patches in every class, randomly augment some patches until there are M patches
            curr_num = len(patches)
            count = curr_num
            while count < num_patches:
                sample_patch_index = np.random.randint(0, curr_num)
                patch_img = Image.fromarray((patches[sample_patch_index]*255.).astype(np.uint8))
                augmented_patch = np.array(self.augment_instance(patch_img))
                patches.append(augmented_patch)
                count += 1
        if len(patches) >= num_patches:
            patches = random.sample(patches, num_patches)
            patches = np.array(patches)
        if len(patches) == 0: # in case there is no pseudolabel coming.
            patches = []
        return patches

    def augment_instance(self, img, flip_hor=None, flip_ver=None, rotate_90=None, brightness_factor=None, contrast_factor=None):
        """
        Args:
            PIL img
        Takes in an image and creates M+1 transformations of it and returns a query image along with its positive key images.
        """
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
        
        # Flip top-bottom
        if flip_ver == 1:
            img = img.transpose(Image.FLIP_TOP_BOTTOM)
        
        # rotate 90 degrees anticlock
        if rotate_90 >= 1:
            img = img.rotate(90, expand = True)
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

            # random brightness and contrast   
            brighten = ImageEnhance.Brightness(img)
            img = brighten.enhance(brightness_factor)
            contrast = ImageEnhance.Contrast(img)
            img = contrast.enhance(contrast_factor)
            
        return img

    def preprocess_instance(self, image):
        """
        Args:
            PIL image
        """
        
        w, h = image.size
        image = np.array(image)
        # Convert (H, W, C) to (W. H, C)
        image = np.transpose(image, (1, 0, 2))
        # img = np.clip(img - np.median(img)+127, 0, 255)
        image = image.astype(np.float32)
        # img = (img - np.min(img)) / (np.max(img) - np.min(img))
        image = image/255.0
        return image

    def get_instance(self, index):
        """
        index is the index of the sample in the main array of indices
        returns the PIL image, a dict of label: masks with bboxes of IRs in format (x, y, w, h) where x, y are top left coords
        """
        # start = time.time()
        # Load the source image and its annotations
        img = self.load_image(index)
        w, h = img.size
        blank = np.zeros((w, h, 3))
        # Perform random augmentations
        if self.augment:
            img = self.augment_instance(img)
        
        real_image = img.copy()
        real_image = np.array(real_image)
        real_image = np.transpose(real_image, (1,0,2))
        real_image = real_image.astype(np.float32)
        real_image = real_image/255.
        
        # Preprocess the image, masks and bboxes
        img = self.preprocess_instance(img)
        img = np.expand_dims(img, 0)
        if self.is_labelled[index]:
            pseudolabels = self.load_annotations(index)
        elif self.model is not None:
            pseudolabels = self.model.predict_on_batch(img)
            pseudolabels = np.squeeze(pseudolabels, 0)
        else:
            pseudolabels = np.zeros(img.shape)[0]
        
        # ADDING AUXILLIARY CLASS IN PSEUDOLABELS
        if self.aux_class:
            compliment_mask = np.clip(np.sum(pseudolabels, axis=-1), 0, 1)
            compliment_mask = np.expand_dims(1 - compliment_mask, -1)
            pseudolabels = np.concatenate(pseudolabels, compliment_mask, axis=-1)
        img = np.squeeze(img, 0)
        pseudolabels = (pseudolabels > 0.01)*1.

        num_valid_masks = 0
        valid_mask = [False] * pseudolabels.shape[-1]
        for ch in range(pseudolabels.shape[-1]):
            if np.sum(pseudolabels[:, :, ch]) > 400:
                num_valid_masks += 1
                valid_mask[ch] = True
        if num_valid_masks == 0:
            return [], [], real_image, pseudolabels

        patches = []
        patch_labels = []
        num_patches = self.S // num_valid_masks
        # Collect masks from each class with other classes masked
        for ch in range(pseudolabels.shape[-1]):
            if valid_mask[ch]:
                complement_mask = pseudolabels.copy()
                complement_mask[:, :, ch] = 0
                complement_mask = np.clip(np.sum(complement_mask, axis=-1), 0, 1)
                complement_mask = 1 - np.expand_dims(complement_mask, axis=-1)
                # Image.fromarray(np.uint8(img * complement_mask * 255)).show()
                tissue_patches = self.generate_patches(pseudolabels[:, :, ch], img * complement_mask, num_patches)
                if tissue_patches is None:
                    # print(ch)
                    Image.fromarray(np.uint8(pseudolabels[:, :, ch] * 255)).show()
                    Image.fromarray(np.uint8(img * complement_mask * 255)).show()
                # print(ch, len(tissue_patches))
                # for tpch in tissue_patches:
                #     Image.fromarray(np.uint8(tpch*255)).show()

                patches.extend(tissue_patches)
                patch_labels.extend([ch] * len(tissue_patches))
        
        patch_labels = list(to_categorical(patch_labels, num_classes=4))
        return patches, patch_labels, real_image, pseudolabels

    def __getitem__(self, index):
        """
        index is the index of batch here
        """
        start = time.time()

        batch_indices = [i for i in range(index*self.batch_size, (index+1)*self.batch_size)]
        batch_indices = [i % len(self.image_ids) for i in batch_indices]
        patches = []
        patch_labels = []
        images = []
        pseudolabels = []

        for ind in batch_indices:
            # istart = time.time()
            pat, plabels, image, pslabels = self.get_instance(ind)
            patches.extend(pat)
            patch_labels.extend(plabels)
            images.append(image)
            pseudolabels.append(pslabels)
       
        patches = np.array(patches)
        patch_labels = np.array(patch_labels)
        images = np.array(images)
        pseudolabels = np.array(pseudolabels)

        return [patches, images], [patch_labels, pseudolabels]


if __name__ == "__main__":
    
    np.random.seed(0)
    random.seed(0)
    debug = True
    train_path = "Path here"
    val_path = "Path here"

    img_size = (320, 256)
    # img_size = (2448, 1920)
    batch_size = 4
    # target_classes = ["Good Crypts"  , "Interpretable Region"]
    target_classes = ["Good Crypts", "Good Villi", "Epithelium", "Brunner's Gland", "Interpretable Region"]
