import os

import numpy as np

from pycocotools import coco
from PIL import Image
from skimage.transform import resize
from torchvision import transforms


# TODO: here the DataSet parent maybe needed (check that if something will go wrong)
class DataGen:
    def __init__(self, **kwargs):
        """
        :param resize_strategy: The way the images will be transformed before packed into ndarray.
                                Possible values are
                                    stretch: stretch the image directly to desired size.
                                    fit: make the image fit into the desired shape by saving aspect ratio,
                                         non occupied space will be filled with black pixels. The image will be placed
                                         in the middle. [TBA]
                                    put: make the image fit into the desired shape by saving aspect ratio,
                                         non occupied space will be filled with black pixels. The image will be placed
                                         to the top left corner. [TBA]
        """
        self._sets = kwargs
        self._annotation_file = kwargs['annotation_file']
        self._categories = kwargs['categories']
        self._batch_size = kwargs['batch_size']
        self._imgs_dir = kwargs['imgs_dir']
        self._target_img_size = kwargs['target_img_size']
        self._resize_strategy = kwargs['resize_strategy']
        self._channels_format = 'HWC'

        if 'channels_format' in self._sets.keys():
            self._channels_format = self._sets['channels_format']

        self._ann_coco = coco.COCO(annotation_file=self._annotation_file)

        self._cat_ids = self._ann_coco.getCatIds(self._categories)
        self._cat_index_to_id = {k: self._cat_ids[k] for k in range(len(self._cat_ids))}
        self._cat_id_to_index = {v: k for k, v in self._cat_index_to_id.items()}

        self._ann_ids = self._ann_coco.getAnnIds(catIds=self._cat_ids)
        self._anns = self._ann_coco.loadAnns(self._ann_ids)

        ann_ids_for_imgs = dict()
        for item in self._anns:
            img_id = item['image_id']

            if img_id in ann_ids_for_imgs.keys():
                ann_ids_for_imgs[img_id].append(item['id'])
            else:
                ann_ids_for_imgs[img_id] = [item['id']]

        self._ann_ids_for_imgs = ann_ids_for_imgs # maps img_id -> annotations
        self._img_ids = list(self._ann_ids_for_imgs.keys())

        # TODO: figure this out later -> move from here to approach.
        # The approach has to transorm the image into required format
        self.transformer = transforms.Compose([
            transforms.ToTensor(),
        ])

    @property
    def batch_size(self):
        return self._batch_size

    def _mask_tensor_for_img_id(self, img_id):
        curr_img_ann_ids = self._ann_ids_for_imgs[img_id]

        curr_anns = self._ann_coco.loadAnns(ids=curr_img_ann_ids)

        masks_for_class = dict()
        for ann in curr_anns:
            category_id = ann['category_id']

            if category_id not in masks_for_class.keys():
                masks_for_class[category_id] = [ann]
            else:
                masks_for_class[category_id].append(ann)

        masks_original_sizes = dict()
        for k in masks_for_class.keys():
            item = masks_for_class[k]
            masks = [self._ann_coco.annToMask(ann) for ann in item]

            mask = np.logical_or.reduce(masks)
            mask = mask.astype(np.uint8) * 255
            masks_original_sizes[k] = np.sum(mask)
            mask = resize(mask, self._target_img_size, preserve_range=True)
            mask = np.array(mask > 0, np.float32)
            masks_for_class[k] = mask

        tensor_shape = (self._target_img_size[0], self._target_img_size[1], len(self._cat_ids))
        tensor = np.zeros(tensor_shape, np.float32)

        for k, v in masks_for_class.items():
            tensor[:, :, self._cat_id_to_index[k]] = v

        if self._channels_format == 'CHW':
            tensor = np.rollaxis(tensor, 2)

        self._masks_for_class = masks_for_class
        self._masks_original_sizes = masks_original_sizes

        return tensor

    def _img_tensor_for_img_id(self, img_id):
        img = np.array(Image.open(os.path.join(self._imgs_dir, '{:0>12d}.jpg'.format(img_id))).convert("RGB"))
        # TODO: resize is converting the image into range [0, 1]
        img = resize(img, self._target_img_size)

        # TODO: add data preprocessing for such cases
        # img = img / 255.0
        return img.astype(np.float32)

    # TODO: Pytorch part: figure out duplications later
    def __len__(self):
        return len(self._img_ids)

    def __getitem__(self, idx):
        img_id = self._img_ids[idx]
        mask = self._mask_tensor_for_img_id(img_id)
        img = self._img_tensor_for_img_id(img_id)

        img = self.transformer(img)

        return [img, mask]

    def get_summary(self):
        return 'Number of images: {}'.format(len(self._img_ids))

    '''
    def _prepare_masks_for_samples(self):
        self._masks = list()

        for img_id in self._img_ids:
            tensor = self._mask_tensor_for_img_id(img_id)
            self._masks.append(tensor)

        self._masks = np.stack(self._masks)
    def _prepare_imgs_for_samples(self):
        self._imgs = list()

        for img_id in self._img_ids:
            img = np.array(Image.open(os.path.join(self._imgs_dir, '{:0>12d}.jpg'.format(img_id))).convert("RGB"))
            img = resize(img, self._target_img_size)
            self._imgs.append(img)

        self._imgs = np.stack(self._imgs)

    def next_batch(self):
        """
        Generate minibatch for next training iteration. The generation is performed randomly.
        :return: Tuple (imgs, masks)
        """
        self._img_ids = np.random.choice(list(self._ann_ids_for_imgs.keys()), self._batch_size)

        self._prepare_imgs_for_samples()
        self._prepare_masks_for_samples()

        return self._imgs, self._masks
    '''


def make_datagen(**kwargs):
    return DataGen(**kwargs)

