import h5py
import random
import numpy as np
import torch
from torch.utils import data
from utils import centered_resize
import os
from PIL import Image


class Highway_DataSet(data.Dataset):
    """
    class that loads hdf5 dataset object
    """

    def __init__(self,set, root, transform=None):
        """
        Args:
        """
        # self.root = root
        self.root = '../data/projetSession'
        self.transforms = transform
        # load all image files, sorting them to
        # ensure that they are aligned
        # self.imgs = list(sorted(os.listdir(os.path.join(root, "input"))))
        # self.masks = list(sorted(os.listdir(os.path.join(root, "groundtruth"))))

        # self.imgs = [ 'in{:06}.jpg'.format(i) for i in range(1,1701) ]
        # self.masks = ['gt{:06}.png'.format(i) for i in range(1, 1701)]

        # if set =='train':
        #     self.imgs = self.imgs[:int(len(self.imgs)*0.8)]
        #     self.masks = self.masks[:int(len(self.masks)*0.8)]
        # elif set =='test':
        #     self.imgs = self.imgs[int(len(self.imgs)*0.8):]
        #     self.masks = self.masks[int(len(self.masks)*0.8):]

        self.imgs = os.listdir(self.root+'/input2')
        self.masks = os.listdir(self.root+'/groundTruth2')
        self.imgs = sorted(self.imgs)
        self.masks = sorted(self.masks)
        assert np.all([ i[:1] == j[:1] for i,j in zip(self.imgs,self.masks)]) 
        assert len(self.imgs) == len(self.masks)
        if set =='train':
            self.imgs = self.imgs[:int(len(self.imgs)*0.8)]
            self.masks = self.masks[:int(len(self.masks)*0.8)]
        elif set =='test':
            self.imgs = self.imgs[int(len(self.imgs)*0.8):]
            self.masks = self.masks[int(len(self.masks)*0.8):]



    def __getitem__(self, idx):

        """This method loads, transforms and returns slice corresponding to the corresponding index.
        :arg
            index: the index of the slice within patient data
        :return
            A tuple (input, target)

        """
        # load images ad masks
        img_path = os.path.join(self.root, "input", self.imgs[idx])
        mask_path = os.path.join(self.root, "groundtruth", self.masks[idx])
        img_path = os.path.join(self.root, "input2", self.imgs[idx])
        mask_path = os.path.join(self.root, "groundTruth2", self.masks[idx])
        img = Image.open(img_path).convert("RGB")
        # note that we haven't converted the mask to RGB,
        # because each color corresponds to a different instance
        # with 0 being background
        mask = Image.open(mask_path)
        # convert the PIL Image into a numpy array
        mask = np.array(mask)
        # instances are encoded as different colors
        obj_ids = np.unique(mask)
        # first id is the background, so remove it
        obj_ids = obj_ids[1:]
        obj_ids = np.array([50, 170, 255], dtype='uint8')
        
        # split the color-encoded mask into a set
        # of binary masks
        masks = mask == obj_ids[:, None, None]

        masks = torch.as_tensor(masks, dtype=torch.uint8)

        target = masks
        if self.transforms is not None:
            img = self.transforms(img)
            # target = self.transforms(target)
        # print(self.imgs[idx],self.masks[idx])
        return img, np.argmax(masks, axis=0)


    def __len__(self):
        return len(self.imgs)

