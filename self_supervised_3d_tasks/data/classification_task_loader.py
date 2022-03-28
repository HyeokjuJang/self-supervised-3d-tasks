from pathlib import Path

import numpy as np
from scipy import ndimage

from self_supervised_3d_tasks.data.generator_base import DataGeneratorBase
import os

class ClassificationGenerator3D(DataGeneratorBase):
    def __init__(
            self,
            data_path,
            file_list,
            batch_size=8,
            patch_size=(128, 128, 128),
            patches_per_scan=3,
            pre_proc_func=None,
            shuffle=False,
            augment=False,
            label_stem="_label"
    ):
        self.augment_scans_train = augment

        self.label_stem = label_stem
        self.label_dir = data_path
        self.data_dir = data_path
        self.patch_size = patch_size
        self.patches_per_scan = patches_per_scan

        label_dir_list = os.listdir(self.data_dir)
        self.data_xy = {}
        self.label_xy = {}
        label_count = 0
        npy_file_list = []
        for label_dir in label_dir_list:
            file_names = os.listdir(os.path.join(self.data_dir, label_dir))
            for file_name in file_names:
                self.data_xy[os.path.join(label_dir, file_name)] = label_count
                npy_file_list.append(os.path.join(label_dir, file_name))
            self.label_xy[label_count] = os.path.join(self.data_dir, label_dir)
            label_count += 1

        super(ClassificationGenerator3D, self).__init__(npy_file_list, batch_size, shuffle, pre_proc_func)

    def load_image(self, index):
        file_name = self.input_images[index]
        path = "{}/{}".format(self.data_dir, file_name)
        # @TODO need to change this

        path_label = "{}/{}".format(self.label_dir, file_name)

        return np.load(path), np.load(path_label)

    def augment_3d(self, x, y):
        def _distort_color(scan):
            # adjust brightness
            max_delta = 0.125
            delta = np.random.uniform(-max_delta, max_delta)
            scan += delta
            # adjust contrast
            lower = 0.5
            upper = 1.5
            contrast_factor = np.random.uniform(lower, upper)
            scan_mean = np.mean(scan)
            scan = (contrast_factor * (scan - scan_mean)) + scan_mean
            return scan

        processed_image, processed_mask = x.copy(), y.copy()
        for i in range(3):  # arbitrary flipping along each axis
            if np.random.rand() < 0.5:
                processed_image = np.flip(processed_image, axis=i)
                processed_mask = np.flip(processed_mask, axis=i)
        # make rotation arbitrary -5 to 5
        if np.random.rand() < 0.5:
            axis_choice = np.random.randint(0, 3)
            if axis_choice == 0:
                xy_angle = np.random.uniform(-5, 5)
                processed_image = ndimage.rotate(processed_image, xy_angle, axes=(0, 1), reshape=False, order=1)
                processed_mask = ndimage.rotate(processed_mask, xy_angle, axes=(0, 1), reshape=False, order=0)
            elif axis_choice == 1:
                yz_angle = np.random.uniform(-5, 5)
                processed_image = ndimage.rotate(processed_image, yz_angle, axes=(1, 2), reshape=False, order=1)
                processed_mask = ndimage.rotate(processed_mask, yz_angle, axes=(1, 2), reshape=False, order=0)
            else:
                xz_angle = np.random.uniform(-5, 5)
                processed_image = ndimage.rotate(processed_image, xz_angle, axes=(0, 2), reshape=False, order=1)
                processed_mask = ndimage.rotate(processed_mask, xz_angle, axes=(0, 2), reshape=False, order=0)
        if np.random.rand() < 0.7:
            # color distortion (THIS DOESN'T CHANGE IN THE MASK)
            processed_image = _distort_color(processed_image)

        return processed_image, processed_mask

    def data_generation(self, list_files_temp):
        data_x = []
        data_y = []

        for file_name in list_files_temp:
            path = os.path.join(self.data_dir, file_name)
            img = np.load(path)
            img = (img - img.min()) / (img.max() - img.min())
            data_x.append(img)
            data_y.append(self.data_xy[file_name])

        data_x = np.stack(data_x)
        data_y = np.stack(data_y)

        data_y = np.rint(data_y).astype(np.int)
        n_classes = len(self.label_xy)
        data_y = np.eye(n_classes)[data_y]
        # if data_y.shape[-2] == 1:
        #     data_y = np.squeeze(data_y, axis=-2)  # remove second last axis, which is still 1

        return data_x, data_y
