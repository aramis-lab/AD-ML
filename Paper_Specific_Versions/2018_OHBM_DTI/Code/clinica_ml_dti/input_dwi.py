# -*- coding: utf-8 -*-
__author__ = ["Junhao Wen", "Jorge Samper-Gonzalez"]
__copyright__ = "Copyright 2016-2018 The Aramis Lab Team"
__credits__ = ["Nipype"]
__license__ = "See LICENSE.txt file"
__version__ = "0.1.0"
__status__ = "Development"

import os.path as path
import nibabel as nib

import numpy as np
from pandas.io import parsers
from clinica.pipelines.machine_learning import voxel_based_io as vbio

import clinica.pipelines.machine_learning.region_based_io as rbio
from clinica.pipelines.machine_learning.input import CAPSInput


class DWIRegionInput(CAPSInput):

    def __init__(self, caps_directory, subjects_visits_tsv, diagnoses_tsv, atlas, dwi_map,
                 precomputed_kernel=None):
        """

        Args:
            caps_directory:
            subjects_visits_tsv:
            diagnoses_tsv:
            atlas:
            dwi_map:
            precomputed_kernel:
        """
        super(DWIRegionInput, self).__init__(caps_directory, subjects_visits_tsv, diagnoses_tsv, None,
                                       image_type='dwi', precomputed_kernel=precomputed_kernel) ## TODO super from parent class, need to add image_type for DTI

        self._atlas = atlas
        self._dwi_map = dwi_map
        self._orig_shape = None
        self._data_mask = None

        if atlas not in ['JHUDTI81', 'JHUTracts0', 'JHUTracts25']:
            raise Exception("Incorrect atlas name. It must be one of the values 'JHUDTI81', 'JHUTracts0', 'JHUTracts25'")
        if dwi_map not in ['fa', 'md', 'rd', 'ad']:
            raise Exception("Incorrect DWI map name. It must be one of the values 'fa', 'md', 'rd', 'ad'")

    def get_images(self):
        """

        Returns: a list of filenames

        """
        if self._images is not None:
            return self._images

        if self._dwi_map == 'fa':
            self._images = [path.join(self._caps_directory, 'subjects', self._subjects[i], self._sessions[i],
                                      'dwi/atlas_statistics', 'SyN_QuickWarped_thresh_space-%s_map-%s_statistics.tsv'
                                      # % (self._subjects[i], self._sessions[i], self._atlas))
                                      % (self._atlas, self._dwi_map))
                            for i in range(len(self._subjects))]
        else:
            self._images = [path.join(self._caps_directory, 'subjects', self._subjects[i], self._sessions[i],
                                      'dwi/atlas_statistics', 'space-JHUTracts0_%s_thresh_space-%s_map-%s_statistics.tsv'
                                      # % (self._subjects[i], self._sessions[i], self._atlas))
                                      % (self._dwi_map, self._atlas, self._dwi_map))
                            for i in range(len(self._subjects))]

        for image in self._images:
            if not path.exists(image):
                raise Exception("File %s doesn't exists." % image)

        return self._images

    def get_x(self):
        """

        Returns: a numpy 2d-array.

        """
        if self._x is not None:
            return self._x

        print 'Loading ' + str(len(self.get_images())) + ' subjects'
        x = rbio.load_data(self._images, self._subjects) #### get all the data in tsv into a data frame, lines are subjects, columns are ROI measures
        self._x = x[:, 1:] # delete the first column
        print 'Subjects loaded'

        return self._x

    def save_weights_as_nifti(self, weights, output_dir):
        """

        Args:
            weights:
            output_dir:

        Returns:

        """

        # w = np.zeros(shape=(weights.shape[0], 1))
        w = np.insert(weights, 0, 0, axis=0) ## add 0 as weight for unknow voxels
        output_filename = path.join(output_dir, 'weights.nii.gz')
        # normalize the weights infinite normalization
        w = w / abs(w).max()

        def weights_to_nifti(weights, atlas, output_filename):
            """

            Args:
                atlas:
                weights:
                output_filename:

            Returns:

            """

            import numpy as np
            import nibabel as nib
            from clinica.utils.atlas import AtlasAbstract

            atlas_path = None
            atlas_classes = AtlasAbstract.__subclasses__()
            for atlas_class in atlas_classes:
                if atlas_class.get_name_atlas() == atlas:
                    atlas_path = atlas_class.get_atlas_labels()

            if not atlas_path:
                raise ValueError('Atlas path not found for atlas name ' + atlas)

            atlas_image = nib.load(atlas_path)
            atlas_data = atlas_image.get_data()
            labels = list(set(atlas_data.ravel()))
            output_image_weights = np.zeros(shape=atlas_data.shape, dtype='f')

            for i, n in enumerate(labels):
                index = np.array(np.where(atlas_data == n))
                output_image_weights[index[0, :], index[1, :], index[2, :]] = weights[i]

            affine = atlas_image.get_affine()
            output_image = nib.Nifti1Image(output_image_weights, affine)
            nib.save(output_image, output_filename)

        weights_to_nifti(w, self._atlas, output_filename)


class DWIVoxelBasedInput(CAPSInput):
    def __init__(self, caps_directory, subjects_visits_tsv, diagnoses_tsv, dwi_map, tissue_type, threshold, fwhm=None, mask_zeros=True,
                 precomputed_kernel=None):
        """
        This is a class to grab the outputs from CAPS for DWIVoxel based analysis
        :param caps_directory:
        :param subjects_visits_tsv:
        :param diagnoses_tsv:
        :param dwi_map: should be one of 'fa', 'md', 'rd' and 'ad'
        :param fwhm: the smoothing kernel in mm
        :param tissue_type: should be one of 'GM', 'GM_WM' and 'WM'
        :param threshold: the threshold of the mask
        :param mask_zeros:
        :param precomputed_kernel:
        """
        super(DWIVoxelBasedInput, self).__init__(caps_directory, subjects_visits_tsv, diagnoses_tsv, None,
                                                 image_type='dwi', precomputed_kernel=precomputed_kernel) ## TODO super from parent class, need to add image_type for DTI

        self._mask_zeros = mask_zeros
        self._orig_shape = None
        self._data_mask = None
        self._dwi_map = dwi_map
        self._tissue_type = tissue_type
        self._threshold = threshold
        self._fwhm = fwhm

        if dwi_map not in ['fa', 'md', 'rd', 'ad']:
            raise Exception("Incorrect DWI map name. It must be one of the values 'fa', 'md', 'rd', 'ad'")

        if tissue_type not in ['GM', 'GM_WM', 'WM']:
            raise Exception("Incorrect tissue type. It must be one of the values 'GM', 'GM_WM', 'WM'")

    def get_images(self):
        """

        Returns: a list of filenames

        """
        if self._images is not None:
            return self._images

        ### to grab the masked image
        if self._fwhm == None:
            self._images = [path.join(self._caps_directory, 'subjects', self._subjects[i], self._sessions[i],
                                      'dwi/normalized_space', '%s_non_smoothed_%s_masked_threshold-%s.nii.gz'
                                      % (self._tissue_type, self._dwi_map, str(self._threshold)))
                            for i in range(len(self._subjects))]
        else:
            self._images = [path.join(self._caps_directory, 'subjects', self._subjects[i], self._sessions[i],
                                      'dwi/normalized_space', '%s_fwhm-%smm_%s_masked_threshold-%s.nii.gz'
                                      % (self._tissue_type, str(self._fwhm), self._dwi_map, str(self._threshold)))
                            for i in range(len(self._subjects))]


        for image in self._images:
            if not path.exists(image):
                raise Exception("File %s doesn't exists." % image)

        return self._images

    def get_x(self):
        """

        Returns: a numpy 2d-array.

        """
        if self._x is not None:
            return self._x

        print 'Loading ' + str(len(self.get_images())) + ' subjects'
        self._x, self._orig_shape, self._data_mask = vbio.load_data(self._images, mask=self._mask_zeros)
        print 'Subjects loaded'

        return self._x

    def save_weights_as_nifti(self, weights, output_dir):

        if self._images is None:
            self.get_images()

        output_filename = path.join(output_dir, 'weights.nii.gz')
        data = vbio.revert_mask(weights, self._data_mask, self._orig_shape)

        features = data / abs(data).max()

        img = nib.load(self._images[0])

        output_image = nib.Nifti1Image(features, img.affine)

        nib.save(output_image, output_filename)
