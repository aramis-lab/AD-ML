# -*- coding: utf-8 -*-
__author__ = ["Junhao Wen"]
__copyright__ = "Copyright 2016-2018 The Aramis Lab Team"
__credits__ = ["Junhao Wen"]
__license__ = "See LICENSE.txt file"
__version__ = "0.1.0"
__status__ = "Development"

def create_binary_mask(tissues, threshold=0.3):
    """

    Args:
        tissues:
        threshold:

    Returns:

    """
    import nibabel as nib
    import numpy as np
    from os import getcwd
    from os.path import join, basename

    if len(tissues) == 0:
        raise RuntimeError('The length of the list of tissues must be greater than zero.')
    if isinstance(tissues, basestring):  # just one compartment
        img_0 = nib.load(tissues)
    else: # more than one compartment
        img_0 = nib.load(tissues[0])

    shape = list(img_0.get_data().shape)

    data = np.zeros(shape=shape)
    if isinstance(tissues, basestring):  # just one compartment
        data = nib.load(tissues).get_data()
    else:  # more than one compartment
        for image in tissues:
            data = data + nib.load(image).get_data()

    data = (data > threshold) * 1.0
    out_mask = join(getcwd(), basename(tissues[0]) + '_brainmask.nii')

    mask = nib.Nifti1Image(data, img_0.affine, header=img_0.header)
    nib.save(mask, out_mask)
    return out_mask

def apply_binary_mask(image, binary_mask):
    import nibabel as nib
    from os import getcwd
    from os.path import join, basename

    original_image = nib.load(image)
    mask = nib.load(binary_mask)

    data = original_image.get_data() * mask.get_data()

    masked_image_path = join(getcwd(), 'masked_' + basename(image))
    masked_image = nib.Nifti1Image(data, original_image.affine, header=original_image.header)
    nib.save(masked_image, masked_image_path)
    return masked_image_path

def grab_dti_maps_adni(caps_directory, tsv):
    """
    this is to grab the outputs dti maps from dwi-processing pipeline in my home, not the CAPS in Clinica
    Args:
        CAPS_in:
        tsv:

    Returns:

    """
    import os, csv

    # try to get the atlas in $FSLDIR
    try:
        freesurfer_dir = os.environ.get('FREESURFER_HOME', '')
        if not freesurfer_dir:
            raise RuntimeError('FREESURFER_HOME variable is not set')
    except Exception as e:
        print(str(e))
        exit(1)

    subject_list = []
    session_list = []
    subject_id_list = []
    with open(tsv, 'rb') as tsvin:
        tsv_reader = csv.reader(tsvin, delimiter='\t')

        for row in tsv_reader:
            if row[0] == 'participant_id':
                continue
            else:
                subject_list.append(row[0])
                session_list.append(row[1])
                subject_id_list.append((row[0] + '_' + row[1]))
        caps_directory = os.path.expanduser(caps_directory)

    fa_caps_jhu = []
    md_caps_jhu = []
    rd_caps_jhu = []
    ad_caps_jhu = []


    ###### the number of subject_list and session_list should be the same
    try:
        len(subject_list) == len(session_list)
    except RuntimeError:
        print "It seems that the nuber of session_list and subject_list are not in the same length, please check"
        raise

    num_subject = len(subject_list)
    for i in xrange(num_subject):
        ##############
        fa = os.path.join(caps_directory, 'subjects', subject_list[i], session_list[i], 'dwi', 'dti_based_processing', 'normalized_space',
                                   subject_list[i] + '_' + session_list[i] + '_acq-axial_dwi_space-MNI152Lin_res-1x1x1_FA.nii.gz')
        fa_caps_jhu += [fa]

        md = os.path.join(caps_directory, 'subjects', subject_list[i], session_list[i], 'dwi', 'dti_based_processing', 'normalized_space',
                                   subject_list[i] + '_' + session_list[i] + '_acq-axial_dwi_space-MNI152Lin_res-1x1x1_MD.nii.gz')
        md_caps_jhu += [md]

        rd = os.path.join(caps_directory, 'subjects', subject_list[i], session_list[i], 'dwi', 'dti_based_processing', 'normalized_space',
                                    subject_list[i] + '_' + session_list[i] + '_acq-axial_dwi_space-MNI152Lin_res-1x1x1_RD.nii.gz')
        rd_caps_jhu += [rd]

        ad = os.path.join(caps_directory, 'subjects', subject_list[i], session_list[i], 'dwi', 'dti_based_processing', 'normalized_space',
                                    subject_list[i] + '_' + session_list[i] + '_acq-axial_dwi_space-MNI152Lin_res-1x1x1_AD.nii.gz')
        ad_caps_jhu += [ad]

    return fa_caps_jhu, md_caps_jhu, ad_caps_jhu, rd_caps_jhu, subject_list, session_list, subject_id_list


def get_subid_sesid_mask_dti(subject_id, caps_directory, fwhm, compartment_name, threshold):
    """
    This is to extract the base_directory for the DataSink including participant_id and sesion_id in CAPS directory, also the tuple_list for substitution
    :param subject_id:
    :return: base_directory for DataSink
    """
    import os

    ## for MapNode
    participant_id = subject_id.split('_')[0]
    session_id = subject_id.split('_')[1]
    base_directory = os.path.join(caps_directory, 'subjects', participant_id, session_id, 'dwi',
                                          'postprocessing')

    subst_tuple_list = [  # registration
        ('masked_fwhm-' + str(fwhm) + 'mm_masked_' + subject_id + '_acq-axial_dwi_space-MNI152Lin_res-1x1x1_FA.nii.gz',
         subject_id + '_acq-axial_dwi_space-MNI152Lin_res-1x1x1_com-' + compartment_name + '_fwhm-' + str(
             fwhm) + '_threshold-' + str(
             threshold) + '_FA.nii.gz'),
        ('masked_fwhm-' + str(fwhm) + 'mm_masked_' + subject_id + '_acq-axial_dwi_space-MNI152Lin_res-1x1x1_MD.nii.gz',
         subject_id + '_acq-axial_dwi_space-MNI152Lin_res-1x1x1_com-' + compartment_name + '_fwhm-' + str(
             fwhm) + '_threshold-' + str(
             threshold) + '_MD.nii.gz'),
        ('masked_fwhm-' + str(fwhm) + 'mm_masked_' + subject_id + '_acq-axial_dwi_space-MNI152Lin_res-1x1x1_RD.nii.gz',
         subject_id + '_acq-axial_dwi_space-MNI152Lin_res-1x1x1_com-' + compartment_name + '_fwhm-' + str(
             fwhm) + '_threshold-' + str(
             threshold) + '_RD.nii.gz'),
        ('masked_fwhm-' + str(fwhm) + 'mm_masked_' + subject_id + '_acq-axial_dwi_space-MNI152Lin_res-1x1x1_AD.nii.gz',
         subject_id + '_acq-axial_dwi_space-MNI152Lin_res-1x1x1_com-' + compartment_name + '_fwhm-' + str(
             fwhm) + '_threshold-' + str(
             threshold) + '_AD.nii.gz')
        ]

    regexp_substitutions = [
        # I don't know why it's adding this empty folder, so I remove it:
        # NOTE, . means any characters and * means any number of repetition in python regex
        (r'/non_smoothed_fa/_apply_mask_fa\d{1,4}/', r'/'),
        (r'/non_smoothed_md/_apply_mask_md\d{1,4}/', r'/'),
        (r'/non_smoothed_ad/_apply_mask_ad\d{1,4}/', r'/'),
        (r'/non_smoothed_rd/_apply_mask_rd\d{1,4}/', r'/'),
        (r'/_reapply_mask_ad\d{1,4}/', r'/'),
        (r'/_reapply_mask_fa\d{1,4}/', r'/'),
        (r'/_reapply_mask_rd\d{1,4}/', r'/'),
        (r'/_reapply_mask_md\d{1,4}/', r'/'),
        (r'/smoothed_ad/', r'/'),
        (r'/smoothed_fa/', r'/'),
        (r'/smoothed_rd/', r'/'),
        (r'/smoothed_md/', r'/'),
        # I don't know why it's adding this empty folder, so I remove it:
        (r'trait_added/_datasinker\d{1,4}/', r'')
    ]

    return base_directory, subst_tuple_list, regexp_substitutions


def upsample_mask_tissues(in_file):
    """
    To upsample the 1,2,3 compartment segmented volume in SPM
    Args:
        in_file: a list containing the three compartments

    Returns:
    """
    import os
    from nipype.interfaces.freesurfer.preprocess import MRIConvert

    out_file = []

    if isinstance(in_file, basestring):  # just one compartment
        out_mask = os.path.basename(in_file).split('.nii.gz')[0] + '_upsample.nii.gz'
        mc = MRIConvert()
        mc.inputs.in_file = in_file
        mc.inputs.out_file = out_mask
        mc.inputs.vox_size = (1, 1, 1)
        mc.run()

        out_file.append(os.path.abspath(os.path.join(os.getcwd(), out_mask)))
    else: # more than one compartment
        for tissue in in_file:
            out_mask = os.path.basename(tissue).split('.nii.gz')[0] + '_upsample.nii.gz'
            mc = MRIConvert()
            mc.inputs.in_file = tissue
            mc.inputs.out_file = out_mask
            mc.inputs.vox_size = (1, 1, 1)
            mc.run()

            out_file.append(os.path.abspath(os.path.join(os.getcwd(), out_mask)))

    return out_file

def nilearn_smoothing(in_files, fwhm, out_prefix):
    """
    Using nilearn function to do smoothing, it seems better than SPM
    Args:
        in_files:
        fwhm:
        out_prefix:

    Returns:

    """
    from nilearn.image import smooth_img
    import os

    file_name = os.path.basename(in_files)
    smoothed_data = smooth_img(in_files, fwhm)
    smoothed_data.to_filename(out_prefix + file_name)

    smoothed_files = os.path.abspath(os.path.join(os.getcwd(), out_prefix + file_name))

    return smoothed_files

def dti_maps_threshold(in_file, low_thr, high_thr):
    """
    This function is to threshold the abnormal values in DTI maps which casued by diffusion data quality or bad preprocessing,
    e.g. negative value or very high value in MD, RD & AD.
    Args:
        in_file:
        low_thr:
        high_thr:

    Returns:

    """

    import nibabel as nib
    import numpy as np
    import os.path as op
    import os

    outfile = op.abspath(op.join(os.getcwd(), op.basename(in_file)))

    file_obj = nib.load(in_file)
    data = file_obj.get_data()
    img_affine = file_obj.affine

    clip_array = np.clip(data, low_thr, high_thr)
    clip_img = nib.Nifti1Image(clip_array, img_affine)
    clip_img.to_filename(outfile)

    return outfile