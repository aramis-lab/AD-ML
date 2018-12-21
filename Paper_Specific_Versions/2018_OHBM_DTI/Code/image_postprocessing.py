# -*- coding: utf-8 -*-
"""
Created on Mon Nov 06 14:17:41 2017

@author: Junhao WEN
"""

def ADNI_mask_dti_from_spm(caps_directory, tsv, mask_tissues = [1,2,3], mask_threshold=0.3,
                           smooth=4, working_directory=None):
    """
    This is a pipeline to mask the DTI maps in the template space, like JHU in MNI space.
    You should have two CAPS, one from spm pipeline, one from dwi-processing pipeline, all the masked
    and smoothed images will be stored in caps_directory from DTI
    Args:
        caps_directory: dwi-processing CAPS
        tsv:Value to be used as threshold to binarize the tissues mask
        mask_threshold:
        smooth: A list of integers specifying the different isomorphic fwhm in milimeters to smooth the image, default is 4 mm
        mask_tissues: Tissue classes (gray matter, GM; white matter, WM; cerebro-spinal fluid, CSF...) to use for masking the PET image. Ex: 1 2 3 is GM, WM and CSF

    Returns:

    """

    import nipype.interfaces.spm as spm
    import nipype.interfaces.io as nio
    import nipype.interfaces.utility as nutil
    import nipype.pipeline.engine as npe
    import clinica.pipelines.pet_preprocess_volume.pet_preprocess_volume_utils as utils
    import os
    import tempfile
    import nipype.interfaces.fsl as fsl

    if working_directory is None:
        working_directory = tempfile.mkdtemp()

    # read the DTI maps from caps_directory
    fa_map, md_map, ad_map, rd_map, subject_list, session_list, subject_id_list = grab_dti_maps_adni(caps_directory, tsv)

    # read Tissues from caps_directory
    # ====================
    tissue_names = {1: '1',
                    2: '2',
                    3: '3',
                    4: 'bone',
                    5: 'softtissue',
                    6: 'background'
                    }

    if set(mask_tissues) == set([1]):
        compartment_name = 'GM'
    elif set(mask_tissues) == set([1, 2]):
        compartment_name = 'GM_WM'
    elif set(mask_tissues) == set([2]):
        compartment_name = 'WM'
    elif set(mask_tissues) == set([1, 2, 3]):
        compartment_name = 'GM_WM_CSF'
    else:
        raise Exception('This combination has not been considered!')


    tissues_caps_reader = npe.MapNode(
        nio.DataGrabber(infields=['subject_id', 'session', 'tissues'],
                        outfields=['out_files']), iterfield=['subject_id', 'session'], name='tissues_caps_reader')
    tissues_caps_reader.inputs.base_directory = os.path.join(caps_directory, 'subjects')
    tissues_caps_reader.inputs.template = '%s/%s/t1/spm/dartel/group-ADNIbl/wgroup-ADNIbl_template%s.nii'
    tissues_caps_reader.inputs.tissues = [tissue_names[t] for t in mask_tissues]
    tissues_caps_reader.inputs.sort_filelist = False
    tissues_caps_reader.inputs.subject_id = subject_list
    tissues_caps_reader.inputs.session = session_list
   # tissues_caps_reader.inputs.subject_repeat = subject_list
   # tissues_caps_reader.inputs.session_repeat = session_list

    inputnode = npe.Node(nutil.IdentityInterface(
        fields=['fa_map', 'md_map', 'ad_map', 'rd_map', 'subject_id_list']),
        name='inputnode')
    inputnode.inputs.fa_map = fa_map
    inputnode.inputs.md_map = md_map
    inputnode.inputs.ad_map = ad_map
    inputnode.inputs.rd_map = rd_map
    inputnode.inputs.subject_id_list = subject_id_list

    #### threshold the DTI maps to reasonable values, especially voxels near to CSF
    thres_fa = npe.MapNode(nutil.Function(input_names=['in_file', 'low_thr', 'high_thr'],
                                          output_names=['out_file'],
                                          function=dti_maps_threshold), iterfield=['in_file'],
                           name='thres_fa')
    thres_fa.inputs.low_thr = 0
    thres_fa.inputs.high_thr = 1


    thres_md = npe.MapNode(nutil.Function(input_names=['in_file', 'low_thr', 'high_thr'],
                                          output_names=['out_file'],
                                          function=dti_maps_threshold), iterfield=['in_file'],
                           name='thres_md')
    thres_md.inputs.low_thr = 0
    thres_md.inputs.high_thr = 0.009

    thres_rd = npe.MapNode(nutil.Function(input_names=['in_file', 'low_thr', 'high_thr'],
                                          output_names=['out_file'],
                                          function=dti_maps_threshold), iterfield=['in_file'],
                           name='thres_rd')
    thres_rd.inputs.low_thr = 0
    thres_rd.inputs.high_thr = 0.009

    thres_ad = npe.MapNode(nutil.Function(input_names=['in_file', 'low_thr', 'high_thr'],
                                          output_names=['out_file'],
                                          function=dti_maps_threshold), iterfield=['in_file'],
                           name='thres_ad')
    thres_ad.inputs.low_thr = 0
    thres_ad.inputs.high_thr = 0.009

    ### upsampling the binary mask to the DTI's resolution in ADNI (1*1*1 mm)
    upsample_mask = npe.MapNode(nutil.Function(input_names=['in_file'],
                                          output_names=['out_file'],
                                          function=upsample_mask_tissues), iterfield=['in_file'],
                           name='upsample_mask')

    # Create binary mask from upsampling tissues segementation
    # =========================================
    binary_mask = npe.MapNode(nutil.Function(input_names=['tissues', 'threshold'],
                                          output_names=['out_mask'],
                                          function=create_binary_mask), iterfield=['tissues'],
                           name='binary_mask')
    binary_mask.inputs.threshold = mask_threshold


    # Mask DTI image
    # ==============
    apply_mask_fa = npe.MapNode(nutil.Function(input_names=['image', 'binary_mask'],
                                         output_names=['masked_image_path'],
                                         function=apply_binary_mask),  iterfield=['image', 'binary_mask'],
                          name='apply_mask_fa')

    apply_mask_md = npe.MapNode(nutil.Function(input_names=['image', 'binary_mask'],
                                         output_names=['masked_image_path'],
                                         function=apply_binary_mask),  iterfield=['image', 'binary_mask'],
                          name='apply_mask_md')

    apply_mask_ad = npe.MapNode(nutil.Function(input_names=['image', 'binary_mask'],
                                         output_names=['masked_image_path'],
                                         function=apply_binary_mask),  iterfield=['image', 'binary_mask'],
                          name='apply_mask_ad')

    apply_mask_rd = npe.MapNode(nutil.Function(input_names=['image', 'binary_mask'],
                                         output_names=['masked_image_path'],
                                         function=apply_binary_mask),  iterfield=['image', 'binary_mask'],
                          name='apply_mask_rd')

    ################ smooth the DTI maps with nilean not SPM
    smoothing_node_fa = npe.MapNode(nutil.Function(input_names=['in_files', 'fwhm', 'out_prefix'],
                                               output_names=['smoothed_files'],
                                               function=nilearn_smoothing), iterfield=['in_files'],
                                name='smoothing_node_fa')
    smoothing_node_fa.inputs.fwhm = smooth
    smoothing_node_fa.inputs.out_prefix = 'fwhm-' + str(smooth) + 'mm_'

    smoothing_node_md = npe.MapNode(nutil.Function(input_names=['in_files', 'fwhm', 'out_prefix'],
                                               output_names=['smoothed_files'],
                                               function=nilearn_smoothing), iterfield=['in_files'],
                                name='smoothing_node_md')
    smoothing_node_md.inputs.fwhm = smooth
    smoothing_node_md.inputs.out_prefix = 'fwhm-' + str(smooth) + 'mm_'

    smoothing_node_ad = npe.MapNode(nutil.Function(input_names=['in_files', 'fwhm', 'out_prefix'],
                                               output_names=['smoothed_files'],
                                               function=nilearn_smoothing), iterfield=['in_files'],
                                name='smoothing_node_ad')
    smoothing_node_ad.inputs.fwhm = smooth
    smoothing_node_ad.inputs.out_prefix = 'fwhm-' + str(smooth) + 'mm_'

    smoothing_node_rd = npe.MapNode(nutil.Function(input_names=['in_files', 'fwhm', 'out_prefix'],
                                               output_names=['smoothed_files'],
                                               function=nilearn_smoothing), iterfield=['in_files'],
                                name='smoothing_node_rd')
    smoothing_node_rd.inputs.fwhm = smooth
    smoothing_node_rd.inputs.out_prefix = 'fwhm-' + str(smooth) + 'mm_'


    ##### remask the smoothed DTI maps
    reapply_mask_fa = npe.MapNode(nutil.Function(input_names=['image', 'binary_mask'],
                                               output_names=['masked_image_path'],
                                               function=apply_binary_mask), iterfield=['image', 'binary_mask'],
                                name='reapply_mask_fa')

    reapply_mask_md = npe.MapNode(nutil.Function(input_names=['image', 'binary_mask'],
                                               output_names=['masked_image_path'],
                                               function=apply_binary_mask), iterfield=['image', 'binary_mask'],
                                name='reapply_mask_md')

    reapply_mask_ad = npe.MapNode(nutil.Function(input_names=['image', 'binary_mask'],
                                               output_names=['masked_image_path'],
                                               function=apply_binary_mask), iterfield=['image', 'binary_mask'],
                                name='reapply_mask_ad')

    reapply_mask_rd = npe.MapNode(nutil.Function(input_names=['image', 'binary_mask'],
                                               output_names=['masked_image_path'],
                                               function=apply_binary_mask), iterfield=['image', 'binary_mask'],
                                name='reapply_mask_rd')

    outputnode = npe.Node(nutil.IdentityInterface(
        fields=['smoothed_fa', 'smoothed_md', 'smoothed_ad', 'smoothed_rd', 'non_smoothed_fa', 'non_smoothed_md', 'non_smoothed_ad', 'non_smoothed_ad']),
        name='outputnode')


    # get the information for datasinker.
    get_identifiers = npe.Node(nutil.Function(
        input_names=['subject_id', 'caps_directory', 'fwhm', 'compartment_name', 'threshold'], output_names=['base_directory', 'subst_tuple_list', 'regexp_substitutions'],
        function=get_subid_sesid_mask_dti), name='get_subid_sesid_highbval_dti')
    get_identifiers.inputs.caps_directory = caps_directory
    get_identifiers.inputs.fwhm = smooth
    get_identifiers.inputs.compartment_name = compartment_name
    get_identifiers.inputs.threshold = mask_threshold

    ### datasink
    datasink = npe.MapNode(nio.DataSink(infields=['smoothed_fa', 'smoothed_md', 'smoothed_ad', 'smoothed_rd', 'non_smoothed_fa', 'non_smoothed_md', 'non_smoothed_ad', 'non_smoothed_ad']), name='datasinker',
                          iterfield=['smoothed_fa', 'smoothed_md', 'smoothed_ad', 'smoothed_rd', 'non_smoothed_fa', 'non_smoothed_md', 'non_smoothed_ad', 'non_smoothed_ad', 'base_directory'])
    datasink.inputs.remove_dest_dir = True


    wf = npe.Workflow(name='mask_dti_maps')
    wf.base_dir = working_directory

    wf.connect([(tissues_caps_reader, upsample_mask, [('out_files', 'in_file')]),
                (upsample_mask, binary_mask, [('out_file', 'tissues')]),
                # fa
                (binary_mask, apply_mask_fa, [('out_mask', 'binary_mask')]),
                (inputnode, thres_fa, [('fa_map', 'in_file')]),
                (thres_fa, apply_mask_fa, [('out_file', 'image')]),
                (apply_mask_fa, smoothing_node_fa, [('masked_image_path', 'in_files')]),
                (smoothing_node_fa, reapply_mask_fa, [('smoothed_files', 'image')]),
                (binary_mask, reapply_mask_fa, [('out_mask', 'binary_mask')]),
                # md
                (binary_mask, apply_mask_md, [('out_mask', 'binary_mask')]),
                (inputnode, thres_md, [('md_map', 'in_file')]),
                (thres_md, apply_mask_md, [('out_file', 'image')]),
                (apply_mask_md, smoothing_node_md, [('masked_image_path', 'in_files')]),
                (smoothing_node_md, reapply_mask_md, [('smoothed_files', 'image')]),
                (binary_mask, reapply_mask_md, [('out_mask', 'binary_mask')]),
                # ad
                (binary_mask, apply_mask_ad, [('out_mask', 'binary_mask')]),
                (inputnode, thres_ad, [('ad_map', 'in_file')]),
                (thres_ad, apply_mask_ad, [('out_file', 'image')]),
                (apply_mask_ad, smoothing_node_ad, [('masked_image_path', 'in_files')]),
                (smoothing_node_ad, reapply_mask_ad, [('smoothed_files', 'image')]),
                (binary_mask, reapply_mask_ad, [('out_mask', 'binary_mask')]),
                # rd
                (binary_mask, apply_mask_rd, [('out_mask', 'binary_mask')]),
                (inputnode, thres_rd, [('rd_map', 'in_file')]),
                (thres_rd, apply_mask_rd, [('out_file', 'image')]),
                (apply_mask_rd, smoothing_node_rd, [('masked_image_path', 'in_files')]),
                (smoothing_node_rd, reapply_mask_rd, [('smoothed_files', 'image')]),
                (binary_mask, reapply_mask_rd, [('out_mask', 'binary_mask')]),

                ## datasink
                # Saving files with datasink:
                (inputnode, get_identifiers, [('subject_id_list', 'subject_id')]),
                (get_identifiers, datasink, [('base_directory', 'base_directory')]),
                (get_identifiers, datasink, [('subst_tuple_list', 'substitutions')]),
                (get_identifiers, datasink, [('regexp_substitutions', 'regexp_substitutions')]),
                # datasink to save outputs
                # non smoothed dti maps
                (apply_mask_fa, datasink, [('masked_image_path', 'non_smoothed_fa')]),
                (apply_mask_md, datasink, [('masked_image_path', 'non_smoothed_md')]),
                (apply_mask_ad, datasink, [('masked_image_path', 'non_smoothed_ad')]),
                (apply_mask_rd, datasink, [('masked_image_path', 'non_smoothed_rd')]),
                # smoothed dti maps
                (reapply_mask_fa, datasink, [('masked_image_path', 'smoothed_fa')]),
                (reapply_mask_md, datasink, [('masked_image_path', 'smoothed_md')]),
                (reapply_mask_ad, datasink, [('masked_image_path', 'smoothed_ad')]),
                (reapply_mask_rd, datasink, [('masked_image_path', 'smoothed_rd')]),

                ])


    return wf



########################################################################################################################
##################
################## UTILS
##################
########################################################################################################################
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
        caps_directory:
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
        fa = os.path.join(caps_directory, 'subjects', subject_list[i], session_list[i], 'dwi', 'normalized_space',
                                   'SyN_QuickWarped_thresh.nii.gz')
        fa_caps_jhu += [fa]

        md = os.path.join(caps_directory, 'subjects', subject_list[i], session_list[i], 'dwi', 'normalized_space',
                                   'space-JHUTracts0_md_thresh.nii.gz')
        md_caps_jhu += [md]

        rd = os.path.join(caps_directory, 'subjects', subject_list[i], session_list[i], 'dwi', 'normalized_space',
                                    'space-JHUTracts0_rd_thresh.nii.gz')
        rd_caps_jhu += [rd]

        ad = os.path.join(caps_directory, 'subjects', subject_list[i], session_list[i], 'dwi', 'normalized_space',
                                    'space-JHUTracts0_ad_thresh.nii.gz')
        ad_caps_jhu += [ad]

    return fa_caps_jhu, md_caps_jhu, ad_caps_jhu, rd_caps_jhu, subject_list, session_list, subject_id_list


def get_subid_sesid_mask_dti(subject_id, caps_directory, fwhm, compartment_name, threshold):
    """
    This is to extract the base_directory for the DataSink including participant_id and sesion_id in CAPS directory, also the tuple_list for substitution
    :param subject_id:
    :return: base_directory for DataSink
    """
    import os

    base_directory=[]
    for sub in subject_id:
        participant_id = sub.split('_')[0]
        session_id = sub.split('_')[1]
        base_director_path = os.path.join(caps_directory, 'subjects', participant_id, session_id, 'dwi', 'normalized_space')
        base_directory.append(base_director_path)
    subst_tuple_list = [# registration
    ('masked_SyN_QuickWarped_thresh.nii.gz', compartment_name + '_non_smoothed_fa_masked_threshold-' + str(threshold) + '.nii.gz'),
    ('masked_space-JHUTracts0_ad_thresh.nii.gz', compartment_name + '_non_smoothed_ad_masked_threshold-' + str(threshold) + '.nii.gz'),
    ('masked_space-JHUTracts0_md_thresh.nii.gz', compartment_name + '_non_smoothed_md_masked_threshold-' + str(threshold) + '.nii.gz'),
    ('masked_space-JHUTracts0_rd_thresh.nii.gz', compartment_name + '_non_smoothed_rd_masked_threshold-' + str(threshold) + '.nii.gz'),
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
        (r'masked_fwhm-'+ str(fwhm) + 'mm_' + compartment_name + '_non_smoothed_fa_masked_threshold-' + str(threshold) + '.nii.gz', compartment_name + '_fwhm-'+ str(fwhm) + 'mm_fa_masked_threshold-' + str(threshold) + '.nii.gz'),
        (r'masked_fwhm-'+ str(fwhm) + 'mm_' + compartment_name + '_non_smoothed_ad_masked_threshold-' + str(threshold) + '.nii.gz', compartment_name + '_fwhm-'+ str(fwhm) + 'mm_ad_masked_threshold-' + str(threshold) + '.nii.gz'),
        (r'masked_fwhm-'+ str(fwhm) + 'mm_' + compartment_name + '_non_smoothed_rd_masked_threshold-' + str(threshold) + '.nii.gz', compartment_name + '_fwhm-'+ str(fwhm) + 'mm_rd_masked_threshold-' + str(threshold) + '.nii.gz'),
        (r'masked_fwhm-'+ str(fwhm) + 'mm_' + compartment_name + '_non_smoothed_md_masked_threshold-' + str(threshold) + '.nii.gz', compartment_name + '_fwhm-'+ str(fwhm) + 'mm_md_masked_threshold-' + str(threshold) + '.nii.gz'),
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

