# -*- coding: utf-8 -*-
__author__ = ["Junhao Wen"]
__copyright__ = "Copyright 2016-2018 The Aramis Lab Team"
__credits__ = ["Junhao Wen"]
__license__ = "See LICENSE.txt file"
__version__ = "0.1.0"
__status__ = "Development"

from dwi_postprocessing_dti_utils import *

def dwi_postprocessing_dti(CAPS, tsv, mask_tissues = [1,2,3], mask_threshold=0.3,
                           smooth=8, working_directory=None):
    """
    This is a pipeline to mask the DTI maps in the template space, like JHU in MNI space.
    The tissue template fro WM GM and CSF should be from t1_spm pipeline
    Args:
        CAPS:  CAPS
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
    import os
    import tempfile
    import nipype.interfaces.fsl as fsl

    if working_directory is None:
        working_directory = tempfile.mkdtemp()

    # read the DTI maps
    fa_map, md_map, ad_map, rd_map, subject_list, session_list, subject_id_list = grab_dti_maps_adni(CAPS, tsv)

    # read Tissues from CAPS_in
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

    inputnode = npe.Node(nutil.IdentityInterface(
        fields=['fa_map', 'md_map', 'ad_map', 'rd_map', 'subject_id_list']),
        name='inputnode')
    inputnode.inputs.fa_map = fa_map
    inputnode.inputs.md_map = md_map
    inputnode.inputs.ad_map = ad_map
    inputnode.inputs.rd_map = rd_map
    inputnode.inputs.subject_id_list = subject_id_list

    tissues_caps_reader = npe.Node(
        nio.DataGrabber(infields=['tissues'],
                        outfields=['out_files']), name='tissues_caps_reader')
    tissues_caps_reader.inputs.base_directory = os.path.join(CAPS, 'groups', 'group-ADNIbl', 't1')
    tissues_caps_reader.inputs.template = 'wgroup-ADNIbl_template%s.nii'
    tissues_caps_reader.inputs.tissues = [tissue_names[t] for t in mask_tissues]
    tissues_caps_reader.inputs.sort_filelist = False

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
    upsample_mask = npe.Node(nutil.Function(input_names=['in_file'],
                                          output_names=['out_file'],
                                          function=upsample_mask_tissues),
                           name='upsample_mask')

    # Create binary mask from upsampling tissues segementation
    # =========================================
    binary_mask = npe.Node(nutil.Function(input_names=['tissues', 'threshold'],
                                          output_names=['out_mask'],
                                          function=create_binary_mask),
                           name='binary_mask')
    binary_mask.inputs.threshold = mask_threshold


    # Mask DTI image
    # ==============
    apply_mask_fa = npe.MapNode(nutil.Function(input_names=['image', 'binary_mask'],
                                         output_names=['masked_image_path'],
                                         function=apply_binary_mask),  iterfield=['image'],
                          name='apply_mask_fa')

    apply_mask_md = npe.MapNode(nutil.Function(input_names=['image', 'binary_mask'],
                                         output_names=['masked_image_path'],
                                         function=apply_binary_mask),  iterfield=['image'],
                          name='apply_mask_md')

    apply_mask_ad = npe.MapNode(nutil.Function(input_names=['image', 'binary_mask'],
                                         output_names=['masked_image_path'],
                                         function=apply_binary_mask),  iterfield=['image'],
                          name='apply_mask_ad')

    apply_mask_rd = npe.MapNode(nutil.Function(input_names=['image', 'binary_mask'],
                                         output_names=['masked_image_path'],
                                         function=apply_binary_mask),  iterfield=['image'],
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
                                               function=apply_binary_mask), iterfield=['image'],
                                name='reapply_mask_fa')

    reapply_mask_md = npe.MapNode(nutil.Function(input_names=['image', 'binary_mask'],
                                               output_names=['masked_image_path'],
                                               function=apply_binary_mask), iterfield=['image'],
                                name='reapply_mask_md')

    reapply_mask_ad = npe.MapNode(nutil.Function(input_names=['image', 'binary_mask'],
                                               output_names=['masked_image_path'],
                                               function=apply_binary_mask), iterfield=['image'],
                                name='reapply_mask_ad')

    reapply_mask_rd = npe.MapNode(nutil.Function(input_names=['image', 'binary_mask'],
                                               output_names=['masked_image_path'],
                                               function=apply_binary_mask), iterfield=['image'],
                                name='reapply_mask_rd')

    outputnode = npe.Node(nutil.IdentityInterface(
        fields=['smoothed_fa', 'smoothed_md', 'smoothed_ad', 'smoothed_rd']),
        name='outputnode')


    # get the information for datasinker.
    get_identifiers = npe.MapNode(nutil.Function(
        input_names=['subject_id', 'caps_directory', 'fwhm', 'compartment_name', 'threshold'], output_names=['base_directory', 'subst_tuple_list', 'regexp_substitutions'],
        function=get_subid_sesid_mask_dti), iterfield=['subject_id'], name='get_subid_sesid_mask_dti')
    get_identifiers.inputs.caps_directory = CAPS
    get_identifiers.inputs.fwhm = smooth
    get_identifiers.inputs.compartment_name = compartment_name
    get_identifiers.inputs.threshold = mask_threshold

    ### datasink
    datasink = npe.MapNode(nio.DataSink(infields=['smoothed_fa', 'smoothed_md', 'smoothed_ad', 'smoothed_rd']), name='datasinker',
                          iterfield=['smoothed_fa', 'smoothed_md', 'smoothed_ad', 'smoothed_rd', 'base_directory', 'substitutions', 'regexp_substitutions'])
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
                # smoothed dti maps
                (reapply_mask_fa, datasink, [('masked_image_path', 'smoothed_fa')]),
                (reapply_mask_md, datasink, [('masked_image_path', 'smoothed_md')]),
                (reapply_mask_ad, datasink, [('masked_image_path', 'smoothed_ad')]),
                (reapply_mask_rd, datasink, [('masked_image_path', 'smoothed_rd')]),

                ])


    return wf






