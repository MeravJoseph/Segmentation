""" Semantic Segmentation from Training Image """
# *************************************************
# *** Semantic Segmentation from Training Image ***
# ************ Merav Joseph 200652063 *************
# ************* Shir Amir 209712801 ***************
# *************************************************

import os.path
from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float
import matplotlib.pyplot as plt
import numpy as np
import cv2

def set_img_border(image, border_size, border_value=-1):
    """ confine borders so they wouldn't be chosen as patches.
    :param image: the image to be confined
    :param border_size: the thickness of the border
    :param border_value: the value of the border pixels
    :return: the bordered image
    """
    img = image.copy()
    img[:, 0:border_size] = border_value
    img[:, -border_size:] = border_value
    img[0:border_size, :] = border_value
    img[-border_size:, :] = border_value
    return img

def extract_patches(im_rgb, im_segments, patch=11):
    """ creates patches for each segment.
    :param im_rgb: the original image
    :param im_segments: the segmented image
    :param patch:  the edge size of the patch
    :param sample_size: fraction of sampled pixels in each fragment
    :return: a list of all the patches
    """
    half_patch = int(patch / 2)
    im_segments_with_border = set_img_border(im_segments, half_patch, border_value=-1)

    # array of unique segments numbers
    unique_segments = np.unique(im_segments)

    # loop over all segments
    result = []
    for seg_num in unique_segments:
        # Find all pixels coordinates in the segment
        (i, j) = np.where(im_segments_with_border == seg_num)

        patches = []
        # Make sure we have at least 1 pixel in segment
        if len(i) > 0:
            # Choose randomly pixels from segment
            sample_size = np.floor(np.sqrt(len(i)))
            sample_indices = np.random.random_integers(0, len(i)-1, np.int64(sample_size))

            # Construct patches
            for patch_ind, pix_ind in enumerate(sample_indices):
                roi = im_rgb[(i[pix_ind] - half_patch):(i[pix_ind] + half_patch + 1),
                             (j[pix_ind] - half_patch):(j[pix_ind] + half_patch + 1), :]
                roi = roi - np.mean(roi) #Substract mean from patch
                patches.append(roi)
        result.append(patches)
    return result

def paint_image_fragments(im_rgb, im_segments):
    """ color each segment of the image with the mean value of the segment.
    :param im_rgb: the original image
    :param im_segments: the segmented image
    :return: the original image with painted segments according to their mean value
    """

    # array of unique segments numbers
    unique_segments = np.unique(im_segments)

    # loop over all segments
    result = im_rgb.copy()

    for seg_num in unique_segments:
        # Find the mean value of the segment
        mean_segment_val = np.mean(im_rgb[im_segments == seg_num, :], axis=0)
        # Color the segment in its mean value
        result[im_segments == seg_num, :] = mean_segment_val
    return result

def compute_mask(fragments, distance, threshold, maybe_threshold):
    """ Create a mask for grabcut of a certain label
    :param fragments: The fragmented images, when every pixels contains its fragment's index
    :param distance: The distance matrix column of the current label
    :param threshold: The threshold to determine absolute background / foreground
    :param maybe_threshold: The threshold to determine probable background / foreground
    :param kwargs: the parameter settings from the GUI
    """
    mask = np.zeros(fragments.shape, np.uint8)
    unique_fragments = np.unique(fragments)
    # for each fragment check
    for frag_num in unique_fragments:
        frag_dist_from_label = distance[frag_num]
        if frag_dist_from_label > (1-threshold):
            # fragment is positively background
            mask[fragments == frag_num] = cv2.GC_BGD
        elif frag_dist_from_label > maybe_threshold:
            # fragment is maybe background
            mask[fragments == frag_num] = cv2.GC_PR_BGD
        elif frag_dist_from_label < threshold:
            # fragment is positively foreground
            mask[fragments == frag_num] = cv2.GC_FGD
        else:
            # fragment is maybe foreground
            mask[fragments == frag_num] = cv2.GC_PR_FGD
    return mask

def segment_image(**kwargs):
    """ execute the segmentation itself
    :param kwargs: the parameter settings from the GUI
    """
    # Default parameters
    train_img_path = "../images/girl_train.jpg"
    labels_img_path = "../images/girl_train_labels.tif"
    test_img_path = "../images/girl_test.jpg"
    output_dir = "../results/"
    frag_amount = 100
    patch_size = 9
    grabcut_thresh = 0.01
    grabcut_iter = 5
    slic_sigma = 5

    # Change parameters specified by GUI
    if kwargs.get('TRAIN_IMG_PATH') != '':
        train_img_path = kwargs.get('TRAIN_IMG_PATH')
    if kwargs.get('LABELS_IMG_PATH') != '':
        labels_img_path = kwargs.get('LABELS_IMG_PATH')
    if kwargs.get('TEST_IMG_PATH') != '':
        test_img_path = kwargs.get('TEST_IMG_PATH')
    if kwargs.get('OUTPUT_DIR') != '':
        output_dir = kwargs.get('OUTPUT_DIR')
    if kwargs.get('FRAG_AMOUNT') != '':
        frag_amount = int(kwargs.get('FRAG_AMOUNT'))
    if kwargs.get('PATCH_SIZE') != '':
        patch_size = int(kwargs.get('PATCH_SIZE'))
    if kwargs.get('GRABCUT_THRESH') != '':
        grabcut_thresh = float(kwargs.get('GRABCUT_THRESH'))
    if kwargs.get('GRABCUT_ITER') != '':
        grabcut_iter = int(kwargs.get('GRABCUT_ITER'))
    if kwargs.get('SLIC_SIGMA') != '':
        slic_sigma = int(kwargs.get('SLIC_SIGMA'))

    # Load images
    train_img = cv2.cvtColor(cv2.imread(train_img_path), cv2.COLOR_BGR2RGB)
    labels_img = cv2.imread(labels_img_path, cv2.IMREAD_GRAYSCALE)
    test_img = cv2.cvtColor(cv2.imread(test_img_path), cv2.COLOR_BGR2RGB)

    # Extract train patches from each label in train image
    train_labels_patches = extract_patches(train_img, labels_img, patch_size)
    labels_nums = range(len(train_labels_patches))

    # Compute SLIC fragmentation
    # Each pixel contains the key of its' fragment
    fragments = slic(img_as_float(test_img), n_segments=frag_amount, sigma=slic_sigma)

    # Color each segment of the image with the segment mean value
    graphcut_input_img = paint_image_fragments(test_img, fragments)
    graphcut_input_img = graphcut_input_img.astype('uint8')

    # Show SLIC output and coloring output
    fig = plt.figure("superpixel fragmentation")
    ax = fig.add_subplot(1, 2, 1)
    cax = ax.imshow(mark_boundaries(img_as_float(test_img), fragments))
    ax.set_title("mean value fragmentation color")
    ax = fig.add_subplot(1, 2, 2)
    cax = ax.imshow(graphcut_input_img)

    # Extract test patches from fragments
    fragments_patches = extract_patches(test_img, fragments, patch_size)
    fragments_nums = range(len(fragments_patches))

    # Compute distance
    distance = np.zeros((len(fragments_nums), len(labels_nums)), dtype=np.float32)
    M = float(patch_size**2 * 3)
    # For each fragment
    for frag_key in fragments_nums:
        frag_patches = fragments_patches[frag_key]
        # For each label
        for label_key in labels_nums:
            label_patches = train_labels_patches[label_key]
            label_patches_mat = np.stack(label_patches)
            cost_patches = []
            # for each patch of this fragment
            for frag_p in frag_patches:
                # compute cost between fragment patch and label patch
                diff_sqr = ((label_patches_mat - frag_p)/M)**2
                ssd = np.sum(diff_sqr.reshape((label_patches_mat.shape[0], -1)), axis=1)
                min_ssd = np.min(ssd)
                cost_patches.append(min_ssd)
            distance[frag_key, label_key] = np.median(cost_patches)

    # Normalize distance values
    # When normalizing, crop the range to avoid distant outliers
    valid_percentile = 99
    distance_limits = [np.min(distance), np.percentile(distance, valid_percentile)]
    distance[np.where(distance > distance_limits[1])] = distance_limits[1]
    distance = np.interp(distance, distance_limits, [0, 1])

    # Naive Segmentation - Choosing the best option in distance matrix
    min_dist = np.argmin(distance, axis=1)
    frag_map = np.zeros_like(fragments)
    for i in range(len(min_dist)):
        frag_map[fragments == i] = min_dist[i]

    # Determine final segmentation with multi-label graph-cut
    # Define the probable threshold as the average of average minimum and average maximum
    min_t = np.min(distance, 1)
    max_t = np.max(distance, 1)
    mean_min = np.mean(min_t)
    mean_max = np.mean(max_t)
    maybe_threshold = (mean_min + mean_max) / 2
    graphcut_labeling = np.ones([np.size(test_img, 0), np.size(test_img, 1), len(labels_nums)], np.float32) * np.nan
    #for each label execute grabcut algorithm
    for label_key in labels_nums:
        mask = compute_mask(fragments, distance[:, label_key], grabcut_thresh, maybe_threshold)
        temp_mask = mask.copy()
        unique_fg_bg_vals = np.unique(mask)
        # make sure there are both foregrounds and backgrounds as requiered by grabcut
        if np.logical_and(np.logical_or(cv2.GC_FGD in unique_fg_bg_vals, cv2.GC_PR_FGD in unique_fg_bg_vals), \
                          np.logical_or(cv2.GC_BGD in unique_fg_bg_vals, cv2.GC_PR_BGD in unique_fg_bg_vals)):
            bgd_model = np.zeros((1, 65), np.float64)
            fgd_model = np.zeros((1, 65), np.float64)
            cv2.grabCut(graphcut_input_img, mask, None, bgd_model, fgd_model, grabcut_iter, cv2.GC_INIT_WITH_MASK)
            foreground_mask = np.logical_or(mask == cv2.GC_FGD, mask == cv2.GC_PR_FGD)

            v = distance[:, label_key]
            cost = v[fragments]
            cost[~foreground_mask] = np.nan
            graphcut_labeling[:, :, label_key] = cost

    # Map of fragments that wasn't assigned to any label as foreground
    invalid_map = np.all(np.isnan(graphcut_labeling), axis=2)
    graphcut_labeling[invalid_map, :] = 0
    # Create result image by taking the best label for each fragment
    # according to the label score and the graphcut mapping results
    test_img_result = np.nanargmin(graphcut_labeling, axis=2)
    # If there's a fragment that was not chosen as foreground by any label - assign to the best label (naive)
    test_img_result[invalid_map] = frag_map[invalid_map]

    # Find a file name that isn't taken
    index = 1
    while os.path.isfile('%s%s%d%s' % (output_dir, 'result', index, '.tif')):
        index = index + 1

    # Save the result
    res_path = '%s%s%d%s' % (output_dir, 'result', index, '.tif')
    plt.imsave(res_path, test_img_result)
    
    # Return the result's file name to the GUI
    return res_path