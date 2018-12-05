import argparse
from functools import partial
import logging
import os
import io
import numpy as np
import sys


import PIL.Image
import tensorflow as tf
from tensorflow.python.lib.io import file_io

from config import Config

def main(argv):
    parser = argparse.ArgumentParser(
        description='Convert the ADE20K Challenge dataset'
    )

    parser.add_argument(
        '-w', '--whitelist-labels', type=str,
        help=('A pipe | separated list of object labels to whitelist. '
              'categories can be merged by seperating them by : '
              'e.g. "person|car:truck:van|pavement". To see a'
              ' full list of allowed labels run with  --list-labels.')
    )

    parser.add_argument(
        '-t', '--whitelist-threshold', type=float, default=0.7,
        help=('The fraction of whitelisted labels an image must contain to be '
              'used for training.')
    )

    args = parser.parse_args(argv)
    
    label_filename = 'ADEChallengeData2016/objectInfo150.txt'
   
    # Load the class labels
    class_labels = _load_class_labels(label_filename)
    n_classes = len(class_labels)
    print('initial number of classes:', n_classes)
    
    # If a whitelist is provided, get a list of mask indices that correspond
    # to allowed labels
    whitelist_labels = None
    whitelist_indices = None
    if args.whitelist_labels:
        whitelist_labels = _parse_whitelist_labels(args.whitelist_labels)
        # add a 'none' class with a label of 0
        whitelist_labels.insert(0, ['none'])
        whitelist_indices = _find_whitelist_indices(
            class_labels, whitelist_labels)

        whitelist_filename = os.path.join(
            os.path.dirname(args.output), 'labels.txt')
        _save_whitelist_labels(whitelist_filename, whitelist_labels)
        n_classes = len(whitelist_labels)
        print('number of classes whitelisted: ', n_classes)

    _create_tfrecord_dataset(
        'ADEChallengeData2016/images/training',
        'ADEChallengeData2016/antonations/training',
        n_classes,
        whitelist_indices=whitelist_indices,
        whitelist_threshold=args.whitelist_threshold
    )

    for root, dirs, files in os.walk('ADEChallengeData2016'):
        


def _parse_whitelist_labels(whitelist):
    parsed = whitelist.split('|')
    parsed = [category.split(':') for category in parsed]
    return parsed

def _save_whitelist_labels(whitelist_filename, labels):
    with open(whitelist_filename, 'w') as wfid:
        header = 'idx\tlabel\n'
        wfid.write(header)
        for idx, label_set in enumerate(labels):
            label = ':'.join(label_set[0])
            wfid.write('%d\t%s\n' % (idx, label))


def _load_class_labels(label_filename):
    """Load class labels.

    Assumes the data directory is left unchanged from the original zip.

    Args:
        root_directory (str): the dataset's root directory

    Returns:
        List[(int, str)]: a list of class ids and labels
    """
    class_labels = []
    header = True
    with file_io.FileIO(label_filename, mode='r') as file:
        for line in file.readlines():
            if header:
                class_labels.append((0, 'none'))
                header = False
                continue
            line = line.rstrip()
            line = line.split('\t')
            label = line[-1]
            label_id = int(line[0])
            class_labels.append((label_id, label))
    return class_labels


def _find_whitelist_indices(class_labels, whitelist_labels):
    """Map whitelist labels to indices.

    Args:
        whitelist (List[str]): a list of whitelisted labels

    Returns:
        List[Set]: a list of sets containing index labels
    """
    index = []
    for label_set in whitelist_labels:
        index_set = []
        for label in label_set:
            for class_id, class_label in class_labels:
                if label == class_label:
                    index_set.append(class_id)
        index.append(index_set)
    return index


def _filter_whitelabel_classes(
        filenames,
        whitelist,
        whitelist_threshold,
        whitelist_size=None):
    w_size = whitelist_size or len(whitelist)
    mask = np.array(PIL.Image.open(filenames))
    unique_classes = np.unique(mask)
    num_found = np.intersect1d(unique_classes, whitelist).size
    if float(num_found) / w_size >= whitelist_threshold:
        return True
    return False


def _relabel_mask(seg_img, whitelist_indices):
    # Read the data into a np array.
    mask = np.array(PIL.Image.open(seg_img))
    # Relabel each pixel
    new_mask = np.zeros(mask.shape)
    for new_label, old_label_set in enumerate(whitelist_indices):
        idx = np.where(np.isin(mask, old_label_set))
        new_mask[idx] = new_label
    # Convert the new mask back to an image.
    seg_new = PIL.Image.fromarray(new_mask.astype('uint8')).convert('RGB')
    
    return seg_new


def _create_tfrecord_dataset(
        image_dir,
        segmentation_mask_dir,
        n_classes,
        whitelist_indices=None,
        whitelist_threshold=0.5):
    """Convert the ADE20k dataset into into tfrecord format.

    Args:
        dataset_split: Dataset split (e.g., train, val).
        dataset_dir: Dir in which the dataset locates.
        dataset_label_dir: Dir in which the annotations locates.
    Raises:
        RuntimeError: If loaded image and label have different shape.
    """
    # Get all of the image and segmentation mask file names
    img_names = tf.gfile.Glob(os.path.join(image_dir, '*.jpg'))
    seg_names = []
    for f in img_names:
        # get the filename without the extension
        basename = os.path.basename(f).split('.')[0]
        # cover its corresponding *_seg.png
        seg = os.path.join(segmentation_mask_dir, basename + '_seg.png')
        seg_names.append(seg)
    print('got all the names')
    # If a whitelist has been provided, loop over all of the segmentation
    # masks and find only the images that contain enough classes.
    kept_files = zip(img_names, seg_names)
    if whitelist_indices is not None:
        # Flatten the whitelist because some categories have been merged
        # but make sure to use the orginal list size when
        # computing the threshold.
        print('whitelist: ', whitelist_indices)
        flat_whitelist = np.array(
            [idx for idx_set in whitelist_indices for idx in idx_set]
        ).astype('uint8')
        print('flat whitelist: ')
        merged_whitelist_size = len(whitelist_indices)
        filter_fn = partial(
            _filter_whitelabel_classes,
            whitelist=flat_whitelist,
            whitelist_threshold=whitelist_threshold,
            whitelist_size=merged_whitelist_size
        )
        kept_files = list(filter(filter_fn, seg_names))
        print(
            'Found %d images after whitelist filtereing.' % len(kept_files))

        for idx, (image_filename, seg_filename) in enumerate(kept_files):
            if idx % 100 == 0:
                print('Converting image: ', idx)
            if whitelist_indices is not None:

                orig_image = PIL.Image.open(image_filename)
                orig_image.save('aderelabeled/images/training' + image_filename)

                seg_image = _relabel_mask(seg_filename, whitelist_indices)
                seg_image.save('aderelabeled/images/training' + seg_filename)

if __name__ == "__main__":
    main(sys.argv[1:])
