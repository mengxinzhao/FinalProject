#!/usr/bin/env python
import glob
import logging
import multiprocessing as mp
import os
import time
import cv2
import numpy as np

##from medium_facenet_tutorial.align_dlib import AlignDlib
from align_dlib import AlignDlib

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

align_dlib = AlignDlib(os.path.join(os.path.dirname(__file__), 'shape_predictor_68_face_landmarks.dat'))


def test_process(input_dir, output_dir, crop_dim):
    start_time = time.time()
    pool = mp.Pool(processes=mp.cpu_count())

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    image_paths = glob.glob(os.path.join(input_dir, '*.jpg'))
    logger.info(input_dir)
    for index, image_path in enumerate(image_paths):
        image_name = os.path.basename(image_path)
        output_path= os.path.join(output_dir,image_name)
        logger.info(output_path)
        pool.apply_async(preprocess_image, (image_path, output_path, crop_dim, True))

    pool.close()
    pool.join()
    logger.info('Completed in {} seconds'.format(time.time() - start_time))


def test_resize(input_dir, output_dir, size):
    start_time = time.time()
    pool = mp.Pool(processes=mp.cpu_count())

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    image_paths = glob.glob(os.path.join(input_dir, '*.jpg'))
    logger.info(input_dir)
    for index, image_path in enumerate(image_paths):
        image_name = os.path.basename(image_path)
        output_path= os.path.join(output_dir,image_name)
        logger.info(output_path)
        pool.apply_async(resize_image, (image_path, output_path, size, False, True))

    pool.close()
    pool.join()
    logger.info('Completed in {} seconds'.format(time.time() - start_time))

def resize_image(input_path, output_path, size, random_padding_border_color = False, debug=False):
    """

    :param input_path: input image
    :param output_path: output path when debug flag is true
    :param size: square size
    :param debug: debug flag
    :return: resized image preserving width/height ratio with one side meeting size target. The other side is padded
    """
    image = _buffer_image(input_path)
    height, width = image.shape[:2]
    logger.info('image size {} required square size {}'.format(image.shape,size))
    # only shrink if img is bigger than required
    if size < height or size < width:
        # get scaling factor
        scaling_factor = size / float(height)
        if size / float(width) < scaling_factor:
            scaling_factor = size / float(width)

        # resize image
        img = cv2.resize(image, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_AREA)
        # pad image
        padded = pad_image(img,size,random_border_color = random_padding_border_color)
        logger.info('resize to {} and pad to {}'.format(img.shape, padded.shape))
        if debug == True:
            if output_path is not None:
                padded = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
                cv2.imwrite(output_path, padded)
        return padded
    else:
        return image

def pad_image(img,size = 224, random_border_color = False):
    """
    :param img: a cv image array
    :param size: target square size
    :return:padded image
    """
    height, width = img.shape[:2]

    # randomly select a batch of image and use the mean as boarder color
    top,bottom = np.sort(np.random.choice(height, 2))
    left, right = np.sort(np.random.choice(width,2))
    patch = img[top:bottom,left:right]

    if random_border_color == True:
        mean = cv2.mean(patch)[0]
    else:
        ## black border
        mean = 0

    border_top = 0
    border_bottom = 0
    border_left = 0
    border_right = 0

    # add top and bottom boarder to the image
    if height < size:
        border_top = int((size - height)/2)
        border_bottom = size - height - border_top

    # add left/right boarder to the image
    if width < size:
        border_left = int((size - width)/2)
        border_right = size - width - border_left

    # copy and add border
    border = cv2.copyMakeBorder(img, top=border_top, bottom=border_bottom, left=border_left, right=border_right,
                                borderType=cv2.BORDER_CONSTANT, value=[mean, mean, mean])

    return border

def preprocess_image(input_path, output_path, crop_dim, debug = False):
    """
    Detect face, align and crop
    :param input_path: Path to input image
    :param output_path: Path to write processed image
    :param crop_dim: dimensions to crop image to
    :return numpy int array of (crop_dim, crop_dim, 3)
    """
    image = _process_image(input_path, crop_dim)

    if debug == True:
        if image is not None:
            logger.debug('Writing processed file: {}'.format(output_path))
            cv2.imwrite(output_path, image)
        else:
            logger.warning("Skipping filename: {}".format(input_path))

    return image

def _process_image(filename, crop_dim):
    image = None
    aligned_image = None

    image = _buffer_image(filename)

    if image is not None:
        aligned_image = _align_image(image, crop_dim)
    else:
        raise IOError('Error buffering image: {}'.format(filename))

    return aligned_image


def _buffer_image(filename):
    logger.debug('Reading image: {}'.format(filename))
    image = cv2.imread(filename, )
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def _align_image(image, crop_dim):
    bb = align_dlib.getLargestFaceBoundingBox(image)
    if bb is None:
        logger.info("No face detected")
    aligned = align_dlib.align(crop_dim, image, bb, landmarkIndices=AlignDlib.INNER_EYES_AND_BOTTOM_LIP)
    if aligned is not None:
        aligned = cv2.cvtColor(aligned, cv2.COLOR_BGR2RGB)
    else:
        logger.info("No aligned image returned ")
    return aligned


if __name__ == '__main__':
    import argparse
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument('--input-dir', type=str, action='store', default='data', dest='input_dir')
    parser.add_argument('--output-dir', type=str, action='store', default='output', dest='output_dir')
    parser.add_argument('--crop-dim', type=int, action='store', default=180, dest='crop_dim',
                        help='Size to crop images to')

    args = parser.parse_args()

    test_process(args.input_dir, args.output_dir, args.crop_dim)
    #test_resize(args.input_dir, args.output_dir, 224)
