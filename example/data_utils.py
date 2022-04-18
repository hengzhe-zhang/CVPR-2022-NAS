# encoding: utf-8
"""
Paddle-Paddle的原始处理工具
"""
import cv2
import numpy as np
from PIL import Image, ImageOps

from example.file_io import PathManager


def image_preprocess(im, input_size):
    """ image_preprocess """
    if isinstance(input_size, list) or isinstance(input_size, tuple):
        input_size = input_size[0]
    im_shape = im.shape
    im_size_min = np.min(im_shape[0:2])
    im_size_max = np.max(im_shape[0:2])
    im_scale = float(input_size) / float(im_size_max)
    new_h = int(im_shape[0] * im_scale)
    new_w = int(im_shape[1] * im_scale)
    im = cv2.resize(im, (new_w, new_h), interpolation=cv2.INTER_AREA)
    im_padded = np.zeros((input_size, input_size, 3), dtype=np.float32)
    im_padded[:new_h, :new_w, :] = im
    # im_padded = im_padded.transpose((2, 0, 1))
    return im_padded, im_scale


def read_image(file_name, format=None):
    """
    Read an image into the given format.
    Will apply rotation and flipping if the image has such exif information.

    Args:
        file_name (str): image file path
        format (str): one of the supported image modes in PIL, or "BGR"
    Returns:
        image (np.ndarray): an HWC image
    """
    with PathManager.open(file_name, "rb") as f:
        image = Image.open(f)

        # work around this bug: https://github.com/python-pillow/Pillow/issues/3973
        try:
            image = ImageOps.exif_transpose(image)
        except Exception:
            pass

        if format is not None:
            # PIL only supports RGB, so convert to RGB and flip channels over below
            conversion_format = format
            if format == "BGR":
                conversion_format = "RGB"
            image = image.convert(conversion_format)
        image = np.asarray(image)

        # PIL squeezes out the channel dimension for "L", so make it HWC
        if format == "L":
            image = np.expand_dims(image, -1)

        # handle formats not supported by PIL
        elif format == "BGR":
            # flip channels if needed
            image = image[:, :, ::-1]

        # handle grayscale mixed in RGB images
        elif len(image.shape) == 2:
            image = np.repeat(image[..., np.newaxis], 3, axis=-1)

        image = Image.fromarray(image)

        return image
