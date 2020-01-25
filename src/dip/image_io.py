import cv2
import numpy as np
import pathlib


def load_mask(path: str) -> np.ndarray:
    """Load a image file representing a mask.

    The outputted mask uses 1 to represent the masking
    bits.

    If pixel in the input has any RGB component with a 
    value of 1 or more, then the pixel will be considered
    a mask bit.

    Args:
        path: the file path to the mask image file to load.

    Returns: 
        a 2D array with either 0 or 1 entries.
    """
    mask = cv2.imread(path)
    # Only use 1 channel.
    mask = mask[:, :, 0:1]
    # Make mask elements either 0 or 1.
    mask = np.clip(mask, 0, 1)
    return mask


def load_img(img_path):
    img = cv2.imread(img_path)
    # Transform to range [-1, 1].
    img = img / 127.5 - 1.
    return img


def load_and_mask_img(img_path, mask):
    img = load_img(img_path)
    # Delete masked area.
    img = img * (1.0 - mask[0])
    return img


def print_img(normalized_img, out_file_path):
    """Denormalize and print an image to file.

    The input image data will:
        * be clipped to [-1, 1]
        * transformed to [0, 2]
        * scaled to [0, 256]
    Args:
        normalized_img (numpy.array): numpy array with shape (X, X, 3).
    """
    img = np.clip(normalized_img, -1, 1)
    img = (img + 1.0) * 127.5
    parent_dir = pathlib.Path(out_file_path).parent
    parent_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(out_file_path, img)


def dbg_img(img):
    cv2.imwrite('./debug_out.png', img)


def apply_gaussian_noise(img):
    if np.any((img < 1.0)|(img > 1.0)):
        raise Exception('image needs to be '
            'normalized to [-1, 1].')
    raise NotImplementedError()


