import cv2
import torch
import numpy as np

from .misc_util import is_tuple_of


def imconvert(img, src, dst):
    """Convert an image from the src colorspace to dst colorspace.
    Args:
        img (ndarray): The input image.
        src (str): The source colorspace, e.g., 'rgb', 'hsv'.
        dst (str): The destination colorspace, e.g., 'rgb', 'hsv'.
    Returns:
        ndarray: The converted image.
    """
    code = getattr(cv2, f"COLOR_{src.upper()}2{dst.upper()}")
    out_img = cv2.cvtColor(img, code)
    return out_img


def bgr2gray(img, keepdim=False):
    """Convert a BGR image to grayscale image.
    Args:
        img (ndarray): The input image.
        keepdim (bool): If False (by default), then return the grayscale image
            with 2 dims, otherwise 3 dims.
    Returns:
        ndarray: The converted grayscale image.
    """
    out_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if keepdim:
        out_img = out_img[..., None]
    return out_img


def rgb2gray(img, keepdim=False):
    """Convert a RGB image to grayscale image.
    Args:
        img (ndarray): The input image.
        keepdim (bool): If False (by default), then return the grayscale image
            with 2 dims, otherwise 3 dims.
    Returns:
        ndarray: The converted grayscale image.
    """
    out_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    if keepdim:
        out_img = out_img[..., None]
    return out_img


def gray2bgr(img):
    """Convert a grayscale image to BGR image.
    Args:
        img (ndarray): The input image.
    Returns:
        ndarray: The converted BGR image.
    """
    img = img[..., None] if img.ndim == 2 else img
    out_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return out_img


def gray2rgb(img):
    """Convert a grayscale image to RGB image.
    Args:
        img (ndarray): The input image.
    Returns:
        ndarray: The converted RGB image.
    """
    img = img[..., None] if img.ndim == 2 else img
    out_img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    return out_img


def _convert_input_type_range(img):
    """Convert the type and range of the input image.
    It converts the input image to np.float32 type and range of [0, 1].
    It is mainly used for pre-processing the input image in colorspace
    convertion functions such as rgb2ycbcr and ycbcr2rgb.
    Args:
        img (ndarray): The input image. It accepts:
            1. np.uint8 type with range [0, 255];
            2. np.float32 type with range [0, 1].
    Returns:
        (ndarray): The converted image with type of np.float32 and range of
            [0, 1].
    """
    img_type = img.dtype
    img = img.astype(np.float32)
    if img_type == np.float32:
        pass
    elif img_type == np.uint8:
        img /= 255.0
    else:
        raise TypeError(
            "The img type should be np.float32 or np.uint8, " f"but got {img_type}"
        )
    return img


def _convert_output_type_range(img, dst_type):
    """Convert the type and range of the image according to dst_type.
    It converts the image to desired type and range. If `dst_type` is np.uint8,
    images will be converted to np.uint8 type with range [0, 255]. If
    `dst_type` is np.float32, it converts the image to np.float32 type with
    range [0, 1].
    It is mainly used for post-processing images in colorspace convertion
    functions such as rgb2ycbcr and ycbcr2rgb.
    Args:
        img (ndarray): The image to be converted with np.float32 type and
            range [0, 255].
        dst_type (np.uint8 | np.float32): If dst_type is np.uint8, it
            converts the image to np.uint8 type with range [0, 255]. If
            dst_type is np.float32, it converts the image to np.float32 type
            with range [0, 1].
    Returns:
        (ndarray): The converted image with desired type and range.
    """
    if dst_type not in (np.uint8, np.float32):
        raise TypeError(
            "The dst_type should be np.float32 or np.uint8, " f"but got {dst_type}"
        )
    if dst_type == np.uint8:
        img = img.round()
    else:
        img /= 255.0
    return img.astype(dst_type)


def rgb2ycbcr(img, y_only=False):
    """Convert a RGB image to YCbCr image.
    This function produces the same results as Matlab's `rgb2ycbcr` function.
    It implements the ITU-R BT.601 conversion for standard-definition
    television. See more details in
    https://en.wikipedia.org/wiki/YCbCr#ITU-R_BT.601_conversion.
    It differs from a similar function in cv2.cvtColor: `RGB <-> YCrCb`.
    In OpenCV, it implements a JPEG conversion. See more details in
    https://en.wikipedia.org/wiki/YCbCr#JPEG_conversion.
    Args:
        img (ndarray): The input image. It accepts:
            1. np.uint8 type with range [0, 255];
            2. np.float32 type with range [0, 1].
        y_only (bool): Whether to only return Y channel. Default: False.
    Returns:
        ndarray: The converted YCbCr image. The output image has the same type
            and range as input image.
    """
    img_type = img.dtype
    img = _convert_input_type_range(img)
    if y_only:
        out_img = np.dot(img, [65.481, 128.553, 24.966]) + 16.0
    else:
        out_img = (
            np.matmul(
                img,
                [
                    [65.481, -37.797, 112.0],
                    [128.553, -74.203, -93.786],
                    [24.966, 112.0, -18.214],
                ],
            )
            + [16, 128, 128]
        )
    out_img = _convert_output_type_range(out_img, img_type)
    return out_img


def bgr2ycbcr(img, y_only=False):
    """Convert a BGR image to YCbCr image.
    The bgr version of rgb2ycbcr.
    It implements the ITU-R BT.601 conversion for standard-definition
    television. See more details in
    https://en.wikipedia.org/wiki/YCbCr#ITU-R_BT.601_conversion.
    It differs from a similar function in cv2.cvtColor: `BGR <-> YCrCb`.
    In OpenCV, it implements a JPEG conversion. See more details in
    https://en.wikipedia.org/wiki/YCbCr#JPEG_conversion.
    Args:
        img (ndarray): The input image. It accepts:
            1. np.uint8 type with range [0, 255];
            2. np.float32 type with range [0, 1].
        y_only (bool): Whether to only return Y channel. Default: False.
    Returns:
        ndarray: The converted YCbCr image. The output image has the same type
            and range as input image.
    """
    img_type = img.dtype
    img = _convert_input_type_range(img)
    if y_only:
        out_img = np.dot(img, [24.966, 128.553, 65.481]) + 16.0
    else:
        out_img = (
            np.matmul(
                img,
                [
                    [24.966, 112.0, -18.214],
                    [128.553, -74.203, -93.786],
                    [65.481, -37.797, 112.0],
                ],
            )
            + [16, 128, 128]
        )
    out_img = _convert_output_type_range(out_img, img_type)
    return out_img


def ycbcr2rgb(img):
    """Convert a YCbCr image to RGB image.
    This function produces the same results as Matlab's ycbcr2rgb function.
    It implements the ITU-R BT.601 conversion for standard-definition
    television. See more details in
    https://en.wikipedia.org/wiki/YCbCr#ITU-R_BT.601_conversion.
    It differs from a similar function in cv2.cvtColor: `YCrCb <-> RGB`.
    In OpenCV, it implements a JPEG conversion. See more details in
    https://en.wikipedia.org/wiki/YCbCr#JPEG_conversion.
    Args:
        img (ndarray): The input image. It accepts:
            1. np.uint8 type with range [0, 255];
            2. np.float32 type with range [0, 1].
    Returns:
        ndarray: The converted RGB image. The output image has the same type
            and range as input image.
    """
    img_type = img.dtype
    img = _convert_input_type_range(img) * 255
    out_img = (
        np.matmul(
            img,
            [
                [0.00456621, 0.00456621, 0.00456621],
                [0, -0.00153632, 0.00791071],
                [0.00625893, -0.00318811, 0],
            ],
        )
        * 255.0
        + [-222.921, 135.576, -276.836]
    )
    out_img = _convert_output_type_range(out_img, img_type)
    return out_img


def ycbcr2bgr(img):
    """Convert a YCbCr image to BGR image.
    The bgr version of ycbcr2rgb.
    It implements the ITU-R BT.601 conversion for standard-definition
    television. See more details in
    https://en.wikipedia.org/wiki/YCbCr#ITU-R_BT.601_conversion.
    It differs from a similar function in cv2.cvtColor: `YCrCb <-> BGR`.
    In OpenCV, it implements a JPEG conversion. See more details in
    https://en.wikipedia.org/wiki/YCbCr#JPEG_conversion.
    Args:
        img (ndarray): The input image. It accepts:
            1. np.uint8 type with range [0, 255];
            2. np.float32 type with range [0, 1].
    Returns:
        ndarray: The converted BGR image. The output image has the same type
            and range as input image.
    """
    img_type = img.dtype
    img = _convert_input_type_range(img) * 255
    out_img = (
        np.matmul(
            img,
            [
                [0.00456621, 0.00456621, 0.00456621],
                [0.00791071, -0.00153632, 0],
                [0, -0.00318811, 0.00625893],
            ],
        )
        * 255.0
        + [-276.836, 135.576, -222.921]
    )
    out_img = _convert_output_type_range(out_img, img_type)
    return out_img


def convert_color_factory(src, dst):

    code = getattr(cv2, f"COLOR_{src.upper()}2{dst.upper()}")

    def convert_color(img):
        out_img = cv2.cvtColor(img, code)
        return out_img

    convert_color.__doc__ = f"""Convert a {src.upper()} image to {dst.upper()}
        image.
    Args:
        img (ndarray or str): The input image.
    Returns:
        ndarray: The converted {dst.upper()} image.
    """

    return convert_color


def imnormalize(img, mean, std, to_rgb=True):
    """Normalize an image with mean and std.
    Args:
        img (ndarray): Image to be normalized.
        mean (ndarray): The mean to be used for normalize.
        std (ndarray): The std to be used for normalize.
        to_rgb (bool): Whether to convert to rgb.
    Returns:
        ndarray: The normalized image.
    """
    img = img.copy().astype(np.float32)
    return imnormalize_(img, mean, std, to_rgb)


def imnormalize_(img, mean, std, to_rgb=True):
    """Inplace normalize an image with mean and std.
    Args:
        img (ndarray): Image to be normalized.
        mean (ndarray): The mean to be used for normalize.
        std (ndarray): The std to be used for normalize.
        to_rgb (bool): Whether to convert to rgb.
    Returns:
        ndarray: The normalized image.
    """
    # cv2 inplace normalization does not accept uint8
    assert img.dtype != np.uint8
    mean = np.float64(mean.reshape(1, -1))
    stdinv = 1 / np.float64(std.reshape(1, -1))
    if to_rgb:
        cv2.cvtColor(img, cv2.COLOR_BGR2RGB, img)  # inplace
    cv2.subtract(img, mean, img)  # inplace
    cv2.multiply(img, stdinv, img)  # inplace
    return img


def imdenormalize(img, mean, std, to_bgr=True):
    assert img.dtype != np.uint8
    mean = mean.reshape(1, -1).astype(np.float64)
    std = std.reshape(1, -1).astype(np.float64)
    img = cv2.multiply(img, std)  # make a copy
    cv2.add(img, mean, img)  # inplace
    if to_bgr:
        cv2.cvtColor(img, cv2.COLOR_RGB2BGR, img)  # inplace
    return img


def iminvert(img):
    """Invert (negate) an image.
    Args:
        img (ndarray): Image to be inverted.
    Returns:
        ndarray: The inverted image.
    """
    return np.full_like(img, 255) - img


def solarize(img, thr=128):
    """Solarize an image (invert all pixel values above a threshold)
    Args:
        img (ndarray): Image to be solarized.
        thr (int): Threshold for solarizing (0 - 255).
    Returns:
        ndarray: The solarized image.
    """
    img = np.where(img < thr, img, 255 - img)
    return img


def posterize(img, bits):
    """Posterize an image (reduce the number of bits for each color channel)
    Args:
        img (ndarray): Image to be posterized.
        bits (int): Number of bits (1 to 8) to use for posterizing.
    Returns:
        ndarray: The posterized image.
    """
    shift = 8 - bits
    img = np.left_shift(np.right_shift(img, shift), shift)
    return img


def adjust_color(img, alpha=1, beta=None, gamma=0):
    """It blends the source image and its gray image:
    ``output = img * alpha + gray_img * beta + gamma``
    Args:
        img (ndarray): The input source image.
        alpha (int | float): Weight for the source image. Default 1.
        beta (int | float): Weight for the converted gray image.
            If None, it's assigned the value (1 - `alpha`).
        gamma (int | float): Scalar added to each sum.
            Same as :func:`cv2.addWeighted`. Default 0.
    Returns:
        ndarray: Colored image which has the same size and dtype as input.
    """
    gray_img = bgr2gray(img)
    gray_img = np.tile(gray_img[..., None], [1, 1, 3])
    if beta is None:
        beta = 1 - alpha
    colored_img = cv2.addWeighted(img, alpha, gray_img, beta, gamma)
    if not colored_img.dtype == np.uint8:
        # Note when the dtype of `img` is not defaultly `np.uint8`
        # (e.g. np.float32), the value in `colored_img` got from cv2
        # is not guaranteed to be in range [0, 255], so here clip
        # is needed.
        colored_img = np.clip(colored_img, 0, 255)
    return colored_img


def imequalize(img):
    """Equalize the image histogram.

    This function applies a non-linear mapping to the input image,
    in order to create a uniform distribution of grayscale values
    in the output image.
    Args:
        img (ndarray): Image to be equalized.
    Returns:
        ndarray: The equalized image.
    """

    def _scale_channel(im, c):
        """Scale the data in the corresponding channel."""
        im = im[:, :, c]
        # Compute the histogram of the image channel.
        histo = np.histogram(im, 256, (0, 255))[0]
        # For computing the step, filter out the nonzeros.
        nonzero_histo = histo[histo > 0]
        step = (np.sum(nonzero_histo) - nonzero_histo[-1]) // 255
        if not step:
            lut = np.array(range(256))
        else:
            # Compute the cumulative sum, shifted by step // 2
            # and then normalized by step.
            lut = (np.cumsum(histo) + (step // 2)) // step
            # Shift lut, prepending with 0.
            lut = np.concatenate([[0], lut[:-1]], 0)
        # If step is zero, return the original image.
        # Otherwise, index from lut.
        return np.where(np.equal(step, 0), im, lut[im])

    # Scales each channel independently and then stacks
    # the result.
    s1 = _scale_channel(img, 0)
    s2 = _scale_channel(img, 1)
    s3 = _scale_channel(img, 2)
    equalized_img = np.stack([s1, s2, s3], axis=-1)
    return equalized_img


def adjust_brightness(img, factor=1.0):
    """Adjust image brightness.
    This function controls the brightness of an image. An
    enhancement factor of 0.0 gives a black image.
    A factor of 1.0 gives the original image. This function
    blends the source image and the degenerated black image:
    ``output = img * factor + degenerated * (1 - factor)``
    Args:
        img (ndarray): Image to be brightened.
        factor (float): A value controls the enhancement.
            Factor 1.0 returns the original image, lower
            factors mean less color (brightness, contrast,
            etc), and higher values more. Default 1.
    Returns:
        ndarray: The brightened image.
    """
    degenerated = np.zeros_like(img)
    # Note manually convert the dtype to np.float32, to
    # achieve as close results as PIL.ImageEnhance.Brightness.
    # Set beta=1-factor, and gamma=0
    brightened_img = cv2.addWeighted(
        img.astype(np.float32), factor, degenerated.astype(np.float32), 1 - factor, 0
    )
    return brightened_img.astype(img.dtype)


def adjust_contrast(img, factor=1.0):
    """Adjust image contrast.
    This function controls the contrast of an image. An
    enhancement factor of 0.0 gives a solid grey
    image. A factor of 1.0 gives the original image. It
    blends the source image and the degenerated mean image:
    ``output = img * factor + degenerated * (1 - factor)``
    Args:
        img (ndarray): Image to be contrasted. BGR order.
        factor (float): Same as :func: `adjust_brightness`.
    Returns:
        ndarray: The contrasted image.
    """
    gray_img = bgr2gray(img)
    hist = np.histogram(gray_img, 256, (0, 255))[0]
    mean = round(np.sum(gray_img) / np.sum(hist))
    degenerated = (np.ones_like(img[..., 0]) * mean).astype(img.dtype)
    degenerated = gray2bgr(degenerated)
    contrasted_img = cv2.addWeighted(
        img.astype(np.float32), factor, degenerated.astype(np.float32), 1 - factor, 0
    )
    return contrasted_img.astype(img.dtype)


def lut_transform(img, lut_table):
    """Transform array by look-up table.
    The function lut_transform fills the output array with values from the
    look-up table. Indices of the entries are taken from the input array.
    Args:
        img (ndarray): Image to be transformed.
        lut_table (ndarray): look-up table of 256 elements; in case of
            multi-channel input array, the table should either have a single
            channel (in this case the same table is used for all channels) or
            the same number of channels as in the input array.
    Returns:
        ndarray: The transformed image.
    """
    assert isinstance(img, np.ndarray)
    assert 0 <= np.min(img) and np.max(img) <= 255
    assert isinstance(lut_table, np.ndarray)
    assert lut_table.shape == (256,)

    return cv2.LUT(np.array(img, dtype=np.uint8), lut_table)


def clahe(img, clip_limit=40.0, tile_grid_size=(8, 8)):
    """Use CLAHE method to process the image.
    See `ZUIDERVELD,K. Contrast Limited Adaptive Histogram Equalization[J].
    Graphics Gems, 1994:474-485.` for more information.
    Args:
        img (ndarray): Image to be processed.
        clip_limit (float): Threshold for contrast limiting. Default: 40.0.
        tile_grid_size (tuple[int]): Size of grid for histogram equalization.
            Input image will be divided into equally sized rectangular tiles.
            It defines the number of tiles in row and column. Default: (8, 8).
    Returns:
        ndarray: The processed image.
    """
    assert isinstance(img, np.ndarray)
    assert img.ndim == 2
    assert isinstance(clip_limit, (float, int))
    assert is_tuple_of(tile_grid_size, int)
    assert len(tile_grid_size) == 2

    clahe = cv2.createCLAHE(clip_limit, tile_grid_size)
    return clahe.apply(np.array(img, dtype=np.uint8))


def tensor2imgs(tensor, mean=(0, 0, 0), std=(1, 1, 1), to_rgb=True):
    """Convert tensor to 3-channel images.
    Args:
        tensor (torch.Tensor): Tensor that contains multiple images, shape (
            N, C, H, W).
        mean (tuple[float], optional): Mean of images. Defaults to (0, 0, 0).
        std (tuple[float], optional): Standard deviation of images.
            Defaults to (1, 1, 1).
        to_rgb (bool, optional): Whether the tensor was converted to RGB
            format in the first place. If so, convert it back to BGR.
            Defaults to True.
    Returns:
        list[np.ndarray]: A list that contains multiple images.
    """

    if torch is None:
        raise RuntimeError("pytorch is not installed")
    assert torch.is_tensor(tensor) and tensor.ndim == 4
    assert len(mean) == 3
    assert len(std) == 3

    num_imgs = tensor.size(0)
    mean = np.array(mean, dtype=np.float32)
    std = np.array(std, dtype=np.float32)
    imgs = []
    for img_id in range(num_imgs):
        img = tensor[img_id, ...].cpu().numpy().transpose(1, 2, 0)
        img = imdenormalize(img, mean, std, to_bgr=to_rgb).astype(np.uint8)
        imgs.append(np.ascontiguousarray(img))
    return imgs


bgr2rgb = convert_color_factory("bgr", "rgb")

rgb2bgr = convert_color_factory("rgb", "bgr")

bgr2hsv = convert_color_factory("bgr", "hsv")

hsv2bgr = convert_color_factory("hsv", "bgr")

bgr2hls = convert_color_factory("bgr", "hls")

hls2bgr = convert_color_factory("hls", "bgr")
