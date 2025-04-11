import numpy as np


def resize_x0p75(img: np.ndarray) -> np.ndarray:
    """
       Image Downscaling x0.75 
    """

    assert img.dtype == np.float32 or img.dtype == np.float64, "invalid dtype"
    assert img.ndim == 2 or img.ndim == 3, "invalid shape"

    dtype = img.dtype

    if img.ndim == 2:
        h, w = img.shape

        img_x = np.empty((h, w * 3 // 4), dtype)
        img_y = np.empty((h * 3 // 4, w * 3 // 4), dtype)
    else:
        h, w, c = img.shape

        img_x = np.empty((h, w * 3 // 4, c), dtype)
        img_y = np.empty((h * 3 // 4, w * 3 // 4, c), dtype)

    assert (w % 4) == 0 and (h % 4) == 0, "invalid size"

    img_x[:, 0::3] = img[:, 0::4] * 3 + img[:, 1::4]
    img_x[:, 1::3] = (img[:, 1::4] + img[:, 2::4]) * 2
    img_x[:, 2::3] = img[:, 2::4] + img[:, 3::4] * 3

    img_y[0::3, :] = img_x[0::4, :] * 3 + img_x[1::4, :]
    img_y[1::3, :] = (img_x[1::4, :] + img_x[2::4, :]) * 2
    img_y[2::3, :] = img_x[2::4, :] + img_x[3::4, :] * 3

    img_y /= 16

    return img_y


def resize_x0p5(img: np.ndarray) -> np.ndarray:
    """
       Image Downscaling x0.5 
    """

    assert img.dtype == np.float32 or img.dtype == np.float64, "invalid dtype"
    assert img.ndim == 2 or img.ndim == 3, "invalid shape"

    h, w = img.shape[:2]

    assert (w % 2) == 0 and (h % 2) == 0, "invalid size"

    img_resize = img[0::2, 0::2] + img[1::2, 0::2] + img[0::2, 1::2] + img[1::2, 1::2]

    img_resize /= 4

    return img_resize


def resize_x0p25(img: np.ndarray) -> np.ndarray:
    """
       Image Downscaling x0.25 
    """

    img_resize = resize_x0p5(resize_x0p5(img))

    return img_resize