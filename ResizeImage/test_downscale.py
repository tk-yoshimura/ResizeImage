import numpy as np
import cv2
from downscale import resize_x0p75, resize_x0p5, resize_x0p25

import unittest

class Test_downscale(unittest.TestCase):
    def test_resize_x0p75_withchannel_float32(self):
        img = np.zeros((64, 48, 3), dtype=np.uint8)
        img = cv2.circle(img, (24, 16), 20, color=(255, 128, 64), thickness=-1)
        img = cv2.circle(img, (24, 48), 16, color=(128, 64, 255), thickness=-1)

        img_resize = resize_x0p75(img.astype(np.float32))
        img_cv2_resize = cv2.resize(img.astype(np.float32), (36, 48), interpolation=cv2.INTER_AREA)

        np.testing.assert_allclose(img_resize, img_cv2_resize)

    def test_resize_x0p75_withchannel_float64(self):
        img = np.zeros((64, 48, 3), dtype=np.uint8)
        img = cv2.circle(img, (24, 16), 20, color=(255, 128, 64), thickness=-1)
        img = cv2.circle(img, (24, 48), 16, color=(128, 64, 255), thickness=-1)

        img_resize = resize_x0p75(img.astype(np.float64))
        img_cv2_resize = cv2.resize(img.astype(np.float64), (36, 48), interpolation=cv2.INTER_AREA)

        np.testing.assert_allclose(img_resize, img_cv2_resize)

    def test_resize_x0p75_withoutchannel_float32(self):
        img = np.zeros((64, 48), dtype=np.uint8)
        img = cv2.circle(img, (24, 16), 20, color=255, thickness=-1)
        img = cv2.circle(img, (24, 48), 16, color=64, thickness=-1)

        img_resize = resize_x0p75(img.astype(np.float32))
        img_cv2_resize = cv2.resize(img.astype(np.float32), (36, 48), interpolation=cv2.INTER_AREA)

        np.testing.assert_allclose(img_resize, img_cv2_resize)

    def test_resize_x0p75_withoutchannel_float64(self):
        img = np.zeros((64, 48), dtype=np.uint8)
        img = cv2.circle(img, (24, 16), 20, color=255, thickness=-1)
        img = cv2.circle(img, (24, 48), 16, color=64, thickness=-1)

        img_resize = resize_x0p75(img.astype(np.float64))
        img_cv2_resize = cv2.resize(img.astype(np.float64), (36, 48), interpolation=cv2.INTER_AREA)

        np.testing.assert_allclose(img_resize, img_cv2_resize)

    def test_resize_x0p5_withchannel_float32(self):
        img = np.zeros((64, 48, 3), dtype=np.uint8)
        img = cv2.circle(img, (24, 16), 20, color=(255, 128, 64), thickness=-1)
        img = cv2.circle(img, (24, 48), 16, color=(128, 64, 255), thickness=-1)

        img_resize = resize_x0p5(img.astype(np.float32))
        img_cv2_resize = cv2.resize(img.astype(np.float32), (24, 32), interpolation=cv2.INTER_AREA)

        np.testing.assert_allclose(img_resize, img_cv2_resize)

    def test_resize_x0p5_withchannel_float64(self):
        img = np.zeros((64, 48, 3), dtype=np.uint8)
        img = cv2.circle(img, (24, 16), 20, color=(255, 128, 64), thickness=-1)
        img = cv2.circle(img, (24, 48), 16, color=(128, 64, 255), thickness=-1)

        img_resize = resize_x0p5(img.astype(np.float64))
        img_cv2_resize = cv2.resize(img.astype(np.float64), (24, 32), interpolation=cv2.INTER_AREA)

        np.testing.assert_allclose(img_resize, img_cv2_resize)

    def test_resize_x0p5_withoutchannel_float32(self):
        img = np.zeros((64, 48), dtype=np.uint8)
        img = cv2.circle(img, (24, 16), 20, color=255, thickness=-1)
        img = cv2.circle(img, (24, 48), 16, color=64, thickness=-1)

        img_resize = resize_x0p5(img.astype(np.float32))
        img_cv2_resize = cv2.resize(img.astype(np.float32), (24, 32), interpolation=cv2.INTER_AREA)

        np.testing.assert_allclose(img_resize, img_cv2_resize)

    def test_resize_x0p5_withoutchannel_float64(self):
        img = np.zeros((64, 48), dtype=np.uint8)
        img = cv2.circle(img, (24, 16), 20, color=255, thickness=-1)
        img = cv2.circle(img, (24, 48), 16, color=64, thickness=-1)

        img_resize = resize_x0p5(img.astype(np.float64))
        img_cv2_resize = cv2.resize(img.astype(np.float64), (24, 32), interpolation=cv2.INTER_AREA)

        np.testing.assert_allclose(img_resize, img_cv2_resize)

    def test_resize_x0p25_withchannel_float32(self):
        img = np.zeros((64, 48, 3), dtype=np.uint8)
        img = cv2.circle(img, (24, 16), 20, color=(255, 128, 64), thickness=-1)
        img = cv2.circle(img, (24, 48), 16, color=(128, 64, 255), thickness=-1)

        img_resize = resize_x0p25(img.astype(np.float32))
        img_cv2_resize = cv2.resize(img.astype(np.float32), (12, 16), interpolation=cv2.INTER_AREA)

        np.testing.assert_allclose(img_resize, img_cv2_resize)

    def test_resize_x0p25_withchannel_float64(self):
        img = np.zeros((64, 48, 3), dtype=np.uint8)
        img = cv2.circle(img, (24, 16), 20, color=(255, 128, 64), thickness=-1)
        img = cv2.circle(img, (24, 48), 16, color=(128, 64, 255), thickness=-1)

        img_resize = resize_x0p25(img.astype(np.float64))
        img_cv2_resize = cv2.resize(img.astype(np.float64), (12, 16), interpolation=cv2.INTER_AREA)

        np.testing.assert_allclose(img_resize, img_cv2_resize)

    def test_resize_x0p25_withoutchannel_float32(self):
        img = np.zeros((64, 48), dtype=np.uint8)
        img = cv2.circle(img, (24, 16), 20, color=255, thickness=-1)
        img = cv2.circle(img, (24, 48), 16, color=64, thickness=-1)

        img_resize = resize_x0p25(img.astype(np.float32))
        img_cv2_resize = cv2.resize(img.astype(np.float32), (12, 16), interpolation=cv2.INTER_AREA)

        np.testing.assert_allclose(img_resize, img_cv2_resize)

    def test_resize_x0p25_withoutchannel_float64(self):
        img = np.zeros((64, 48), dtype=np.uint8)
        img = cv2.circle(img, (24, 16), 20, color=255, thickness=-1)
        img = cv2.circle(img, (24, 48), 16, color=64, thickness=-1)

        img_resize = resize_x0p25(img.astype(np.float64))
        img_cv2_resize = cv2.resize(img.astype(np.float64), (12, 16), interpolation=cv2.INTER_AREA)

        np.testing.assert_allclose(img_resize, img_cv2_resize)

if __name__ == '__main__':
    unittest.main()
