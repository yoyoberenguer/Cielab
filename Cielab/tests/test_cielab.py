
"""
                 GNU GENERAL PUBLIC LICENSE
                       Version 3, 29 June 2007

 Copyright (C) 2007 Free Software Foundation, Inc. <https://fsf.org/>
 Everyone is permitted to copy and distribute verbatim copies
 of this license document, but changing it is not allowed.

Copyright Yoann Berenguer
"""

import sys
import unittest

import numpy
import Cielab

try:
    from Cielab import rgb_to_xyz, xyz_to_rgb, xyz_adobe98, \
        adobe98_xyz, array3d_mean, xyz_to_cielab, cielab_to_xyz, rgb_to_cielab, \
        cielab_to_rgb, rgb2adobe, rgb_2_cielab, cielab_2_rgb, rgb2adobe_inplace, \
        WhiteBalance, adobe2rgb, array3d_stats, WhiteBalanceInplace

except ImportError:
    raise ImportError(
        "\n<Cielab> library is missing on your system."
        "\nTry: \n   C:\\pip install Cielab on a window command prompt.")

import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"

try:
    import pygame
except ImportError:
    raise ImportError(
        "\n<pygame> library is missing on your system."
        "\nTry: \n   C:\\pip install pygame on a window command prompt.")


try:
    import numpy as np
except ImportError:
    raise ImportError(
        "\n<numpy> library is missing on your system."
        "\nTry: \n   C:\\pip install numpy on a window command prompt.")

# todo test 24 and 32 bit images

PROJECT_PATH = list(Cielab.__path__)
os.chdir(PROJECT_PATH[0] + "/tests")

WIDTH = 1280
HEIGHT = 1024
SCREEN = pygame.display.set_mode((WIDTH, HEIGHT))


class array3d_mean_t(unittest.TestCase):
    """

    """

    # pylint: disable=too-many-statements
    def runTest(self) -> None:
        """

        :return:  void
        """

        self.assertRaises(TypeError, array3d_mean)

        rgb = np.zeros((10, 10, 3), dtype=np.uint8)
        arr = np.full_like(rgb, 25, dtype=np.uint8)
        self.assertIsInstance(array3d_mean(arr), tuple)
        # Check the tuple length must be 4 (r, g, b, c)
        self.assertEqual(len(array3d_mean(arr)), 4)
        r, g, b, c = array3d_mean(arr)
        self.assertTupleEqual((r, g, b, c), (25, 25, 25, 100))
        arr[:, :, :] = 10, 255, 15
        r, g, b, c = array3d_mean(arr)
        self.assertTupleEqual((r, g, b, c), (10, 255, 15, 100))

        rgb = np.zeros((10, 10, 3), dtype=np.float32)
        arr = np.full_like(rgb, 18, dtype=np.float32)
        r, g, b, c = array3d_mean(arr)
        self.assertTupleEqual((r, g, b, c), (18, 18, 18, 100))

        rgb = np.zeros((10, 10, 3), dtype=np.int32)
        self.assertRaises(TypeError, array3d_mean, rgb)
        rgb = np.zeros((10, 10, 3), dtype=np.int8)
        self.assertRaises(TypeError, array3d_mean, rgb)

        rgb = np.zeros((10, 10, 8), dtype=np.uint8)
        self.assertRaises(TypeError, array3d_mean, rgb)
        # testing no attributes
        self.assertRaises(TypeError, array3d_mean)
        r, g, b, c = array3d_mean(arr)
        self.assertIsInstance(r, float)
        self.assertIsInstance(g, float)
        self.assertIsInstance(b, float)
        self.assertIsInstance(c, int)


class rgb_to_xyz_t(unittest.TestCase):
    """

    """

    # pylint: disable=too-many-statements
    def runTest(self) -> None:
        """

        :return:  void
        """
        r, g, b = 255, 0, 0  # XYZ = 41.246   21.267    1.933
        res = rgb_to_xyz(r, g, b)
        self.assertIsInstance(res, dict)

        self.assertAlmostEqual(res['x'], 41.246, delta=1e-3)
        self.assertAlmostEqual(res['y'], 21.267, delta=1e-3)
        self.assertAlmostEqual(res['z'], 1.933, delta=1e-3)

        # Testing out of range sRGB values
        r, g, b = 300, 256, 800
        # XYZ when rgb not limited to 255.0 values = 353.112  205.881 1369.677

        res = rgb_to_xyz(r, g, b)
        self.assertAlmostEqual(res['x'], 353.112, delta=1e-3)
        self.assertAlmostEqual(res['y'], 205.881, delta=1e-3)
        self.assertAlmostEqual(res['z'], 1369.677, delta=1e-3)

        r, g, b = 0, 255, 0  # XYZ = 35.758   71.515   11.919
        res = rgb_to_xyz(r, g, b)

        self.assertAlmostEqual(res['x'], 35.758, delta=1e-3)
        self.assertAlmostEqual(res['y'], 71.515, delta=1e-3)
        self.assertAlmostEqual(res['z'], 11.919, delta=1e-3)

        r, g, b = 0, 0, 255  # XYZ = 18.044    7.217   95.030
        res = rgb_to_xyz(r, g, b)

        self.assertAlmostEqual(res['x'], 18.044, delta=1e-3)
        self.assertAlmostEqual(res['y'], 7.217, delta=1e-3)
        self.assertAlmostEqual(res['z'], 95.030, delta=1e-3)

        r, g, b = 128, 64, 154  # 16.567   10.590   31.737
        res = rgb_to_xyz(r, g, b)

        self.assertAlmostEqual(res['x'], 16.567, delta=1e-3)
        self.assertAlmostEqual(res['y'], 10.590, delta=1e-3)
        self.assertAlmostEqual(res['z'], 31.737, delta=1e-3)

        r, g, b = 1.0, 0.5, 0.3  # 0.020    0.018    0.011
        res = rgb_to_xyz(r, g, b)

        self.assertAlmostEqual(res['x'], 0.020, delta=1e-3)
        self.assertAlmostEqual(res['y'], 0.018, delta=1e-3)
        self.assertAlmostEqual(res['z'], 0.011, delta=1e-3)

        r, g, b = -20.0, -300, -800  # no rgb limit -7.888   -8.394  -24.173
        res = rgb_to_xyz(r, g, b)

        self.assertAlmostEqual(res['x'], -7.888, delta=1e-3)
        self.assertAlmostEqual(res['y'], -8.394, delta=1e-3)
        self.assertAlmostEqual(res['z'], -24.173, delta=1e-3)

        self.assertRaises(TypeError, rgb_to_xyz, "11.0", g, b)
        self.assertRaises(TypeError, rgb_to_xyz, r, "7.0", b)
        self.assertRaises(TypeError, rgb_to_xyz, r, g, "2")

        self.assertRaises(TypeError, rgb_to_xyz)
        self.assertRaises(TypeError, rgb_to_xyz, r)
        self.assertRaises(TypeError, rgb_to_xyz, r, g)

        res = rgb_to_xyz(r, g, b)
        self.assertIsInstance(res['x'], float)
        self.assertIsInstance(res['y'], float)
        self.assertIsInstance(res['z'], float)

        self.assertEqual(len(rgb_to_xyz(r, g, b, ref='D65').values()), 3)


class xyz_to_rgb_t(unittest.TestCase):
    """

    """

    # pylint: disable=too-many-statements

    def runTest(self) -> None:
        """

        :return:  void
        """
        x, y, z = 41.246, 21.267, 1.933
        res = xyz_to_rgb(x, y, z)
        self.assertIsInstance(res, dict)

        self.assertAlmostEqual(res['r'], 255.0, delta=1e-3)
        self.assertAlmostEqual(res['g'], 0.0, delta=1e-3)
        self.assertAlmostEqual(res['b'], 0.0, delta=1e-3)

        x, y, z = 353.112, 205.881, 1369.677
        res = xyz_to_rgb(x, y, z)

        self.assertAlmostEqual(res['r'], 255.0, delta=1e-3)
        self.assertAlmostEqual(res['g'], 255.0, delta=1e-3)
        self.assertAlmostEqual(res['b'], 255.0, delta=1e-3)

        x, y, z = 35.758, 71.515, 11.919
        res = xyz_to_rgb(x, y, z)

        self.assertAlmostEqual(res['r'], 0.056, delta=1e-3)
        self.assertAlmostEqual(res['g'], 255, delta=1e-3)
        self.assertAlmostEqual(res['b'], 0, delta=1e-3)

        x, y, z = 18.044, 7.217, 95.030
        res = xyz_to_rgb(x, y, z)

        self.assertAlmostEqual(res['r'], 0.0589, delta=1e-3)
        self.assertAlmostEqual(res['g'], 0, delta=1e-3)
        self.assertAlmostEqual(res['b'], 255, delta=1e-3)

        x, y, z = 16.567, 10.590, 31.737
        res = xyz_to_rgb(x, y, z)

        self.assertAlmostEqual(res['r'], 127.995, delta=1e-3)
        self.assertAlmostEqual(res['g'], 64.006, delta=1e-3)
        self.assertAlmostEqual(res['b'], 154, delta=1e-3)

        x, y, z = 0.020, 0.018, 0.011
        res = xyz_to_rgb(x, y, z)
        self.assertAlmostEqual(res['r'], 1.042, delta=1e-3)
        self.assertAlmostEqual(res['g'], 0.488, delta=1e-3)
        self.assertAlmostEqual(res['b'], 0.298, delta=1e-3)

        res_ = rgb_to_xyz(res['r'], res['g'], res['b'])
        self.assertAlmostEqual(res_['x'], x, delta=1e-3)
        self.assertAlmostEqual(res_['y'], y, delta=1e-3)
        self.assertAlmostEqual(res_['z'], z, delta=1e-3)

        x, y, z = -7.888, -8.394, -24.173
        res = xyz_to_rgb(x, y, z)

        self.assertAlmostEqual(res['r'], 0, delta=1e-3)
        self.assertAlmostEqual(res['g'], 0, delta=1e-3)
        self.assertAlmostEqual(res['b'], 0, delta=1e-3)

        self.assertRaises(TypeError, xyz_to_rgb, "11.0", y, z)
        self.assertRaises(TypeError, xyz_to_rgb, x, "7.0", z)
        self.assertRaises(TypeError, xyz_to_rgb, x, y, "2")

        self.assertRaises(TypeError, xyz_to_rgb)
        self.assertRaises(TypeError, xyz_to_rgb, x)
        self.assertRaises(TypeError, xyz_to_rgb, x, y)

        res = xyz_to_rgb(x, y, z)
        self.assertIsInstance(res['r'], float)
        self.assertIsInstance(res['g'], float)
        self.assertIsInstance(res['b'], float)

        self.assertEqual(len(xyz_to_rgb(x, y, z, ref='D65').values()), 3)


class xyz_adobe98_t(unittest.TestCase):
    """

    """

    # pylint: disable=too-many-statements
    def runTest(self) -> None:
        """

        :return:  void
        """

        # RGB 255.0 0 0
        # XYZ 41.246 | 21.267 | 1.933
        # Adobe 218.946 0 0.048
        x, y, z = 41.246, 21.267, 1.933

        r, g, b = xyz_adobe98(x, y, z, ref='D65').values()
        xx, yy, zz = adobe98_xyz(r, g, b, ref='D65').values()

        self.assertAlmostEqual(xx, x, delta=1e-2)
        self.assertAlmostEqual(yy, y, delta=1e-2)
        self.assertAlmostEqual(zz, z, delta=1e-2)
        # Testing ref attribute must be D65 or D50
        self.assertRaises(ValueError, xyz_adobe98, x, y, z, ref='D62')
        self.assertRaises(TypeError, xyz_adobe98, "2.0", y, z, ref='D65')
        # Testing wrong type for ref attribute (must be string)
        D65 = 65
        self.assertRaises(TypeError, xyz_adobe98, x, y, z, ref=D65)
        self.assertRaises(TypeError, xyz_adobe98, ref='D65')
        self.assertRaises(TypeError, xyz_adobe98, x, ref='D65')
        self.assertRaises(TypeError, xyz_adobe98, x, y, ref='D65')

        self.assertRaises(TypeError, xyz_adobe98, x, y, ref='D655')

        res = xyz_adobe98(x, y, z, ref='D65')
        self.assertIsInstance(res, dict)

        self.assertEqual(len(xyz_adobe98(x, y, z, ref='D65').values()), 3)

        r, g, b = xyz_adobe98(x, y, z, ref='D65').values()
        self.assertIsInstance(r, float)
        self.assertIsInstance(g, float)
        self.assertIsInstance(b, float)


class adobe98_xyz_t(unittest.TestCase):
    """

    """

    # pylint: disable=too-many-statements
    def runTest(self) -> None:
        """

        :return:  void
        """
        # RGB 255.0 0 0
        # XYZ 41.246 | 21.267 | 1.933
        # Adobe 218.946 0 0.048
        x, y, z = 41.246, 21.267, 1.933
        r, g, b = xyz_adobe98(x, y, z, ref='D65').values()
        xx, yy, zz = adobe98_xyz(r, g, b, ref='D65').values()
        self.assertAlmostEqual(xx, x, delta=1e-2)
        self.assertAlmostEqual(yy, y, delta=1e-2)
        self.assertAlmostEqual(zz, z, delta=1e-2)

        self.assertRaises(TypeError, adobe98_xyz, r, g, b, ref=65)

        # Testing ref attribute must be D65 or D50
        self.assertRaises(ValueError, adobe98_xyz, r, g, b, ref='D6')

        self.assertRaises(TypeError, adobe98_xyz, "2.0", g, b, ref='D65')
        # Testing wrong type for ref attribute (must be string)
        D65 = 65
        self.assertRaises(TypeError, adobe98_xyz, r, g, b, ref=D65)
        self.assertRaises(TypeError, adobe98_xyz, ref='D65')
        self.assertRaises(TypeError, adobe98_xyz, r, ref='D65')
        self.assertRaises(TypeError, adobe98_xyz, r, g, ref='D65')

        self.assertRaises(TypeError, adobe98_xyz, r, g, ref='D655')

        xx, yy, zz = adobe98_xyz(r, g, b, ref='D65').values()
        self.assertIsInstance(xx, float)
        self.assertIsInstance(yy, float)
        self.assertIsInstance(zz, float)

        res = xyz_adobe98(x, y, z, ref='D65')
        self.assertIsInstance(res, dict)

        self.assertEqual(len(adobe98_xyz(r, g, b, ref='D65').values()), 3)


"""
# CONVERT XYZ to CIELAB
cpdef lab xyz_to_cielab(
        float x,
        float y,
        float z,
        const float [:] model=*,
        bint format_8b = *
)nogil 
"""


class xyz_CIELAB_t(unittest.TestCase):
    """

    """

    # pylint: disable=too-many-statements

    def runTest(self) -> None:
        """

        :return:  void
        """

        # RGB 255.0 0 0
        # XYZ 41.246 | 21.267 | 1.933
        # Adobe 218.946 0 0.048
        x, y, z = 41.246, 21.267, 1.933
        model_ = np.array([0.9642, 1.0000, 0.8251], dtype=np.float32)

        self.assertIsInstance(xyz_to_cielab(x, y, z, model=model_, format_8b=False), dict)

        # Test number of returned values
        self.assertEqual(len(xyz_to_cielab(x, y, z, model=model_, format_8b=False).values()), 3)

        # Check value types
        l, a, b = xyz_to_cielab(x, y, z, model=model_, format_8b=False).values()
        self.assertIsInstance(l, float)
        self.assertIsInstance(a, float)
        self.assertIsInstance(b, float)

        # todo this should raise a TypeError, but cython set True or
        #  False for incorrect type as long as the value is not null| None
        # xyz_to_cielab(x, y, z, model=model_, format_8b=12)
        # self.assertRaises(TypeError, xyz_to_cielab, x, y, z, model=model_, format_8b='True')

        # RGB 255.0 0 0
        # XYZ 41.246 | 21.267 | 1.933
        # Adobe 218.946 0 0.048
        x, y, z = 41.246, 21.267, 1.933
        cielab_model_d65 = np.array([0.9504, 1.0000, 1.0888], dtype=np.float32)
        # using D65
        l, a, b = xyz_to_cielab(x, y, z, model=cielab_model_d65, format_8b=False).values()
        # reverse method using D65
        xx, yy, zz = cielab_to_xyz(l, a, b, model=cielab_model_d65, format_8b=False).values()
        self.assertAlmostEqual(xx, x, delta=1e-3)
        self.assertAlmostEqual(yy, y, delta=1e-3)
        self.assertAlmostEqual(zz, z, delta=1e-3)

        # Using Cielab model D50
        cielab_model_d50 = np.array([0.9642, 1.0000, 0.8251], dtype=np.float32)
        l, a, b = xyz_to_cielab(x, y, z, model=cielab_model_d50, format_8b=False).values()
        # reverse method using D65
        xx, yy, zz = cielab_to_xyz(l, a, b, model=cielab_model_d50, format_8b=False).values()
        self.assertAlmostEqual(xx, x, delta=1e-3)
        self.assertAlmostEqual(yy, y, delta=1e-3)
        self.assertAlmostEqual(zz, z, delta=1e-3)

        self.assertRaises(TypeError, xyz_to_cielab, x, y)
        self.assertRaises(TypeError, xyz_to_cielab, x)

        # Model uint8 instead of float32
        incorrect_model = np.array([0.9504, 1.0000, 1.0888], dtype=np.uint8)
        self.assertRaises(ValueError, xyz_to_cielab, l, a, b, model=incorrect_model)

        # model with incorrect shapes
        incorrect_model = np.array([0.9504, 1.0000, 1.0888, 2.2], dtype=np.float32)
        self.assertRaises(TypeError, xyz_to_cielab, x, y, z, model=incorrect_model)

        incorrect_model = np.array([0.9504], dtype=np.float32)
        self.assertRaises(TypeError, xyz_to_cielab, x, y, z, model=incorrect_model)

        self.assertRaises(TypeError, xyz_to_cielab, x, y, z, model=[])

        incorrect_model = np.array([0.9504, 1.0000, 1.0888], dtype=np.float32, copy=False)
        xyz_to_cielab(x, y, z, model = incorrect_model)

        incorrect_model = np.array([0.9504, 1.0000, 1.0888], dtype=np.float32, copy=True)
        xyz_to_cielab(x, y, z, model=incorrect_model)


class CIELAB_to_xyz_t(unittest.TestCase):
    """

    """

    # pylint: disable=too-many-statements
    def runTest(self) -> None:
        """

        :return:  void
        """
        # RGB 255.0 0 0
        # XYZ 41.246 | 21.267 | 1.933
        # lab = 53.241 | 80.092 | 67.203
        # Adobe 218.946 0 0.048
        # x, y, z = 41.246, 21.267, 1.933
        l, a, b = 53.241, 80.092, 67.203

        model_ = np.array([0.9642, 1.0000, 0.8251], dtype=np.float32)

        self.assertIsInstance(cielab_to_xyz(l, a, b, model=model_, format_8b=False), dict)

        # Test number of returned values
        self.assertEqual(len(cielab_to_xyz(l, a, b, model=model_, format_8b=False).values()), 3)

        # Check value types
        x, y, z = cielab_to_xyz(l, a, b, model=model_, format_8b=False).values()
        self.assertIsInstance(x, float)
        self.assertIsInstance(y, float)
        self.assertIsInstance(z, float)

        # todo this should raise a TypeError, but cython set True or
        #  False for incorrect type as long as the value is not null| None
        # cielab_to_xyz(x, y, z, model=model_, format_8b=12)
        # self.assertRaises(TypeError, cielab_to_xyz, x, y, z, model=model_, format_8b='True')

        # RGB 255.0 0 0
        # XYZ 41.246 | 21.267 | 1.933
        # lab = 53.241 | 80.092 | 67.203
        # Adobe 218.946 0 0.048
        # x, y, z = 41.246, 21.267, 1.933
        l, a, b = 53.241, 80.092, 67.203
        cielab_model_d65 = np.array([0.9504, 1.0000, 1.0888], dtype=np.float32)

        # using D65
        x, y, z = cielab_to_xyz(l, a, b, model=cielab_model_d65, format_8b=False).values()

        # reverse method using D65
        ll, aa, bb = xyz_to_cielab(x, y, z, model=cielab_model_d65, format_8b=False).values()
        self.assertAlmostEqual(ll, l, delta=1e-3)
        self.assertAlmostEqual(aa, a, delta=1e-3)
        self.assertAlmostEqual(bb, b, delta=1e-3)

        # Using Cielab model D50
        cielab_model_d50 = np.array([0.9642, 1.0000, 0.8251], dtype=np.float32)
        x, y, z = cielab_to_xyz(l, a, b, model=cielab_model_d50, format_8b=False).values()
        # reverse method using D65
        ll, aa, bb = xyz_to_cielab(x, y, z, model=cielab_model_d50, format_8b=False).values()
        self.assertAlmostEqual(ll, l, delta=1e-3)
        self.assertAlmostEqual(aa, a, delta=1e-3)
        self.assertAlmostEqual(bb, b, delta=1e-3)

        self.assertRaises(TypeError, cielab_to_xyz, l, a)
        self.assertRaises(TypeError, cielab_to_xyz, a)

        # Model uint8 instead of float32
        incorrect_model = np.array([0.9504, 1.0000, 1.0888], dtype=np.uint8)
        self.assertRaises(ValueError, cielab_to_xyz, l, a, b, model=incorrect_model)

        # model with incorrect shapes
        incorrect_model = np.array([0.9504, 1.0000, 1.0888, 2.2], dtype=np.float32)
        self.assertRaises(TypeError, cielab_to_xyz, l, a, b, model=incorrect_model)

        incorrect_model = np.array([0.9504], dtype=np.float32)
        self.assertRaises(TypeError, cielab_to_xyz, l, a, b, model=incorrect_model)

        self.assertRaises(TypeError, cielab_to_xyz, l, a, b, model=[])

        incorrect_model = np.array([0.9504, 1.0000, 1.0888], dtype=np.float32, copy=False)
        cielab_to_xyz(l, a, b, model=incorrect_model)

        incorrect_model = np.array([0.9504, 1.0000, 1.0888], dtype=np.float32, copy=True)
        xyz_to_cielab(l, a, b, model=incorrect_model)


class rgb_to_CIELAB_t(unittest.TestCase):
    """

    """

    # pylint: disable=too-many-statements
    def runTest(self) -> None:
        """

        :return:  void
        """
        # RGB 255.0 0 0
        # XYZ 41.246 | 21.267 | 1.933
        # lab = 53.241 | 80.092 | 67.203
        # Adobe 218.946 0 0.048
        # x, y, z = 41.246, 21.267, 1.933
        # l, a, b = 53.241, 80.092, 67.203

        r, g, b = 255.0, 0, 0

        # D50
        cielab_model_d50 = np.array([0.9642, 1.0000, 0.8251], dtype=np.float32)

        self.assertIsInstance(rgb_to_cielab(r, g, b, model=cielab_model_d50, format_8b=False), dict)

        # Test number of returned values
        self.assertEqual(len(rgb_to_cielab(r, g, b, model=cielab_model_d50, format_8b=False).values()), 3)

        # Check value types
        x, y, z = rgb_to_cielab(r, g, b, model=cielab_model_d50, format_8b=False).values()
        self.assertIsInstance(x, float)
        self.assertIsInstance(y, float)
        self.assertIsInstance(z, float)

        # using D65
        rr, gg, bb = 255.0, 0, 0
        cielab_model_d65 = np.array([0.9504, 1.0000, 1.0888], dtype=np.float32)
        l, a, b = rgb_to_cielab(rr, gg, bb, model=cielab_model_d65, format_8b=False).values()

        # reverse method using D65
        rr_, gg_, bb_ = cielab_to_rgb(l, a, b, model=cielab_model_d65, format_8b=False).values()
        self.assertAlmostEqual(rr, rr_, delta=1e-3)
        self.assertAlmostEqual(gg, gg_, delta=1e-3)
        self.assertAlmostEqual(bb, bb_, delta=1e-3)

        self.assertRaises(TypeError, rgb_to_cielab, r, g)
        self.assertRaises(TypeError, rgb_to_cielab, r)

        # Model uint8 instead of float32
        incorrect_model = np.array([0.9504, 1.0000, 1.0888], dtype=np.uint8)
        self.assertRaises(ValueError, rgb_to_cielab, r, g, b, model=incorrect_model)

        # model with incorrect shapes
        incorrect_model = np.array([0.9504, 1.0000, 1.0888, 2.2], dtype=np.float32)
        self.assertRaises(TypeError, rgb_to_cielab, r, g, b, model=incorrect_model)

        incorrect_model = np.array([0.9504], dtype=np.float32)
        self.assertRaises(TypeError, rgb_to_cielab, r, g, b, model=incorrect_model)

        self.assertRaises(TypeError, rgb_to_cielab, r, g, b, model=[])

        incorrect_model = np.array([0.9504, 1.0000, 1.0888], dtype=np.float32, copy=False)
        rgb_to_cielab(r, g, b, model=incorrect_model)

        incorrect_model = np.array([0.9504, 1.0000, 1.0888], dtype=np.float32, copy=True)
        rgb_to_cielab(r, g, b, model=incorrect_model)


class CIELAB_to_rgb_t(unittest.TestCase):
    """

    """

    # pylint: disable=too-many-statements
    def runTest(self) -> None:
        """

        :return:  void
        """
        # RGB 255.0 0 0
        # XYZ 41.246 | 21.267 | 1.933
        # lab = 53.241 | 80.092 | 67.203
        # Adobe 218.946 0 0.048
        # x, y, z = 41.246, 21.267, 1.933
        l, a, b = 53.241, 80.092, 67.203
        # r, g, b = 255.0, 0, 0

        # D50
        cielab_model_d50 = np.array([0.9642, 1.0000, 0.8251], dtype=np.float32)

        self.assertIsInstance(cielab_to_rgb(l, a, b, model=cielab_model_d50, format_8b=False), dict)

        # Test number of returned values
        self.assertEqual(len(cielab_to_rgb(l, a, b, model=cielab_model_d50, format_8b=False).values()), 3)

        # Check value types
        x, y, z = cielab_to_rgb(l, a, b, model=cielab_model_d50, format_8b=False).values()
        self.assertIsInstance(x, float)
        self.assertIsInstance(y, float)
        self.assertIsInstance(z, float)

        # using D65
        cielab_model_d65 = np.array([0.9504, 1.0000, 1.0888], dtype=np.float32)
        l, a, b = rgb_to_cielab(255.0, 0, 0, model=cielab_model_d65, format_8b=False).values()
        rr_, gg_, bb_ = cielab_to_rgb(l, a, b, model=cielab_model_d65, format_8b=False).values()
        # reverse method using D65
        ll, aa, bb = rgb_to_cielab(rr_, gg_, bb_, model=cielab_model_d65, format_8b=False).values()
        self.assertAlmostEqual(ll, l, delta=1e-3)
        self.assertAlmostEqual(aa, a, delta=1e-3)
        self.assertAlmostEqual(bb, b, delta=1e-3)

        self.assertRaises(TypeError, cielab_to_rgb, l, a)
        self.assertRaises(TypeError, cielab_to_rgb, l)

        # Model uint8 instead of float32
        incorrect_model = np.array([0.9504, 1.0000, 1.0888], dtype=np.uint8)
        self.assertRaises(ValueError, cielab_to_rgb, l, a, b, model=incorrect_model)

        # model with incorrect shapes
        incorrect_model = np.array([0.9504, 1.0000, 1.0888, 2.2], dtype=np.float32)
        self.assertRaises(TypeError, cielab_to_rgb, l, a, b, model=incorrect_model)

        incorrect_model = np.array([0.9504], dtype=np.float32)
        self.assertRaises(TypeError, cielab_to_rgb, l, a, b, model=incorrect_model)

        self.assertRaises(TypeError, cielab_to_rgb, l, a, b, model=[])

        incorrect_model = np.array([0.9504, 1.0000, 1.0888], dtype=np.float32, copy=False)
        cielab_to_rgb(l, a, b, model=incorrect_model)

        incorrect_model = np.array([0.9504, 1.0000, 1.0888], dtype=np.float32, copy=True)
        cielab_to_rgb(l, a, b, model=incorrect_model)


class rgb_2_cielab_t(unittest.TestCase):
    """

    """

    # pylint: disable=too-many-statements
    def runTest(self) -> None:
        """

        :return:  void
        """
        """
        cpdef rgb_2_cielab(
        unsigned char[:, :, :] adobe98_array,
        str illuminant ='d65',
        bint format_8b=False)
        
        """

        rgb_array = np.ndarray(shape=(640, 480, 3), dtype=np.uint8, order='C')

        # check the output type | must be ndarray
        self.assertIsInstance(rgb_2_cielab(rgb_array, illuminant='d65', format_8b=False), np.ndarray)
        # Check the output array shape | must be 640x480x3
        arr = rgb_2_cielab(rgb_array, illuminant='d65', format_8b=False)
        self.assertTupleEqual(arr.shape, (640, 480, 3))
        # Check the output array type | must be float32
        self.assertEqual(arr.dtype, numpy.float32)
        # Testing no argument | must raise a TypeError
        self.assertRaises(TypeError, rgb_2_cielab)
        # illuminant can be 'a','c','e','d50', 'd55', 'd65', 'icc'
        self.assertRaises(TypeError, rgb_2_cielab, illuminant='d64', format_8b=False)

        # This should work with D65, string made uppercase
        rgb_2_cielab(rgb_array, illuminant='D65', format_8b=False)

        # Testing all illuminant
        rgb_2_cielab(rgb_array, illuminant='a', format_8b=False)
        rgb_2_cielab(rgb_array, illuminant='c', format_8b=False)
        rgb_2_cielab(rgb_array, illuminant='e', format_8b=False)
        rgb_2_cielab(rgb_array, illuminant='d50', format_8b=False)
        rgb_2_cielab(rgb_array, illuminant='d55', format_8b=False)
        rgb_2_cielab(rgb_array, illuminant='d65', format_8b=False)
        rgb_2_cielab(rgb_array, illuminant='icc', format_8b=False)

        rgb_array = np.ndarray(shape=(640, 480, 3), dtype=np.uint8, order='C')
        cielab_array = rgb_2_cielab(rgb_array, illuminant='d65', format_8b=False)
        rgb_array_ = cielab_2_rgb(cielab_array, illuminant='d65', format_8b=False)
        self.assertTrue(numpy.array_equiv(rgb_array, rgb_array_))

        # testing array with extra dim
        rgb_array = np.ndarray(shape=(640, 480, 8), dtype=np.uint8, order='C')
        self.assertRaises(TypeError, rgb_2_cielab, rgb_array, illuminant='d65', format_8b=False)

        rgb_array = np.ndarray(shape=(640, 480, 8, 2), dtype=np.uint8, order='C')
        self.assertRaises(ValueError, rgb_2_cielab, rgb_array, illuminant='d65', format_8b=False)

        # Only passing Red should raise an alert
        rgb_array = np.ndarray(shape=(640, 480, 1), dtype=np.uint8, order='C')
        self.assertRaises(TypeError, rgb_2_cielab, rgb_array, illuminant='d65', format_8b=False)


class cielab_2_rgb_t(unittest.TestCase):
    """

    """

    # pylint: disable=too-many-statements
    def runTest(self) -> None:
        """

        :return:  void
        """

        cielab_array = np.ndarray(shape=(640, 480, 3), dtype=np.float32, order='C')

        # check the output type | must be ndarray
        self.assertIsInstance(cielab_2_rgb(cielab_array, illuminant='d65', format_8b=False), np.ndarray)
        # Check the output array shape | must be 640x480x3
        arr = cielab_2_rgb(cielab_array, illuminant='d65', format_8b=False)
        self.assertTupleEqual(arr.shape, (640, 480, 3))

        self.assertEqual(arr.dtype, numpy.uint8)
        # Testing no argument | must raise a TypeError
        self.assertRaises(TypeError, cielab_2_rgb)
        # illuminant can be 'a','c','e','d50', 'd55', 'd65', 'icc'
        self.assertRaises(TypeError, cielab_2_rgb, illuminant='d64', format_8b=False)
        # Testing all illuminant
        cielab_2_rgb(cielab_array, illuminant='a', format_8b=False)
        cielab_2_rgb(cielab_array, illuminant='c', format_8b=False)
        cielab_2_rgb(cielab_array, illuminant='e', format_8b=False)
        cielab_2_rgb(cielab_array, illuminant='d50', format_8b=False)
        cielab_2_rgb(cielab_array, illuminant='d55', format_8b=False)
        cielab_2_rgb(cielab_array, illuminant='d65', format_8b=False)
        cielab_2_rgb(cielab_array, illuminant='icc', format_8b=False)

        rgb_array = np.ndarray(shape=(640, 480, 3), dtype=np.uint8, order='C')
        cielab_array = rgb_2_cielab(rgb_array, illuminant='d65', format_8b=False)
        rgb_array_ = cielab_2_rgb(cielab_array, illuminant='d65', format_8b=False)
        self.assertTrue(numpy.array_equiv(rgb_array, rgb_array_))

        # testing array with extra dim
        rgb_array = np.ndarray(shape=(640, 480, 8), dtype=np.float32, order='C')
        self.assertRaises(TypeError, cielab_2_rgb, rgb_array, illuminant='d65', format_8b=False)

        rgb_array = np.ndarray(shape=(640, 480, 8, 2), dtype=np.float32, order='C')
        self.assertRaises(ValueError, cielab_2_rgb, rgb_array, illuminant='d65', format_8b=False)

        # Only passing Red should raise an alert
        rgb_array = np.ndarray(shape=(640, 480, 1), dtype=np.float32, order='C')
        self.assertRaises(TypeError, cielab_2_rgb, rgb_array, illuminant='d65', format_8b=False)


class WhiteBalance_t(unittest.TestCase):
    """
    cpdef WhiteBalance(
        unsigned char[:, :, :] adobe98_array,
        float c1=1.0,
        str illuminant='d50',
        bint format_8b = False
    ):
    """

    # pylint: disable=too-many-statements
    def runTest(self) -> None:
        """

        :return:  void
        """
        texture = pygame.image.load("..//Assets//background2.png").convert()
        texture = pygame.transform.smoothscale(texture, (640, 512))
        arr = pygame.surfarray.array3d(texture)
        white_b = WhiteBalance(arr, illuminant='D65', format_8b=False)
        # Testing instance
        self.assertIsInstance(white_b, numpy.ndarray)
        # Testing arrays shapes
        self.assertTupleEqual(white_b.shape, arr.shape)

        gimp_white_balance = pygame.image.load("..//Assets//white_surf.png")
        gimp_white_balance = pygame.transform.smoothscale(texture, (640, 512))
        arr1 = pygame.surfarray.pixels3d(gimp_white_balance)
        # compare gimp white balance with Cielab lib white balance
        # Gimp white balance seems to use the illuminant d65
        # self.assertTrue(numpy.allclose(white_b, arr1, rtol=2, atol=2))


class WhiteBalanceInplace_t(unittest.TestCase):
    """

    """

    # pylint: disable=too-many-statements
    def runTest(self) -> None:
        """

        :return:  void
        """
        texture = pygame.image.load("..//Assets//background2.png").convert()
        texture = pygame.transform.smoothscale(texture, (640, 512))
        arr = pygame.surfarray.array3d(texture)
        WhiteBalanceInplace(arr, illuminant='D65')

        # Testing instance
        self.assertIsInstance(arr, numpy.ndarray)

        # Testing arrays shapes
        self.assertTupleEqual(arr.shape, arr.shape)

        gimp_white_balance = pygame.image.load("..//Assets//white_surf.png")
        arr1 = pygame.surfarray.pixels3d(gimp_white_balance)

        # compare gimp white balance with Cielab lib white balance
        # Gimp white balance seems to use the illuminant d65
        # self.assertTrue(numpy.allclose(arr, arr1, rtol=1, atol=1))



class rgb2adobe_t(unittest.TestCase):
    """
    cpdef rgb2adobe(unsigned char[:, :, :] adobe98_array, str ref='D65'):
    """

    # pylint: disable=too-many-statements
    def runTest(self) -> None:
        """

        :return:  void
        """
        # testing 24 bit image
        texture = pygame.image.load("..//Assets//background2.png").convert()
        texture = pygame.transform.smoothscale(texture, (640, 512))
        arr = pygame.surfarray.array3d(texture)
        # Check the output type
        self.assertIsInstance(rgb2adobe(arr), np.ndarray)
        # Check the array shape
        self.assertTupleEqual(arr.shape, rgb2adobe(arr).shape)

        self.assertEqual(arr.dtype, np.uint8)
        # Check array data type
        self.assertEqual(rgb2adobe(arr).dtype, np.float32)

        # Testing 32 bit image
        texture = pygame.image.load("..//Assets//background2.png").convert_alpha()
        texture = pygame.transform.smoothscale(texture, (640, 512))
        arr32 = pygame.surfarray.array3d(texture)
        self.assertIsInstance(rgb2adobe(arr32), np.ndarray)
        self.assertTupleEqual(arr.shape, rgb2adobe(arr32).shape)

        # testing RGBA array
        rgba_array = np.ndarray(shape=(640, 480, 4), dtype=np.uint8, order='C')
        self.assertIsInstance(rgb2adobe(rgba_array), np.ndarray)

        rgba_array = np.ndarray(shape=(640, 480, 4), dtype=np.uint8, order='F')
        self.assertIsInstance(rgb2adobe(rgba_array), np.ndarray)

        # Test float32 instead of uint8
        rgba_array = np.ndarray(shape=(640, 480, 3), dtype=np.float32, order='C')
        self.assertRaises(ValueError, rgb2adobe, rgba_array)

        # Test no argument
        self.assertRaises(TypeError, rgb2adobe)

        # Test incorrect argument for ref
        self.assertRaises(TypeError, rgb2adobe, ref = 'D64')

        rgb2adobe(arr, ref='a')
        rgb2adobe(arr, ref='c')
        rgb2adobe(arr, ref='e')
        rgb2adobe(arr, ref='d50')
        rgb2adobe(arr, ref='d55')
        rgb2adobe(arr, ref='d65')
        rgb2adobe(arr, ref='icc')

        arr1 = rgb2adobe(arr, ref='d65')
        arr2 = adobe2rgb(arr1, ref='d65')
        self.assertTrue(numpy.allclose(arr, arr2, rtol=1e-02, atol=1e-02, equal_nan=False))


class rgb2adobe_inplace_t(unittest.TestCase):
    """
    cpdef void rgb2adobe_inplace(unsigned char[:, :, :] adobe98_array, str ref='D65'):
    """

    # pylint: disable=too-many-statements
    def runTest(self) -> None:
        """

        :return:  void
        """
        # testing 24 bit image
        texture = pygame.image.load("..//Assets//background2.png").convert()
        texture = pygame.transform.smoothscale(texture, (640, 512))
        arr = pygame.surfarray.array3d(texture)
        # Check the output type

        self.assertEqual(rgb2adobe_inplace(arr), None)

        self.assertEqual(rgb2adobe_inplace(arr), None)

        # testing RGBA array
        rgba_array = np.ndarray(shape=(640, 480, 4), dtype=np.uint8, order='C')
        self.assertEqual(rgb2adobe_inplace(rgba_array), None)

        rgba_array = np.ndarray(shape=(640, 480, 4), dtype=np.uint8, order='F')
        rgb2adobe_inplace(rgba_array)

        # Test float32 instead of uint8
        rgba_array = np.ndarray(shape=(640, 480, 3), dtype=np.float32, order='C')
        self.assertRaises(ValueError, rgb2adobe_inplace, rgba_array)

        # Test no argument
        self.assertRaises(TypeError, rgb2adobe_inplace)

        # Test incorrect argument for ref
        self.assertRaises(TypeError, rgb2adobe_inplace, ref='D64')

        rgb2adobe_inplace(arr, ref='a')
        rgb2adobe_inplace(arr, ref='c')
        rgb2adobe_inplace(arr, ref='e')
        rgb2adobe_inplace(arr, ref='d50')
        rgb2adobe_inplace(arr, ref='d55')
        rgb2adobe_inplace(arr, ref='d65')
        rgb2adobe_inplace(arr, ref='icc')

        # Todo do the reverse
        # self.assertTrue(arr.all()==)


class array3d_stats_t(unittest.TestCase):
    """
    cpdef inline im_stats array3d_stats(object array)
    """

    # pylint: disable=too-many-statements
    def runTest(self) -> None:
        """

        :return:  void
        """
        # todo with float array

        arr = numpy.empty((100, 100, 3), dtype=np.uint8)
        arr[:, :, 0] = 84
        arr[:, :, 1] = 67
        arr[:, :, 2] = 194
        res = array3d_stats(arr)
        self.assertIsInstance(res, dict)

        self.assertTrue(len(res) == 6)
        r_mean, r_std, g_mean, g_std, b_mean, b_std = array3d_stats(arr).values()

        # Checking the mean values
        self.assertAlmostEqual(numpy.mean(arr[:, :, 0]) / 255.0, r_mean, delta=1e-4)
        self.assertAlmostEqual(numpy.mean(arr[:, :, 1]) / 255.0, g_mean, delta=1e-4)
        self.assertAlmostEqual(numpy.mean(arr[:, :, 2]) / 255.0, b_mean, delta=1e-4)

        # Checking the standard deviation
        self.assertAlmostEqual(numpy.std(arr[:, :, 0]) / 255.0, r_std, delta=1e-4)
        self.assertAlmostEqual(numpy.std(arr[:, :, 1]) / 255.0, g_std, delta=1e-4)
        self.assertAlmostEqual(numpy.std(arr[:, :, 2]) / 255.0, b_std, delta=1e-4)

        # Testing with a full array
        texture = pygame.image.load("..//Assets//background2.png").convert()
        texture = pygame.transform.smoothscale(texture, (640, 512))
        arr = pygame.surfarray.array3d(texture)

        r_mean, r_std, g_mean, g_std, b_mean, b_std = array3d_stats(arr).values()

        # Checking the mean values
        self.assertAlmostEqual(numpy.mean(arr[:, :, 0]) / 255.0, r_mean, delta=1e-4)
        self.assertAlmostEqual(numpy.mean(arr[:, :, 1]) / 255.0, g_mean, delta=1e-4)
        self.assertAlmostEqual(numpy.mean(arr[:, :, 2]) / 255.0, b_mean, delta=1e-4)
        # Checking the standard deviation
        self.assertAlmostEqual(numpy.std(arr[:, :, 0]) / 255.0, r_std, delta=1e-4)
        self.assertAlmostEqual(numpy.std(arr[:, :, 1]) / 255.0, g_std, delta=1e-4)
        self.assertAlmostEqual(numpy.std(arr[:, :, 2]) / 255.0, b_std, delta=1e-4)

        arr = numpy.empty((100, 100, 3), dtype=np.float32)
        arr[:, :, 0] = 84.0
        arr[:, :, 1] = 67.0
        arr[:, :, 2] = 194.0
        res = array3d_stats(arr)
        self.assertIsInstance(res, dict)

        self.assertTrue(len(res) == 6)
        r_mean, r_std, g_mean, g_std, b_mean, b_std = array3d_stats(arr).values()

        # Checking the mean values
        self.assertAlmostEqual(numpy.mean(arr[:, :, 0]) / 255.0, r_mean, delta=1e-4)
        self.assertAlmostEqual(numpy.mean(arr[:, :, 1]) / 255.0, g_mean, delta=1e-4)
        self.assertAlmostEqual(numpy.mean(arr[:, :, 2]) / 255.0, b_mean, delta=1e-4)

        # Checking the standard deviation
        self.assertAlmostEqual(numpy.std(arr[:, :, 0]) / 255.0, r_std, delta=1e-4)
        self.assertAlmostEqual(numpy.std(arr[:, :, 1]) / 255.0, g_std, delta=1e-4)
        self.assertAlmostEqual(numpy.std(arr[:, :, 2]) / 255.0, b_std, delta=1e-4)




def run_testsuite():
    """
    test suite

    :return: void
    """

    suite = unittest.TestSuite()

    suite.addTests([
        array3d_mean_t(),
        rgb_to_xyz_t(),
        xyz_to_rgb_t(),

        xyz_adobe98_t(),
        adobe98_xyz_t(),

        xyz_CIELAB_t(),
        CIELAB_to_xyz_t(),

        rgb_to_CIELAB_t(),
        CIELAB_to_rgb_t(),

        rgb_2_cielab_t(),
        cielab_2_rgb_t(),

        WhiteBalance_t(),
        WhiteBalanceInplace_t(),

        rgb2adobe_t(),
        rgb2adobe_inplace_t(),

        array3d_stats_t()

    ])

    unittest.TextTestRunner().run(suite)
    sys.exit(0)


if __name__ == '__main__':

    run_testsuite()
    sys.exit(0)
