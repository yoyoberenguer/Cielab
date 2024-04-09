
"""
                 GNU GENERAL PUBLIC LICENSE
                       Version 3, 29 June 2007

 Copyright (C) 2007 Free Software Foundation, Inc. <https://fsf.org/>
 Everyone is permitted to copy and distribute verbatim copies
 of this license document, but changing it is not allowed.

Copyright Yoann Berenguer
"""

import timeit
import os

try:
    import pygame
except ImportError:
    raise ImportError(
        "\n<pygame> library is missing on your system."
        "\nTry: \n   C:\\pip install pygame on a window command prompt.")

try:
    import numpy
except ImportError:
    raise ImportError(
        "\n<numpy> library is missing on your system."
        "\nTry: \n   C:\\pip install numpy on a window command prompt.")

try:
    import Cielab
except ImportError:
    raise ImportError(
        "\n<Cielab> library is missing on your system."
        "\nTry: \n   C:\\pip install Cielab on a window command prompt.")


try:
    from Cielab import rgb_to_xyz, xyz_to_rgb, xyz_adobe98, \
        adobe98_xyz, xyz_to_cielab, cielab_to_xyz, rgb_to_cielab, \
        cielab_to_rgb, rgb2adobe, adobe2rgb, rgb_2_cielab, cielab_2_rgb, rgb2adobe_inplace, \
        adobe2rgb_inplace, WhiteBalance, WhiteBalanceInplace, array3d_mean, array3d_stats, \
        rgb2xyz

except ImportError:
    raise ImportError(
        "\n<Cielab> library is missing on your system."
        "\nTry: \n   C:\\pip install Cielab on a window command prompt.")

PROJECT_PATH = list(Cielab.__path__)
os.chdir(PROJECT_PATH[0] + "/demo")


surf = pygame.image.load("..\\Assets\\background2.png")
arr = pygame.surfarray.pixels3d(surf)

if __name__ == '__main__':
    n = 100000
    # cpdef inline xyz rgb_to_xyz(float r, float g, float b, str ref='D65')nogil:
    t = timeit.timeit("rgb_to_xyz(10, 10, 10)", "from __main__ import rgb_to_xyz", number=n)/n
    print("rgb_to_xyz %s executions timing %s" % (n, t))

    t = timeit.timeit("xyz_to_rgb(10, 10, 10)", "from __main__ import xyz_to_rgb", number=n) / n
    print("xyz_to_rgb %s executions timing %s" % (n, t))

    # cpdef inline rgb xyz_adobe98(float x, float y, float z, str ref='D65')nogil:
    t = timeit.timeit("xyz_adobe98(10, 10, 10)", "from __main__ import xyz_adobe98", number=n) / n
    print("xyz_adobe98 %s executions timing %s" % (n, t))

    t = timeit.timeit("adobe98_xyz(10, 10, 10)", "from __main__ import adobe98_xyz", number=n) / n
    print("adobe98_xyz %s executions timing %s" % (n, t))

    """
    cpdef lab xyz_to_cielab(
        float x,
        float y,
        float z,
        const float [:] model=cielab_model_d50,
        bint format_8b = True
    )nogil:
    """
    t = timeit.timeit("xyz_to_cielab(10, 10, 10)", "from __main__ import xyz_to_cielab", number=n) / n
    print("xyz_to_cielab %s executions timing %s" % (n, t))

    t = timeit.timeit("cielab_to_xyz(10, 10, 10)", "from __main__ import cielab_to_xyz", number=n) / n
    print("cielab_to_xyz %s executions timing %s" % (n, t))

    """
    cpdef inline lab rgb_to_cielab(
        float r,
        float g,
        float b,
        const float [:] model=cielab_model_d50,
        bint format_8b = True
    )nogil:
    """

    t = timeit.timeit("rgb_to_cielab(10, 10, 10)", "from __main__ import rgb_to_cielab", number=n) / n
    print("rgb_to_cielab %s executions timing %s" % (n, t))

    t = timeit.timeit("cielab_to_rgb(10, 10, 10)", "from __main__ import cielab_to_rgb", number=n) / n
    print("cielab_to_rgb %s executions timing %s" % (n, t))

    n = 100
    t = timeit.timeit("rgb2adobe(arr)", "from __main__ import rgb2adobe, arr", number=n) / n
    print("rgb2adobe %s executions timing %s" % (n, t))

    # cpdef adobe2rgb(float [:, :, :] adobe98_array, str ref='D65'):
    arr1 = arr.astype(numpy.float32)
    t = timeit.timeit("adobe2rgb(arr1)", "from __main__ import adobe2rgb, arr1", number=n) / n
    print("adobe2rgb %s executions timing %s" % (n, t))

    t = timeit.timeit("rgb_2_cielab(arr)", "from __main__ import rgb_2_cielab, arr", number=n) / n
    print("rgb_2_cielab %s executions timing %s" % (n, t))

    t = timeit.timeit("cielab_2_rgb(arr1)", "from __main__ import cielab_2_rgb, arr1", number=n) / n
    print("cielab_2_rgb %s executions timing %s" % (n, t))

    t = timeit.timeit("rgb2adobe_inplace(arr)", "from __main__ import rgb2adobe_inplace, arr", number=n) / n
    print("rgb2adobe_inplace %s executions timing %s" % (n, t))

    t = timeit.timeit("adobe2rgb_inplace(arr)", "from __main__ import adobe2rgb_inplace, arr", number=n) / n
    print("adobe2rgb_inplace %s executions timing %s" % (n, t))

    t = timeit.timeit("WhiteBalance(arr)", "from __main__ import WhiteBalance, arr", number=n) / n
    print("WhiteBalance %s executions timing %s" % (n, t))

    t = timeit.timeit("array3d_mean(arr)", "from __main__ import array3d_mean, arr", number=n) / n
    print("array3d_mean %s executions timing %s" % (n, t))

    t = timeit.timeit("array3d_stats(arr)", "from __main__ import array3d_stats, arr", number=n) / n
    print("array3d_stats %s executions timing %s" % (n, t))

    t = timeit.timeit("WhiteBalance(arr)", "from __main__ import WhiteBalance, arr", number=n) / n
    print("WhiteBalance %s executions timing %s" % (n, t))

    t = timeit.timeit("WhiteBalanceInplace(arr)", "from __main__ import WhiteBalanceInplace, arr", number=n) / n
    print("WhiteBalanceInplace %s executions timing %s" % (n, t))

    t = timeit.timeit("rgb2xyz(arr)", "from __main__ import rgb2xyz, arr", number=n) / n
    print("rgb2xyz %s executions timing %s" % (n, t))
