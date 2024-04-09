
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
        adobe98_xyz, array3d_mean, xyz_to_cielab, cielab_to_xyz, rgb_to_cielab, \
        cielab_to_rgb, rgb2adobe, rgb_2_cielab, cielab_2_rgb, rgb2adobe_inplace, \
        WhiteBalance, adobe2rgb, array3d_stats, WhiteBalanceInplace, rgb2xyz

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

PROJECT_PATH = list(Cielab.__path__)
os.chdir(PROJECT_PATH[0] + "/demo")

WIDTH = 1280
HEIGHT = 1024
SCREEN = pygame.display.set_mode((WIDTH, HEIGHT))

texture = pygame.image.load("..//Assets//background2.png").convert()
texture = pygame.transform.smoothscale(texture, (640, 512))
arr = pygame.surfarray.array3d(texture)

cielab_array = rgb_2_cielab(arr)
cielab_surface = pygame.surfarray.make_surface(cielab_array)

adobe_array = rgb2adobe(arr)
adobe_surface = pygame.surfarray.make_surface(adobe_array)

xyz_array = rgb2xyz(arr)
xyz_surface = pygame.surfarray.make_surface(xyz_array)


a = 0
while 1:
    if a > 2000:
        break
    SCREEN.fill((0, 0, 0, 0))
    SCREEN.blit(texture, (0, 0))
    SCREEN.blit(xyz_surface, (640, 0))

    SCREEN.blit(cielab_surface, (0, 512))
    SCREEN.blit(adobe_surface, (640, 512))

    pygame.event.pump()
    pygame.display.flip()
    a += 1

