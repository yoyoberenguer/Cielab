# cython: binding=False, boundscheck=False, wraparound=False, nonecheck=False, cdivision=True,
# profile=False, initializedcheck=False, exceptval=False
# cython: optimize.use_switch=True
# cython: warn.maybe_uninitialized=False
# cython: warn.unused=False
# cython: warn.unused_result=False
# cython: warn.unused_arg=False
# cython: language_level=3
# encoding: utf-8

"""
                 GNU GENERAL PUBLIC LICENSE
                       Version 3, 29 June 2007

 Copyright (C) 2007 Free Software Foundation, Inc. <https://fsf.org/>
 Everyone is permitted to copy and distribute verbatim copies
 of this license document, but changing it is not allowed.

Copyright Yoann Berenguer
"""
import ctypes

try:
    cimport cython
    from cython.parallel cimport prange

except ImportError:
    raise ImportError("\n<cython> library is missing on your system."
          "\nTry: \n   C:\\pip install cython on a window command prompt.")


try:
    cimport cython
    from cython.parallel cimport prange
    from cpython cimport PyObject_CallFunctionObjArgs, PyObject, \
        PyList_SetSlice, PyObject_HasAttr, PyObject_IsInstance, \
        PyObject_CallMethod, PyObject_CallObject


except ImportError:
    raise ImportError("\n<cython> library is missing on your system."
          "\nTry: \n   C:\\pip install cython on a window command prompt.")


import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"

try:
    import pygame
    from pygame import BLEND_RGB_ADD
    from pygame import Surface, SRCALPHA, RLEACCEL
    from pygame.transform import scale, smoothscale
    from pygame.surfarray import array3d, pixels3d, array_alpha, pixels_alpha, \
        make_surface
    from pygame.image import frombuffer

except ImportError:
    raise ImportError("\n<Pygame> library is missing on your system."
          "\nTry: \n   C:\\pip install pygame on a window command prompt.")


# NUMPY IS REQUIRED
try:
    import numpy
    from numpy import ndarray, zeros, empty, uint8, int32, float64, \
        float32, dstack, full, ones, asarray, ascontiguousarray, full_like,\
        int16, arange
except ImportError:
    raise ImportError("\n<numpy> library is missing on your system."
          "\nTry: \n   C:\\pip install numpy on a window command prompt.")

cimport numpy as np

from libc.math cimport sqrtf as sqrt, powf as pow, roundf as round_f


DEF SCHEDULE = 'static'

from Cielab.config import OPENMP, THREAD_NUMBER

cdef int THREADS = 1

if OPENMP:
     THREADS = THREAD_NUMBER

cdef float c_1_255 = <float>1.0/<float>255.0


# https://en.wikipedia.org/wiki/Adobe_RGB_color_space#cite_note-AdobeRGBColorImagingEncoding-4
# https://nanotools.app/rgb-to-xyz?rgb=%28255%2C52%2C1%29
# https://www.easyrgb.com/en/convert.php#inputFORM
# https://www.adobe.com/digitalimag/pdfs/AdobeRGB1998.pdf
# https://ajalt.github.io/colormath/converter/
# http://www.brucelindbloom.com/index.html?Eqn_RGB_to_XYZ.html

# todo test 8 bits texture



@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cpdef tuple array3d_mean(object array):
    """
    RETURN MEAN VALUES FOR EACH CHANNELS OF AN RGB ARRAY 
    
    Python hook method
    
    The input array parameter is a classic python object as the data type will 
    be determine below (uint8 or float32). This allow to process different array's 
    data types 
    
    :param array: numpy.ndarray type (w, h, 3) of type uint8 or float32 
    :return : mean values for all channels and pixel count (c)
    """
    return array3d_mean_c(array)


@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cdef tuple array3d_mean_c(object array):
    """
    RETURN MEAN VALUES FOR EACH CHANNELS OF AN ARRAY 
    
    Call this function from Cython otherwise use array3d_mean from python
    
    The input array parameter is a classic python object as the data type will 
    be determine below (uint8 or float32). This allow to process different array's 
    data types 
    
    :param array: numpy.ndarray type (w, h, 3) of type uint8 or float32 
    :return : mean values for all channels and pixel count (c), (r, g, b, c) with Red, Green, Blue as float and 
        c as unsigned int
    
    """

    cdef:
        Py_ssize_t w, h, n

    try:
        w, h, n = array.shape[:3]
    except Exception as e:
        raise ValueError("adobe98_array argument must be shape (h, w, 3) type uint8.\n %s " % e)

    if array.dtype != 'uint8' and array.dtype != 'float32':
        raise TypeError('Array must be type uint8 or float32; %s is not supported.' % array.dtype)

    if n!= 3:
        raise TypeError('Array must be shape (w, h, 3); (%s, %s, %s) is not supported.' % (w, h, n))

    cdef:
        int i, j
        unsigned int c = 0

        float * fr
        float * fg
        float * fb

        unsigned char * ur
        unsigned char * ug
        unsigned char * ub

        float r = 0
        float g = 0
        float b = 0

        unsigned char [:, :, :] u_array = array if array.dtype == 'uint8' else None
        float [:, :, :] f_array = array if array.dtype == 'float32' else None

    c = w * h

    if array.dtype == 'float32':
        with nogil:
            for j in prange(h, schedule='dynamic', num_threads=THREADS):
                for i in range(w):
                    fr = &f_array[i, j, 0]
                    fg = &f_array[i, j, 1]
                    fb = &f_array[i, j, 2]
                    r += fr[0]
                    g += fg[0]
                    b += fb[0]

    else:
        with nogil:
            for j in prange(h, schedule='dynamic', num_threads=THREADS):
                for i in range(w):
                    ur = &u_array[i, j, 0]
                    ug = &u_array[i, j, 1]
                    ub = &u_array[i, j, 2]
                    r += ur[0]
                    g += ug[0]
                    b += ub[0]

    return <float> r / <float> c, <float> g / <float> c, <float> b / <float> c, c


@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cpdef inline im_stats array3d_stats(object array):

    """
    RETURN MEAN RGB VALUES AND STANDARD DEVIATION FOR EACH CHANNELS
    
    Python hook method 
    
    Array_ parameter is a python object as the data type will be determine below 
    
    :param array: numpy.ndarray; array shape (w, h, 3) type uint8 containing RGB pixels
    :return: im_stats; returns a structure im_stats containing R,G,B mean values and R,G,B standard deviation.
        In python this is similar to a dictionary. 
    
    """

    cdef:
        int w, h, i, j
        unsigned int wh
        im_stats image_stats

    array_type = array.dtype

    w, h = array.shape[:2]

    wh = w * h * 255

    if wh == 0:
        raise ValueError('\nArray sizes cannot be null.')

    cdef:
        float mean_r, mean_g, mean_b, c
        float dev_r=0.0, dev_g=0.0, dev_b=0.0
        # array is most likely to be uint8)
        unsigned char [:, :, :] u_array = array if array_type == 'uint8' else None
        # array is most likely to be float32
        float [:, :, :] f_array = array if array_type == 'float32' else None

    # Find the mean values for each channels
    mean_r, mean_g, mean_b, c = array3d_mean_c(array)

    if array_type == 'uint8':
        with nogil:
            for j in range(h):
                for i in range(w):
                    dev_r = dev_r + <float>pow(<float>u_array[i, j, 0] - mean_r, <float>2.0)
                    dev_g = dev_g + <float>pow(<float>u_array[i, j, 1] - mean_g, <float>2.0)
                    dev_b = dev_b + <float>pow(<float>u_array[i, j, 2] - mean_b, <float>2.0)

    else:
        with nogil:
            for j in range(h):
                for i in range(w):
                    dev_r = dev_r + <float>pow(f_array[i, j, 0] - mean_r, <float>2.0)
                    dev_g = dev_g + <float>pow(f_array[i, j, 1] - mean_g, <float>2.0)
                    dev_b = dev_b + <float>pow(f_array[i, j, 2] - mean_b, <float>2.0)

    cdef float std_dev_r, std_dev_g, std_dev_b

    std_dev_r = <float>sqrt(dev_r/c)
    std_dev_g = <float>sqrt(dev_g/c)
    std_dev_b = <float>sqrt(dev_b/c)

    image_stats.red_mean = mean_r * c_1_255
    image_stats.red_std_dev = std_dev_r * c_1_255
    image_stats.green_mean = mean_g * c_1_255
    image_stats.green_std_dev = std_dev_g * c_1_255
    image_stats.blue_mean = mean_b * c_1_255
    image_stats.blue_std_dev = std_dev_b * c_1_255

    return image_stats


# -----------------------------------------------------------------
# Illuminant models
cdef list illuminant_list = ['a','c','e','d50', 'd55', 'd65', 'icc']

# Hooks for python
model_a = numpy.array([1.0985, 1.0000, 0.3558], dtype=float32)
model_c = numpy.array([0.9807, 1.0000, 1.1822], dtype=float32)
model_e = numpy.array([1.0000, 1.0000, 1.0000], dtype=float32)
model_d50 = numpy.array([0.9642, 1.0000, 0.8251], dtype=float32)
model_d55 = numpy.array([0.9568, 1.0000, 0.9214], dtype=float32)
model_d65 = numpy.array([0.9504, 1.0000, 1.0888], dtype=float32)
model_icc = numpy.array([0.9642, 1.0000, 0.8249], dtype=float32)

# Hooks for cython
cdef float [:] cielab_model_a = numpy.array([1.0985, 1.0000, 0.3558], dtype=float32)
cdef float [:] cielab_model_c = numpy.array([0.9807, 1.0000, 1.1822], dtype=float32)
cdef float [:] cielab_model_e = numpy.array([1.0000, 1.0000, 1.0000], dtype=float32)
cdef float [:] cielab_model_d50 = numpy.array([0.9642, 1.0000, 0.8251], dtype=float32)
cdef float [:] cielab_model_d55 = numpy.array([0.9568, 1.0000, 0.9214], dtype=float32)
cdef float [:] cielab_model_d65 = numpy.array([0.9504, 1.0000, 1.0888], dtype=float32)
cdef float [:] cielab_model_icc = numpy.array([0.9642, 1.0000, 0.8249], dtype=float32)

# Constants for CIELAB conversions
cdef float SIGMA = <float>6.0 / <float>29.0
cdef float SIGMA_SQR = SIGMA ** <float>2.0
cdef float LAMBDA = <float>16.0 / <float>116.0  # 4/29
cdef float _1_24 = <float>1.0/<float>2.4
cdef float _1_3 = <float>1.0/<float>3.0
cdef float _1_100 = <float>1.0 / <float>100.0
cdef float _1_116 = <float>1.0 / <float>116.0
cdef float _1_255 = <float>1.0 / <float>255.0
cdef float _1_200 = <float>1.0 / <float>200.0
cdef float _1_500 = <float>1.0 / <float>500.0
cdef float _255_100 = <float>255.0/<float>100.0
cdef float _100_255 = <float>100.0/<float>255.0

# Adobe 1998 D65
cdef float [:, :] Adobe1998_d65=numpy.array([
[0.5767309,  0.1855540,  0.1881852],
[0.2973769,  0.6273491,  0.0752741],
[0.0270343,  0.0706872,  0.9911085]], dtype=numpy.float32)

cdef float [:, :] Adobe1998_d65_inv=numpy.array([
[2.0413690,  -0.5649464,  -0.3446944],
[-0.9692660,  1.8760108,  0.0415560],
[0.0134474,  -0.1183897,  1.0154096]], dtype=numpy.float32)

# sRGB D65
cdef float [:, :] srgb_d65=numpy.array([
[0.4124564,  0.3575761,  0.1804375],
[0.2126729,  0.7151522,  0.0721750],
[0.0193339,  0.1191920,  0.9503041]], dtype=numpy.float32)

cdef float [:, :] srgb_d65_inv=numpy.array([
[3.2404542,  -1.5371385,  -0.4985314],
[-0.9692660,  1.8760108,  0.0415560],
[0.0556434, -0.2040259,  1.0572252]], dtype=numpy.float32)


@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cdef float [:] cielab_illuminant_c(str illuminant) nogil:
    """
    Cython internal function only  
    
    :param illuminant: string representing the illuminant model 
    :return : cython memoryview (selected illuminant model). 
        1D buffer containing the CIELAB illuminant model 
    """
    cdef float [:] illuminant_model

    with gil:
        illuminant = illuminant.upper()

    if illuminant == 'A':
        illuminant_model = cielab_model_a
    elif illuminant == 'C':
        illuminant_model = cielab_model_c
    elif illuminant == 'E':
        illuminant_model = cielab_model_e
    elif illuminant == 'D50':
        illuminant_model = cielab_model_d50
    elif illuminant == 'D55':
        illuminant_model = cielab_model_d55
    elif illuminant == 'D65':
        illuminant_model = cielab_model_d65
    elif illuminant == 'ICC':
        illuminant_model = cielab_model_icc
    else:
        with gil:
            raise ValueError(
                "\nArgument Illuminant expect ('a','c','e',"
                "'d50', 'd55', 'd65', 'icc'; got %s)" % illuminant)

    return illuminant_model




# ADOBE 1998 RGB COLOR SPACE
# Corresponding absolute XYZ tristimulus values for the reference
# display white and black points
# WHITE point
cdef float xw, yw, zw
xw = <float>152.07
yw = <float>160.00
zw = <float>174.25

cdef float xk, yk, zk
# BLACK point
xk = <float>0.5282
yk = <float>0.5557
zk = <float>0.6052
# xk = <float>0.0
# yk = <float>0.0
# zk = <float>0.0
cdef float ADOBE_GAMMA = <float>1.0 / <float>(<float>563.0/<float>256.0)


@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cpdef inline rgb xyz_adobe98(float x, float y, float z, str ref='D65')nogil:
    """
    CONVERSION FROM XYZ (D65 & D50) TO 24-BIT ADOBE RGB
    
    e.g:
    >>>rgb = xyz_adobe98(41.246, 21.267, 1.933)
    {'r': 218.9474334716797, 'g': 0.0, 'b': 0.0}
    
    >>>r, g, b = xyz_adobe98(41.246, 21.267, 1.933).values()
    
    RGB values are capped in range 0..255

    :param x: X color components 
    :param y: Y color components 
    :param z: Z color components 
    :param ref: reference 'D50' or 'D65' default D65
    :return : return RGB structure containing RGB values in range [0..255], this will be 
        identical to a dictionary in python e.g {'r': 218.9474334716797, 'g': 0.0, 'b': 0.0}

    """
    return xyz_adobe98_c(x, y, z, ref)

@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)

cdef inline rgb xyz_adobe98_c(float x, float y, float z, str ref='D65')nogil:
    """
    CONVERSION FROM XYZ (D65) TO 24-BIT ADOBE RGB    
    
     e.g:
    >>>rgb = xyz_adobe98_c(41.246, 21.267, 1.933)
    {'r': 218.9474334716797, 'g': 0.0, 'b': 0.0}
    
    >>>r, g, b = xyz_adobe98_c(41.246, 21.267, 1.933).values()
    
    RGB values are capped in range 0..255

    :param x: X color component
    :param y: Y color component
    :param z: Z color component
    :param ref: reference 'D50' or 'D65' default D65
    :return : return RGB structure containing RGB values in range [0..255], this will be 
        identical to a dictionary in python e.g {'r': 218.9474334716797, 'g': 0.0, 'b': 0.0}

    """
    cdef:
        rgb rgb_
        float k0
        float k1
        float k2

    with gil:
        ref = ref.upper()

    if ref != 'D50' and ref !='D65':
        with gil:
            raise ValueError("\nAttribute ref must be D50 or D65")

    x *= _1_100
    y *= _1_100
    z *= _1_100

    k0 = yw - yk
    k1 = xw - xk
    k2 = zw - zk

    xa = x * k1 * (yw / xw) + xk
    ya = y * k0 + yk
    za = z * k2 * (yw / zw) + zk

    x = (xa - xk) / k1 * (xw / yw)
    y = (ya - yk) / k0
    z = (za - zk) / k2 * (zw / yw)

    if ref == 'D65':
        # Adobe 1998 Calibration D65
        rgb_.r = x * +<float> 2.0413690 + y * -<float> 0.5649464 + z * -<float> 0.3446944
        rgb_.g = x * -<float> 0.9692660 + y * +<float> 1.8760108 + z * +<float> 0.0415560
        rgb_.b = x * +<float> 0.0134474 + y * -<float> 0.1183897 + z * +<float> 1.0154096

    if ref == 'D50':
        # D50
        rgb_.r = x * +<float> 1.9624274 + y * -<float> 0.6105343 + z * -<float> 0.3413404
        rgb_.g = x * -<float> 0.9787684 + y * +<float> 1.9161415 + z * +<float> 0.0334540
        rgb_.b = x * +<float> 0.0286869 + y * -<float> 0.1406752 + z * +<float> 1.3487655

    if rgb_.r < 0.0:
        rgb_.r = <float>0.0
    else:
        rgb_.r = <float>255.0 * pow(rgb_.r, ADOBE_GAMMA)

    if rgb_.g < 0.0:
        rgb_.g = <float> 0.0
    else:
        rgb_.g = <float>255.0 * pow(rgb_.g, ADOBE_GAMMA)

    if rgb_.b < 0.0:
        rgb_.b = <float> 0.0
    else:
        rgb_.b = <float>255.0 * pow(rgb_.b, ADOBE_GAMMA)


    # CAP the RGB values 0 .. 255
    if rgb_.r > 255:
        rgb_.r = <float> 255.0

    if rgb_.g > 255:
        rgb_.g = <float> 255.0

    if rgb_.b > 255:
        rgb_.b = <float> 255.0

    rgb_.r = <float> rgb_.r
    rgb_.g = <float> rgb_.g
    rgb_.b = <float> rgb_.b

    return rgb_



@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cpdef inline xyz adobe98_xyz(float r, float g, float b, str ref='D65')nogil:
    """
    Conversion from 24-bit Adobe RGB to XYZ (D65)
   
    e.g
    >>>xyz = adobe98_xyz(255, 0, 0)
    {'x': 57.673091888427734, 'y': 29.737689971923828, 'z': 2.703429937362671}
    
    >>>x, y, z = adobe98_xyz(255, 0, 0).values()
    
    XYZ values are not normalized 

    :param r: Red color components in range 0..255 
    :param g: Green color components in range 0..255 
    :param b: Blue color components in range 0..255 
    :param ref: reference 'D50' or 'D65' default D65
    :return : return xyz structure containing x,y,z values, this will be 
        identical to a dictionary in python e.g 
        {'x': 57.673091888427734, 'y': 29.737689971923828, 'z': 2.703429937362671}

    """
    return adobe98_xyz_c(r, g, b, ref)

@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cdef inline xyz adobe98_xyz_c(float r, float g, float b, str ref='D65')nogil:
    """
    Conversion from 24-bit Adobe RGB to XYZ (D65)
   
    e.g
    >>>xyz = adobe98_xyz_c(255, 0, 0)
    {'x': 57.673091888427734, 'y': 29.737689971923828, 'z': 2.703429937362671}
    
    >>>x, y, z = adobe98_xyz_c(255, 0, 0).values()
    
    XYZ values are not normalized 
    
    :param r: Red color in range [0.0, 255.0]
    :param g: Green color in  range [0.0, 255.0]
    :param b: Blue color in range [0.0, 255.0]
    :param ref: reference 'D50' or 'D65' default D65
    :return : return xyz structure containing x,y,z values, this will be 
        identical to a dictionary in python e.g
         {'x': 57.673091888427734, 'y': 29.737689971923828, 'z': 2.703429937362671}

    """
    cdef:
        xyz xyz_

    with gil:
        ref = ref.upper()

    if ref != 'D50' and ref != 'D65':
        with gil:
            raise ValueError("\nAttribute ref must be D50 or D65")

    r = pow(r * _1_255, <float>2.199)
    g = pow(g * _1_255, <float>2.199)
    b = pow(b * _1_255, <float>2.199)

    # Adobe 1998 Calibration D65
    if ref == 'D65':
        xyz_.x = r * <float> 0.5767309 + g * <float> 0.1855540 + b * <float> 0.1881852
        xyz_.y = r * <float> 0.2973769 + g * <float> 0.6273491 + b * <float> 0.0752741
        xyz_.z = r * <float> 0.0270343 + g * <float> 0.0706872 + b * <float> 0.9911085

    if ref == 'D50':
        xyz_.x = r * <float> 0.6097559 + g * <float> 0.2052401 + b * <float> 0.1492240
        xyz_.y = r * <float> 0.3111242 + g * <float> 0.6256560 + b * <float> 0.0632197
        xyz_.z = r * <float> 0.0194811 + g * <float> 0.0608902 + b * <float> 0.7448387

    xyz_.x *= <float> 100.0
    xyz_.y *= <float> 100.0
    xyz_.z *= <float> 100.0

    return xyz_


@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cpdef inline xyz rgb_to_xyz(float r, float g, float b, str ref='D65')nogil:
    """
    sRGB to CIE XYZ (simple precision) D65/2 & D50°

    Python hook method 
    
    e.g
    
    >>>xyz = rgb_to_xyz(255, 0, 0)
    {'x': 41.24563980102539, 'y': 21.267290115356445, 'z': 1.9333901405334473}
    
    >>>x, y, z = rgb_to_xyz(255, 0, 0).values()
    
    
    Color component rgb values are in the range of 0 to 255 
    Like most of RGB to XYZ algorithm out here, this algorithm does 
    not control the capping of RGB values.

    :param r: float; Red components in range 0.0 .. 255.0 
    :param g: float; Green component in range 0.0 .. 255.0
    :param b: float; Blue component in range 0.0..255.0
    :param ref: reference 'D50' or 'D65' default D65
    :return : tuple; XYZ tuple, float values 0.0 .. 1.0. 
        This will be identical to a dictionary in python e.g 
        {'x': 41.24563980102539, 'y': 21.267290115356445, 'z': 1.9333901405334473}
    """

    return rgb_to_xyz_c(r, g, b, ref)



@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cdef inline xyz rgb_to_xyz_c(float r, float g, float b, str ref='D65') nogil:
    """
    sRGB to CIE XYZ (simple precision) D65/2° & D50
    
    e.g
    
    >>>xyz = rgb_to_xyz_c(255, 0, 0)
    {'x': 41.24563980102539, 'y': 21.267290115356445, 'z': 1.9333901405334473}
    
    >>>x, y, z = rgb_to_xyz_c(255, 0, 0).values()
    
    Prefer calling rgb_to_xyz from Python, otherwise call rgb_to_xyz_c
    from cython code (faster)
    
    Color component rgb values are in the range of 0 to 255
    Like most of RGB to XYZ algorithm out here, this algorithm does 
    not control the capping of RGB values.
    
    :param r: float; Red components in range 0 .. 255 
    :param g: float; Green component in range 0 .. 255
    :param b: float; Blue component in range 0..255
    :param ref: reference 'D50' or 'D65' default D65
    :return : tuple; XYZ tuple, float values 0 .. 1.0. 
        This will be identical to a dictionary in python e.g
        {'x': 41.24563980102539, 'y': 21.267290115356445, 'z': 1.9333901405334473}
       
    
    """
    cdef:
        xyz xyz_

    with gil:
        ref = ref.upper()

    if ref!='D65' and ref!='D50':
        with gil:
            raise ValueError('\nAttribute ref must be D65 or D50.')

    # No capping
    # if r > 255.0: r = 255.0
    # if g > 255.0: g = 255.0
    # if b > 255.0: b = 255.0

    r *= _1_255
    g *= _1_255
    b *= _1_255

    if r > 0.04045:
        r = ((r + <float>0.055) / <float>1.055 ) ** <float>2.4
    else:
        r /= <float>12.92

    if g > 0.04045:
        g = ((g + <float>0.055) / <float>1.055 ) ** <float>2.4
    else:
        g /= <float>12.92

    if b > 0.04045:
        b = ((b + <float>0.055) / <float>1.055 ) ** <float>2.4
    else:
        b /= <float>12.92

    r *= <float>100.0
    g *= <float>100.0
    b *= <float>100.0

    # These gamma-expanded values (sometimes called "linear values" or "linear-light values")
    # are multiplied by a matrix to obtain CIE XYZ (the matrix has infinite precision, any
    # change in its values or adding not zeroes is not allowed)

    if ref == 'D65':
        # d65
        xyz_.x = r * <float>0.4124564 + g * <float>0.3575761 + b * <float>0.1804375
        xyz_.y = r * <float>0.2126729 + g * <float>0.7151522 + b * <float>0.0721750
        xyz_.z = r * <float>0.0193339 + g * <float>0.1191920 + b * <float>0.9503041

    if ref == 'D50':
        # d50
        xyz_.x = r * <float> 0.4360747 + g * <float> 0.3850649 + b * <float> 0.1430804
        xyz_.y = r * <float> 0.2225045 + g * <float> 0.7168786 + b * <float> 0.0606169
        xyz_.z = r * <float> 0.0139322 + g * <float> 0.0971045 + b * <float> 0.7141733

    return xyz_


@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cpdef inline rgb xyz_to_rgb(float x, float y, float z, str ref='D65')nogil:
    """
    CIE XYZ to sRGB (simple precision) D65/2° & D50
    
    e.g 
    
    >>>rgb = xyz_to_rgb(41.24, 21.267, 1.933)
    {'r': 254.98020935058594, 'g': 0.16205523908138275, 'b': 0.0}
    
    >>>r, g, b = xyz_to_rgb(41.24, 21.267, 1.933).values()
    
    D65 - 2° standard colorimetric observer for CIE XYZ
    Returned rgb values are capped from 0.0 - 255.0 
    Python hook method 
    
    :param x: X color whose components are in the nominal range [0.0, 1.0]
    :param y: Y color whose components are in the nominal range [0.0, 1.0]
    :param z: Z color whose components are in the nominal range [0.0, 1.0]
    :param ref: reference 'D50' or 'D65' default D65
    :return : return RGB structure containing RGB values in range [0..255], this will be 
        identical to a dictionary in python e.g
        {'r': 254.98020935058594, 'g': 0.16205523908138275, 'b': 0.0}

    """
    return xyz_to_rgb_c(x, y, z, ref)


@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cdef inline rgb xyz_to_rgb_c(float x, float y, float z, str ref='D65')nogil:
    """
    CIE XYZ to sRGB (simple precision) D65/2° & D50
    
    e.g 
    
    >>>rgb = xyz_to_rgb_c(41.24, 21.267, 1.933)
    {'r': 254.98020935058594, 'g': 0.16205523908138275, 'b': 0.0}
    
    >>>r, g, b = xyz_to_rgb_c(41.24, 21.267, 1.933).values()
    
    D65 - 2° standard colorimetric observer for CIE XYZ
    Prefer calling xyz_to_rgb from python instead otherwise use 
    xyz_to_rgb_c from cython code
    
    Returned rgb values are capped from 0.0 - 255.0 
    
    :param x: X color whose components are in the nominal range [0.0, 1.0]
    :param y: Y color whose components are in the nominal range [0.0, 1.0]
    :param z: Z color whose components are in the nominal range [0.0, 1.0]
    :param ref: reference 'D50' or 'D65' default D65
    :return : return RGB structure containing RGB values in range [0..255], this will be 
        identical to a dictionary in python e.g
        {'r': 254.98020935058594, 'g': 0.16205523908138275, 'b': 0.0}


    """
    cdef:
        rgb rgb_

    # The first step in the calculation of sRGB from CIE XYZ is a linear
    # transformation, which may be carried out by a matrix multiplication.
    # (The numerical values below match those in the official sRGB specification,
    # which corrected small rounding errors in the original publication by
    # sRGB's creators, and assume the 2° standard colorimetric observer
    # for CIE XYZ.) This matrix depends on the bitdepth.

    with gil:
        ref = ref.upper()

    if ref!='D65' and ref!='D50':
        with gil:
            raise ValueError('\nAttribute ref must be D65 or D50.')

    if ref == 'D65':
        # Calibration D65
        rgb_.r = x * +<float>3.2404542 + y * -<float>1.5371385  + z * -<float>0.4985314
        rgb_.g = x * -<float>0.9692660 + y * +<float>1.8760108 + z * +<float>0.0415560
        rgb_.b = x * +<float>0.0556434 + y * -<float>0.2040259 + z * +<float>1.0572252

    if ref == 'D50':
        # d50
        rgb_.r = x * +<float> 3.1338561 + y * -<float> 1.6168667 + z * -<float> 0.4906146
        rgb_.g = x * -<float> 0.9787684 + y * +<float> 1.9161415 + z * +<float> 0.0334540
        rgb_.b = x * +<float> 0.0719453 + y * -<float> 0.2289914 + z * +<float> 1.4052427

    rgb_.r*=_1_100
    rgb_.g*=_1_100
    rgb_.b*=_1_100

    # These linear RGB values are not the final result;
    # gamma correction must still be applied. The following formula transforms
    # the linear values into sRGB:
    if rgb_.r <= <float>0.0031308:
        rgb_.r  = <float>12.92 * rgb_.r
    else:
        rgb_.r = <float>1.055 * (rgb_.r ** _1_24) - <float>0.055

    if rgb_.g <= <float>0.0031308:
        rgb_.g = <float> 12.92 * rgb_.g
    else:
        rgb_.g = <float> 1.055 * (rgb_.g ** _1_24) - <float>0.055

    if rgb_.b <= <float>0.0031308:
        rgb_.b = <float>12.92 * rgb_.b
    else:
        rgb_.b = <float>1.055 * (rgb_.b ** _1_24) - <float>0.055


    rgb_.r*=<float>255.0
    rgb_.g*=<float>255.0
    rgb_.b*=<float>255.0

    # CAP the RGB values 0 .. 255
    if rgb_.r < 0:
        rgb_.r = <float> 0.0
    if rgb_.r > 255:
        rgb_.r = <float> 255.0

    if rgb_.g < 0:
        rgb_.g = <float> 0.0
    if rgb_.g > 255:
        rgb_.g = <float> 255.0

    if rgb_.b < 0:
        rgb_.b = <float> 0.0
    if rgb_.b > 255:
        rgb_.b = <float> 255.0

    # round a float
    # rgb_.r = <float>round_f(rgb_.r)
    # rgb_.g = <float>round_f(rgb_.g)
    # rgb_.b = <float>round_f(rgb_.b)
    rgb_.r = <float>rgb_.r
    rgb_.g = <float>rgb_.g
    rgb_.b = <float>rgb_.b

    return rgb_



@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cpdef lab xyz_to_cielab(
        float x,
        float y,
        float z,
        const float [:] model=cielab_model_d65,
        bint format_8b = False
)nogil:
    """
    CONVERT XYZ VALUES TO CIELAB  

    e.g
    
    support illuminant model ['a','c','e','d50', 'd55', 'd65', 'icc']
    
    >>>lab = xyz_to_cielab(41.245, 21.267, 1.9333)  # default d65 
    {'l': 53.240478515625, 'a': 80.10113525390625, 'b': 67.20298767089844}
    
    >>>lab = xyz_to_cielab(41.245, 21.267, 1.9333, model=model_d50)  # d50
    {'l': 53.240478515625, 'a': 78.28646850585938, 'b': 62.14963912963867} 
    
    >>>l, a, b = xyz_to_cielab(41.245, 21.267, 1.9333, model=model_e).values()  # model e
    
    Python hook method 

    X, Y, Z describe the color stimulus considered 
    
    :param x: X color whose components are in the nominal range [0.0, 1.0]
    :param y: Y color whose components are in the nominal range [0.0, 1.0]
    :param z: Z color whose components are in the nominal range [0.0, 1.0]
    :param model: illuminant color model
    :param format_8b : True | False; Default False; Set this variable to True when using 8-bit images/surfaces
        otherwise set it to False for 24- 32-bit images (float32) 
    :return : return structure lab containing the l, a, b values of the xyz -> lab conversion. 
        This will be identical to a dictionary in python e.g 
        {'l': 53.240478515625, 'a': 78.28646850585938, 'b': 62.14963912963867} 

    """
    return xyz_to_cielab_c(x, y, z, model, format_8b)

@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cdef inline lab xyz_to_cielab_c(
        float x,
        float y,
        float z,
        const float [:] model=cielab_model_d65,
        bint format_8b = False
)nogil:

    """
    CONVERT XYZ VALUES TO LAB 
    
    e.g
    
    support illuminant model ['a','c','e','d50', 'd55', 'd65', 'icc']
    
    >>>lab = xyz_to_cielab_c(41.245, 21.267, 1.9333)  # default d65 
    {'l': 53.240478515625, 'a': 80.10113525390625, 'b': 67.20298767089844}
    
    >>>lab = xyz_to_cielab_c(41.245, 21.267, 1.9333, model=model_d50)  # d50
    {'l': 53.240478515625, 'a': 78.28646850585938, 'b': 62.14963912963867} 
    
    >>>l, a, b = xyz_to_cielab_c(41.245, 21.267, 1.9333, model=model_e).values()  # model e
    
    Cython function doing the heavy lifting. 
    Prefer to call xyz_to_cielab instead from Python otherwise 
    use xyz_to_cielab_c from cython code.
     
    X, Y, Z describe the color stimulus considered 
    
    :param x: X color whose components are in the nominal range [0.0, 1.0]
    :param y: Y color whose components are in the nominal range [0.0, 1.0]
    :param z: Z color whose components are in the nominal range [0.0, 1.0]
    :param model: illuminant color model
    :param format_8b : True | False; Default False; Set this variable to True when using 8-bit images/surfaces
        otherwise set it to False for 24 - 32-bit images (float32) 
    :return : return structure lab containing the l, a, b values of the xyz -> lab conversion.
        this will be identical to a dictionary in python e.g
        {'l': 53.240478515625, 'a': 78.28646850585938, 'b': 62.14963912963867}
 
    """
    if len(model) != 3:
        with gil:
            raise TypeError(
                'Argument model has an invalid length of %s; expecting 3' % len(model))

    cdef lab lab_
    cdef float refX, refY, refZ

    refX = model[0]
    refY = model[1]
    refZ = model[2]

    x/= refX * <float>100.0
    y/= refY * <float>100.0
    z/= refZ * <float>100.0

    # 903.3Actual CIE standard
    # k / 116.0 = 7.787
    if x > <float>0.008856:
        x = <float>pow(x, _1_3)
    else:
        x = (<float>7.787 * x) + LAMBDA

    if y > <float>0.008856:
        y = <float>pow(y, _1_3)
    else:
        y = (<float>7.787 * y) + LAMBDA

    if z > <float>0.008856:
        z = <float>pow(z, _1_3)
    else:
        z = (<float>7.787 * z) + LAMBDA

    lab_.l = <float>116.0 * y - <float>16.0
    lab_.a = <float>500.0 * (x - y)
    lab_.b = <float>200.0 * (y - z)

    if format_8b:
        lab_.l *= _255_100
        lab_.a += <float>128.0
        lab_.b += <float>128.0

    return lab_



@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cdef inline float inv_f_t(float c)nogil:
    """
    From CIELAB to CIEXYZ reverse transformation (function used by cielab_to_xyz_c)
    This function does not have any other purpose and should not 
    be call from Python
    """
    if c > SIGMA:
        c = <float>pow(c, <float>3.0)
    else:
        c = <float>3.0 * SIGMA_SQR * (c - LAMBDA)
    return c


@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cpdef inline xyz cielab_to_xyz(
        float l ,
        float a,
        float b,
        const float [:] model=cielab_model_d65,
        bint format_8b = False
)nogil:
    """
    CONVERT CIELAB to XYZ 
    
    e.g:
    
    support illuminant model ['a','c','e','d50', 'd55', 'd65', 'icc']
    
    >>>xyz = cielab_to_xyz(l, a, b, model=model_d65)
    {'x': 39.19924545288086, 'y': 21.26700210571289, 'z': 2.1049764156341553}
    
    >>>xyz = cielab_to_xyz(l, a, b, model=model_d55)  # d55 
    
    >>>x, y, z = cielab_to_xyz(l, a, b, model=model_d65).values()
    
    Python hook method 

    The three coordinates of CIELAB represent the lightness of the color (L* = 0 yields 
    black and L* = 100 indicates diffuse white; specular white may be higher), its position
    between magenta and green (a*, where negative values indicate green and positive values
    indicate magenta) and its position between yellow and blue (b*, where negative values
    indicate blue and positive values indicate yellow). 
    
    :param l : float; l perceptual lightness
    :param a : float; a*, where negative values indicate green and positive values indicate magenta 
        and its position between yellow and blue
    :param b : float; b*, where negative values indicate blue and positive values indicate yellow
    :param model : memoryview array shape (3,) containing the illuminant values 
    :param format_8b : True | False; Default False; Set this variable to True when using 8-bit images/surfaces
        otherwise set it to False for 24 - 32-bit images (float32)
    :return : xyz; structure containing XYZ values. This will be identical to a dictionary in python e.g
        {'x': 39.19924545288086, 'y': 21.26700210571289, 'z': 2.1049764156341553}

    Below all the compatible illuminant models
    "a"
    CIE standard illuminant A, [1.0985, 1.0000, 0.3558].
    Simulates typical, domestic, tungsten-filament lighting with correlated
     color temperature of 2856 K.
    "c"
    CIE standard illuminant C, [0.9807, 1.0000, 1.1822]. Simulates average or
     north sky daylight with correlated color temperature of 6774 K.
     Deprecated by CIE.
    "e"
    Equal-energy radiator, [1.000, 1.000, 1.000]. Useful as a theoretical reference.
    "d50"
    CIE standard illuminant D50, [0.9642, 1.0000, 0.8251]. Simulates warm daylight
     at sunrise or sunset with correlated color temperature of 5003 K.
     Also known as horizon light.
    "d55"
    CIE standard illuminant D55, [0.9568, 1.0000, 0.9214]. Simulates mid-morning
    or mid-afternoon daylight with correlated color temperature of 5500 K.
    "d65"
    CIE standard illuminant D65, [0.9504, 1.0000, 1.0888].
     Simulates noon daylight with correlated color temperature of 6504 K.
    "icc"
    Profile Connection Space (PCS) illuminant used in ICC profiles.
     Approximation of [0.9642, 1.000, 0.8249] using fixed-point, signed,
     32-bit numbers with 16 fractional bits. Actual value:
     [31595,32768, 27030]/32768.

    """
    return cielab_to_xyz_c(l, a, b, model, format_8b)

@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cdef inline xyz cielab_to_xyz_c(
        float l ,
        float a,
        float b,
        const float [:] model=cielab_model_d65,
        bint format_8b = False
)nogil:
    """
    CONVERT CIELAB to XYZ 
    
    e.g:
    
    support illuminant model ['a','c','e','d50', 'd55', 'd65', 'icc']
    
    >>>xyz = cielab_to_xyz_c(l, a, b, model=model_d65)
    {'x': 39.19924545288086, 'y': 21.26700210571289, 'z': 2.1049764156341553}
    
    >>>xyz = cielab_to_xyz_c(l, a, b, model=model_d55)  # d55 
    
    >>>x, y, z = cielab_to_xyz_c(l, a, b, model=model_d65).values()
    
    Cython function doing the heavy lifting.
    Prefer calling this function from cython code and 
    cielab_to_xyz from python instead
    
    :param l : float; l perceptual lightness
    :param a : float; a*, where negative values indicate green and positive values indicate magenta 
        and its position between yellow and blue
    :param b : float; b*, where negative values indicate blue and positive values indicate yellow
    :param model : memoryview array shape (3,) containing the illuminant values 
    :param format_8b : True | False; Default False; Set this variable to True when using 8-bit images/surfaces
        otherwise set it to False for 24 - 32-bit images (float32)
    :return : xyz; structure containing XYZ values. This will be identical to a dictionary in python e.g
        {'x': 39.19924545288086, 'y': 21.26700210571289, 'z': 2.1049764156341553}
    """
    cdef:
        xyz xyz_
        float refX, refY, refZ
        float tmp_

    if len(model) != 3:
        with gil:
            raise TypeError(
                'Argument model has an invalid length of %s; expecting 3' % len(model))

    if format_8b:
        l /= _255_100
        a -= <float>128.0
        b -= <float>128.0

    refX = model[0]
    refY = model[1]
    refZ = model[2]


    refX *= <float> 100.0
    refY *= <float> 100.0
    refZ *= <float> 100.0

    tmp_ = (l + <float> 16.0) * _1_116
    xyz_.x = refX * inv_f_t(tmp_ + a * _1_500)
    xyz_.y = refY * inv_f_t(tmp_)
    xyz_.z = refZ * inv_f_t(tmp_ - b * _1_200)


    return xyz_

@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cpdef inline lab rgb_to_cielab(
        float r,
        float g,
        float b,
        const float [:] model=cielab_model_d65,
        bint format_8b = False
)nogil:
    """
    CONVERT RGB COLOR VALUES TO CIELAB COLOR SPACE 
    
    e.g: 
    support illuminant model ['a','c','e','d50', 'd55', 'd65', 'icc']
    
    >>>lab = rgb_to_cielab(255, 255, 255)
    {'l': 100.0, 'a': 0.012278556823730469, 'b': -0.0018358230590820312}
    
    >>>l, a, b = rgb_to_cielab(255, 255, 255, model=model_c).values()

    Python hook method
    
    :param r : float; Red component value 0..255 
    :param g : float; green component value 0..255
    :param b : float; blue component value 0..255
    :param model : memoryview array shape (3,) containing the illuminant values
    :param format_8b : True | False; Default False; Set this variable to True when using 8-bit images/surfaces
        otherwise set it to False for 24-32-bit images (float32)  
    :return : lab; structure containing the cielab values (l, a, b) type float simple precision.
        This will be identical to a dictionary in python e.g 
        {'l': 100.0, 'a': 0.012278556823730469, 'b': -0.0018358230590820312}
    """
    return rgb_to_cielab_c(r, g, b, model, format_8b)


@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cdef inline lab rgb_to_cielab_c(
        float r,
        float g,
        float b,
        const float [:] model=cielab_model_d65,
        bint format_8b = False
)nogil:
    """
    CONVERT RGB COLOR VALUES TO CIELAB COLOR SPACE 
    
    e.g: 
    support illuminant model ['a','c','e','d50', 'd55', 'd65', 'icc']
    
    >>>lab = rgb_to_cielab_c(255, 255, 255)
    {'l': 100.0, 'a': 0.012278556823730469, 'b': -0.0018358230590820312}
    
    >>>l, a, b = rgb_to_cielab_c(255, 255, 255, model=model_c).values()
    
    Python hook method
    
    :param r : float; Red component value 0..255 
    :param g : float; green component value 0..255
    :param b : float; blue component value 0..255
    :param model : memoryview array shape (3,) containing the illuminant values 
    :param format_8b : True | False; Default False; Set this variable to True when using 8-bit images/surfaces
        otherwise set it to False for 24-32-bit images (float32) 
    :return : lab; structure containing the cielab values (l, a, b) type float simple precision.
        This will be identical to a dictionary in python e.g 
        {'l': 100.0, 'a': 0.012278556823730469, 'b': -0.0018358230590820312}
    
    Below all the compatible illuminant models
    "a"
    CIE standard illuminant A, [1.0985, 1.0000, 0.3558].
    Simulates typical, domestic, tungsten-filament lighting with correlated
     color temperature of 2856 K.
    "c"
    CIE standard illuminant C, [0.9807, 1.0000, 1.1822]. Simulates average or
     north sky daylight with correlated color temperature of 6774 K.
     Deprecated by CIE.
    "e"
    Equal-energy radiator, [1.000, 1.000, 1.000]. Useful as a theoretical reference.
    "d50"
    CIE standard illuminant D50, [0.9642, 1.0000, 0.8251]. Simulates warm daylight
     at sunrise or sunset with correlated color temperature of 5003 K.
     Also known as horizon light.
    "d55"
    CIE standard illuminant D55, [0.9568, 1.0000, 0.9214]. Simulates mid-morning
    or mid-afternoon daylight with correlated color temperature of 5500 K.
    "d65"
    CIE standard illuminant D65, [0.9504, 1.0000, 1.0888].
     Simulates noon daylight with correlated color temperature of 6504 K.
    "icc"
    Profile Connection Space (PCS) illuminant used in ICC profiles.
     Approximation of [0.9642, 1.000, 0.8249] using fixed-point, signed,
     32-bit numbers with 16 fractional bits. Actual value:
     [31595,32768, 27030]/32768.
    """
    cdef xyz xyz_
    xyz_ = rgb_to_xyz_c(r, g, b)  # forced d65
    return xyz_to_cielab_c(xyz_.x, xyz_.y, xyz_.z, model, format_8b)

@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cpdef inline rgb cielab_to_rgb(
        float l,
        float a,
        float b,
        const float [:] model=cielab_model_d65,
        bint format_8b = False
)nogil:
    """
    CONVERT RGB COLOR VALUES TO CIELAB COLOR SPACE
    
    e.g: 
    
    support illuminant model ['a','c','e','d50', 'd55', 'd65', 'icc']
      
    >>>r, g, b = cielab_to_rgb(l, a, b).values()
    
    >>>rgb = cielab_to_rgb(l, a, b)
    {'r': 255.0, 'g': 252.87266540527344, 'b': 0.0}

    Python hook method

    The three coordinates of CIELAB represent the lightness of the color (L* = 0 yields 
    black and L* = 100 indicates diffuse white; specular white may be higher), its position
    between magenta and green (a*, where negative values indicate green and positive values
    indicate magenta) and its position between yellow and blue (b*, where negative values
    indicate blue and positive values indicate yellow). 
    
    :param l : float; l perceptual lightness
    :param a : float; a*, where negative values indicate green and positive values indicate magenta 
        and its position between yellow and blue
    :param b : float; b*, where negative values indicate blue and positive values indicate yellow
    :param model : memoryview array shape (3,) containing the illuminant values 
    :param format_8b : True | False; Default False; Set this variable to True when using 8-bit images/surfaces
        otherwise set it to False for 24-32-bit images (float32)
    :return : rgb; structure containing RGB values (r, g, b) type float simple precision.
        This will be identical to a dictionary in python e.g:
        {'r': 255.0, 'g': 252.87266540527344, 'b': 0.0}
    """

    return cielab_to_rgb_c(l, a, b, model, format_8b)



@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cdef inline rgb cielab_to_rgb_c(
        float l,
        float a,
        float b,
        const float [:] model=cielab_model_d65,
        bint format_8b = False
)nogil:
    """
    CONVERT RGB COLOR VALUES TO CIELAB COLOR SPACE 

    e.g: 
    
    support illuminant model ['a','c','e','d50', 'd55', 'd65', 'icc']
      
    >>>r, g, b = cielab_to_rgb_c(l, a, b).values()
    
    >>>rgb = cielab_to_rgb_c(l, a, b)
    {'r': 255.0, 'g': 252.87266540527344, 'b': 0.0}
    
    Python hook method

    The three coordinates of CIELAB represent the lightness of the color (L* = 0 yields 
    black and L* = 100 indicates diffuse white; specular white may be higher), its position
    between magenta and green (a*, where negative values indicate green and positive values
    indicate magenta) and its position between yellow and blue (b*, where negative values
    indicate blue and positive values indicate yellow). 
    
    :param l : float; l perceptual lightness
    :param a : float; a*, where negative values indicate green and positive values indicate magenta 
        and its position between yellow and blue
    :param b : float; b*, where negative values indicate blue and positive values indicate yellow
    :param model : memoryview array shape (3,) containing the illuminant values 
    :param format_8b : True | False; Default False; Set this variable to True when using 8-bit images/surfaces
        otherwise set it to False for 24- 32-bit images (float32)
    :return : rgb; structure containing RGB values (r, g, b) type float simple precision.
        This will be identical to a dictionary in python e.g: 
        {'r': 255.0, 'g': 252.87266540527344, 'b': 0.0}

    Below all the compatible illuminant models

    "a"
    CIE standard illuminant A, [1.0985, 1.0000, 0.3558].
    Simulates typical, domestic, tungsten-filament lighting with correlated
     color temperature of 2856 K.
    "c"
    CIE standard illuminant C, [0.9807, 1.0000, 1.1822]. Simulates average or
     north sky daylight with correlated color temperature of 6774 K.
     Deprecated by CIE.
    "e"
    Equal-energy radiator, [1.000, 1.000, 1.000]. Useful as a theoretical reference.
    "d50"
    CIE standard illuminant D50, [0.9642, 1.0000, 0.8251]. Simulates warm daylight
     at sunrise or sunset with correlated color temperature of 5003 K.
     Also known as horizon light.
    "d55"
    CIE standard illuminant D55, [0.9568, 1.0000, 0.9214]. Simulates mid-morning
    or mid-afternoon daylight with correlated color temperature of 5500 K.
    "d65"
    CIE standard illuminant D65, [0.9504, 1.0000, 1.0888].
     Simulates noon daylight with correlated color temperature of 6504 K.
    "icc"
    Profile Connection Space (PCS) illuminant used in ICC profiles.
     Approximation of [0.9642, 1.000, 0.8249] using fixed-point, signed,
     32-bit numbers with 16 fractional bits. Actual value:
     [31595,32768, 27030]/32768.
    """
    cdef xyz xyz_
    xyz_ = cielab_to_xyz_c(l, a, b, model, format_8b)
    return xyz_to_rgb_c(xyz_.x, xyz_.y, xyz_.z)


@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cpdef rgb_2_cielab(
        unsigned char[:, :, :] rgb_array,
        str illuminant ='d65',
        bint format_8b=False
):
    """
    CIELAB color space 
    Convert RGB image to CIELAB with specific illuminant
    
    e.g:
    
    >>> arr = numpy.empty((640, 480, 3), dtype=numpy.uint8)
    >>> cielab_array = rgb_2_cielab(arr)
    >>> cielab_array = rgb_2_cielab(arr, illuminant='d50')

    
    Python hook method 
    
    :param rgb_array : numpy.ndarray shape (w, h, 3) containing RGB pixel values (uint8) 
    :param illuminant: Illuminant white point; sting can be 'a','c','e','d50', 'd55', 'd65', 'icc' 
        see below for more details about the illuminant argument.
    :param format_8b : True | False; Default False; Set this variable to True when using 8-bit images/surfaces
        otherwise set it to False for 24 - 32-bit images (float32) 
    :return : Image converted to CIELAB color space (conversion in simple precision).
        
    "a"
    CIE standard illuminant A, [1.0985, 1.0000, 0.3558].
    Simulates typical, domestic, tungsten-filament lighting with correlated
     color temperature of 2856 K.
    "c"
    CIE standard illuminant C, [0.9807, 1.0000, 1.1822]. Simulates average or
     north sky daylight with correlated color temperature of 6774 K.
     Deprecated by CIE.
    "e"
    Equal-energy radiator, [1.000, 1.000, 1.000]. Useful as a theoretical reference.
    "d50"
    CIE standard illuminant D50, [0.9642, 1.0000, 0.8251]. Simulates warm daylight
     at sunrise or sunset with correlated color temperature of 5003 K.
     Also known as horizon light.
    "d55"
    CIE standard illuminant D55, [0.9568, 1.0000, 0.9214]. Simulates mid-morning
    or mid-afternoon daylight with correlated color temperature of 5500 K.
    "d65"
    CIE standard illuminant D65, [0.9504, 1.0000, 1.0888].
     Simulates noon daylight with correlated color temperature of 6504 K.
    "icc"
    Profile Connection Space (PCS) illuminant used in ICC profiles.
     Approximation of [0.9642, 1.000, 0.8249] using fixed-point, signed,
     32-bit numbers with 16 fractional bits. Actual value:
     [31595,32768, 27030]/32768.
    """

    cdef float [:] illuminant_model

    illuminant_model = cielab_illuminant_c(illuminant)

    return rgb_2_cielab_c(rgb_array, illuminant_model, format_8b)



@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cdef rgb_2_cielab_c(
        unsigned char[:, :, :] rgb_array,
        const float [:] illuminant_model = cielab_model_d65,
        bint format_8b = False
):

    """
    CIELAB color space (convert RGB array to CIELAB with specific illuminant) 
    
    e.g:
    
    >>> arr = numpy.empty((640, 480, 3), dtype=numpy.uint8)
    >>> cielab_array = rgb_2_cielab_c(arr)
    >>> cielab_array = rgb_2_cielab_c(arr, illuminant='d50')
    
    Cython function doing the heavy lifting, prefer calling this function from cython code
    and call cielab from python instead

    :param rgb_array : numpy.ndarray shape (w, h, 3) containing RGB pixel values (uint8)
    :param illuminant_model: Illuminant white point; string can be 'a','c','e','d50', 'd55', 'd65', 'icc'
        default is cielab_model_d65
    :param format_8b : True | False; Default False; Set this variable to True when using 8-bit images/surfaces
        otherwise set it to False for 24 - 32-bit images (float32)
    :return : numpy array shape(w h, 3) converted to CIELAB color space (conversion in simple precision)
    """


    cdef Py_ssize_t w, h, dim

    try:
        w, h, dim = rgb_array.shape[:3]
    except Exception as e:
        raise ValueError(
            "adobe98_array argument must be shape (h, w, 3) type uint8.\n %s " % e)

    if dim != 3 and dim != 4:
        raise TypeError(
            "adobe98_array invalid dimensions "
            "for RGB or RGBA array pixels; got (%s, %s, %s).\n" % (w, h, dim))

    cdef:
        int i, j
        float [:, :, ::1 ] tmp = empty((w, h, 3), dtype=numpy.float32)
        float l_, a_, b_
        float refX, refY, refZ
        float r, g, b, x, y, z


    with nogil:
        for i in prange(w, schedule=SCHEDULE, num_threads=THREADS):
            for j in range(h):

                # RGB values in nominal range [0, 1].
                r = <float>rgb_array[i, j, 0] * _1_255
                g = <float>rgb_array[i, j, 1] * _1_255
                b = <float>rgb_array[i, j, 2] * _1_255

                # The same operation is performed on all three channels,
                # but the operation depends on the companding function associated with the RGB color system.

                # Inverse sRGB Companding
                if r > <float>0.04045:
                    r = ((r + <float> 0.055) / <float> 1.055) ** <float> 2.4
                else:
                    r = r / <float> 12.92

                if g > <float>0.04045:
                    g = ((g + <float> 0.055) / <float> 1.055) ** <float> 2.4
                else:
                    g = g / <float> 12.92

                if b > <float>0.04045:
                    b = ((b + <float> 0.055) / <float> 1.055) ** <float> 2.4
                else:
                    b = b / <float> 12.92

                r = r * <float> 100.0
                g = g * <float> 100.0
                b = b * <float> 100.0

                # These gamma-expanded values (sometimes called "linear values" or "linear-light values")
                # are multiplied by a matrix to obtain CIE XYZ (the matrix has infinite precision, any
                # change in its values or adding not zeroes is not allowed)

                # Calibration D65
                x = r * <float> 0.4124564 + g * <float> 0.3575761 + b * <float> 0.1804375
                y = r * <float> 0.2126729 + g * <float> 0.7151522 + b * <float> 0.0721750
                z = r * <float> 0.0193339 + g * <float> 0.1191920 + b * <float> 0.9503041

                refX = illuminant_model[0]
                refY = illuminant_model[1]
                refZ = illuminant_model[2]

                x = x / (refX * <float> 100.0)
                y = y / (refY * <float> 100.0)
                z = z / (refZ * <float> 100.0)

                if x > <float>0.008856:
                    x = <float> pow(x, _1_3)
                else:
                    x = (<float> 7.787 * x) + LAMBDA

                if y > <float>0.008856:
                    y = <float> pow(y, _1_3)
                else:
                    y = (<float> 7.787 * y) + LAMBDA

                if z > <float>0.008856:
                    z = <float> pow(z, _1_3)
                else:
                    z = (<float> 7.787 * z) + LAMBDA

                l_ = <float> 116.0 * y - <float> 16.0
                a_ = <float> 500.0 * (x - y)
                b_ = <float> 200.0 * (y - z)

                if not format_8b:
                    l_ = l_ * _255_100
                    a_ = a_ + <float> 128.0
                    b_ = b_ + <float> 128.0

                tmp[i, j , 0] = l_
                tmp[i, j , 1] = a_
                tmp[i, j , 2] = b_

    return numpy.asarray(tmp, dtype=numpy.float32)


@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cpdef cielab_2_rgb(
        float [:, :, :] lab_array,
        str illuminant ='d65',
        bint format_8b=False
):
    """
    CIELAB color space
    Convert CIELAB image to RGB with specific illuminant

    e.g:
    
    >>> arr = numpy.empty((640, 480, 3), dtype=numpy.uint8)
    >>> rgb_array = cielab_2_rgb(arr)
    >>> rgb_array = cielab_2_rgb(arr, illuminant='d50')
    
    Python hook method

    :param lab_array : numpy.ndarray shape (w, h, 3) containing cielab values (l, a, b float32)
    :param illuminant: Illuminant white point; sting can be 'a','c','e','d50', 'd55', 'd65', 'icc'
        see below for more details about the illuminant argument.
    :param format_8b : True | False; Default False; Set this variable to True when using 8-bit images/surfaces
        otherwise set it to False for 24 - 32-bit images (float32)    
    :return : Image converted to CIELAB color space (conversion in simple precision).
    
    "a"
    CIE standard illuminant A, [1.0985, 1.0000, 0.3558].
    Simulates typical, domestic, tungsten-filament lighting with correlated
     color temperature of 2856 K.
    "c"
    CIE standard illuminant C, [0.9807, 1.0000, 1.1822]. Simulates average or
     north sky daylight with correlated color temperature of 6774 K.
     Deprecated by CIE.
    "e"
    Equal-energy radiator, [1.000, 1.000, 1.000]. Useful as a theoretical reference.
    "d50"
    CIE standard illuminant D50, [0.9642, 1.0000, 0.8251]. Simulates warm daylight
     at sunrise or sunset with correlated color temperature of 5003 K.
     Also known as horizon light.
    "d55"
    CIE standard illuminant D55, [0.9568, 1.0000, 0.9214]. Simulates mid-morning
    or mid-afternoon daylight with correlated color temperature of 5500 K.
    "d65"
    CIE standard illuminant D65, [0.9504, 1.0000, 1.0888].
     Simulates noon daylight with correlated color temperature of 6504 K.
    "icc"
    Profile Connection Space (PCS) illuminant used in ICC profiles.
     Approximation of [0.9642, 1.000, 0.8249] using fixed-point, signed,
     32-bit numbers with 16 fractional bits. Actual value:
     [31595,32768, 27030]/32768.
    """

    cdef float [:] illuminant_model

    illuminant_model = cielab_illuminant_c(illuminant)

    return cielab_2_rgb_c(lab_array, illuminant_model, format_8b)



@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cdef cielab_2_rgb_c(
        float [:, :, :] lab_array,
        const float [:] illuminant_model = cielab_model_d65,
        bint format_8b = False
):
    """
    CIELAB color space (convert CIELAB image to RGB with specific illuminant)

    e.g:
    
    >>> arr = numpy.empty((640, 480, 3), dtype=numpy.uint8)
    >>> rgb_array = cielab_2_rgb_c(arr)
    >>> rgb_array = cielab_2_rgb_c(arr, illuminant='d50')
    
    Cython function doing the heavy lifting, prefer calling this function from cython code
    and call cielab from python instead
    
    :param lab_array : numpy.ndarray shape (w, h, 3) containing cielab values (l, a, b float32)
    :param illuminant_model: Illuminant white point; sting can be 'a','c','e','d50', 'd55', 'd65', 'icc'
    :param format_8b : True | False; Default False; Set this variable to True when using 8-bit images/surfaces
        otherwise set it to False for 24-32-bit images (float32) 
    :return : numpy array shape(w h, 3) converted to CIELAB color space (conversion in simple precision)
    """


    cdef Py_ssize_t w, h, dim

    try:
        w, h, dim = lab_array.shape[:3]
    except Exception as e:
        raise ValueError("adobe98_array argument must be shape (h, w, 3) type uint8.\n %s " % e)

    if dim != 3 and dim != 4:
        raise TypeError(
            "adobe98_array invalid dimensions "
            "for RGB or RGBA array pixels; got (%s, %s, %s).\n" % (w, h, dim))

    cdef:
        int i, j
        float l_, a_, b_
        unsigned char [:, :, ::1 ] rbg_array = empty((w, h, 3), dtype=numpy.uint8)
        float r, g, b
        float x, y, z
        float refX, refY, refZ
        float tmp_

    with nogil:
        for i in prange(w, schedule=SCHEDULE, num_threads=THREADS):
            for j in range(h):

                l_ = lab_array[i, j, 0]
                a_ = lab_array[i, j, 1]
                b_ = lab_array[i, j, 2]

                if not format_8b:
                    # l_ = l_ / _255_100
                    l_ = l_ * _100_255
                    a_ = a_ - <float>128.0
                    b_ = b_ - <float>128.0

                refX = illuminant_model[0] * <float>100.0
                refY = illuminant_model[1] * <float>100.0
                refZ = illuminant_model[2] * <float>100.0

                tmp_ = (l_ + <float> 16.0) * _1_116
                x = refX * inv_f_t(tmp_ + a_ * _1_500)
                y = refY * inv_f_t(tmp_)
                z = refZ * inv_f_t(tmp_ - b_ * _1_200)

                # d65
                r = x * +<float> 3.2404542 + y * -<float> 1.5371385 + z * -<float> 0.4985314
                g = x * -<float> 0.9692660 + y * +<float> 1.8760108 + z * +<float> 0.0415560
                b = x * +<float> 0.0556434 + y * -<float> 0.2040259 + z * +<float> 1.0572252

                r = r * <float> 0.01
                g = g * <float> 0.01
                b = b * <float> 0.01

                # These linear RGB values are not the final result;
                # gamma correction must still be applied. The following formula transforms
                # the linear values into sRGB:
                if r <= <float>0.0031308:
                    r = <float> 12.92 * r
                else:
                    r = <float> 1.055 * (r ** _1_24) - <float> 0.055

                if g <= <float>0.0031308:
                    g = <float> 12.92 * g
                else:
                    g = <float> 1.055 * (g ** _1_24) - <float> 0.055

                if b <= <float>0.0031308:
                    b = <float> 12.92 * b
                else:
                    b = <float> 1.055 * (b ** _1_24) - <float> 0.055

                r *= <float> 255.0
                g *= <float> 255.0
                b *= <float> 255.0

                # CAP the RGB values 0 .. 255
                if r < 0:
                    r = <float> 0.0
                if r > 255:
                    r = <float> 255.0

                if g < 0:
                    g = <float> 0.0
                if g > 255:
                    g = <float> 255.0

                if b < 0:
                    b = <float> 0.0
                if b > 255:
                    b = <float> 255.0

                rbg_array[i, j , 0] = <unsigned char> round_f(r)
                rbg_array[i, j , 1] = <unsigned char> round_f(g)
                rbg_array[i, j , 2] = <unsigned char> round_f(b)

    return numpy.asarray(rbg_array)



@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cpdef WhiteBalance(
        unsigned char[:, :, :] rgb_array,
        float c1=1.0,
        str illuminant='D65',
        bint format_8b = False
):

    """
    The Gray World algorithm for illuminant estimation assumes that
    the average color of the world is gray, or achromatic. Therefore,
    it calculates the scene illuminant as the average RGB value in the image.
    
    e.g
    >>>arr = numpy.empty((640, 480, 3), dtype=numpy.uint8)
    >>>white_balance = WhiteBalance(arr)
    >>>white_balance = WhiteBalance(arr, illuminant='d50')
    
    :param rgb_array: numpy.ndarray containing RGB pixels uint8 values in range 0..255
    :param c1 :
    :param illuminant: Illuminant white point; sting can be 'a','c','e','d50', 'd55', 'd65', 'icc'
    :param format_8b : True | False; Default False; Set this variable to True when using 8-bit images/surfaces
        otherwise set it to False for 24 - 32-bit images (float32)
    :return : Returns numpy.ndarray shape (w, h, 3) type uint8; white balanced
    """

    cdef float [:] illuminant_model_

    illuminant = illuminant.upper()
    illuminant_model_ = cielab_illuminant_c(illuminant)

    cdef Py_ssize_t w, h, dim

    try:
        w, h, dim = rgb_array.shape[:3]
    except Exception as e:
        raise ValueError(
            "adobe98_array argument must be shape (h, w, 3) type uint8.\n %s " % e)

    if dim != 3 and dim != 4:
        raise TypeError(
            "adobe98_array invalid dimensions "
            "for RGB or RGBA array pixels; got (%s, %s, %s).\n" % (w, h, dim))

    cdef:
        int i, j
        unsigned char [:, :, :] rgb_array_new = empty((w, h, 3), dtype=numpy.uint8)
        float l_, a_, b_
        float r, g, b
        float x, y, z
        float refX, refY, refZ
        float ll = 0
        float tmp_

    # Convert RGB array into CIELAB equivalent
    lab_array = \
        rgb_2_cielab_c(rgb_array, illuminant_model=illuminant_model_, format_8b=format_8b)

    cdef:
        float avg_l, avg_a, avg_b, pixels_number

    # Returns the mean values for l, a, b (LAB mean values)
    avg_l, avg_a, avg_b, pixels_number = array3d_mean_c(lab_array)
    avg_a -= <float>128.0
    avg_b -= <float>128.0

    cdef float [:, :, :] cielab_array = lab_array

    with nogil:
        for i in prange(w, schedule=SCHEDULE, num_threads=THREADS):
            for j in range(h):

                l_ = cielab_array[i, j, 0]
                a_ = cielab_array[i, j, 1]
                b_ = cielab_array[i, j, 2]

                ll = l_  * _1_255

                a_ = a_ - avg_a * ll * <float> c1
                b_ = b_ - avg_b * ll * <float> c1

                if not format_8b:
                    # l_ = l_ / _255_100
                    l_ = l_ * _100_255
                    a_ = a_ - <float>128.0
                    b_ = b_ - <float>128.0

                refX = illuminant_model_[0] * <float> 100.0
                refY = illuminant_model_[1] * <float> 100.0
                refZ = illuminant_model_[2] * <float> 100.0

                tmp_ = (l_ + <float> 16.0) * _1_116
                x = refX * inv_f_t(tmp_ + a_ * _1_500)
                y = refY * inv_f_t(tmp_)
                z = refZ * inv_f_t(tmp_ - b_ * _1_200)

                # d65
                # if illuminant == 'D65':
                r = x * +<float> 3.2404542 + y * -<float> 1.5371385 + z * -<float> 0.4985314
                g = x * -<float> 0.9692660 + y * +<float> 1.8760108 + z * +<float> 0.0415560
                b = x * +<float> 0.0556434 + y * -<float> 0.2040259 + z * +<float> 1.0572252

                # if illuminant == 'D50':
                #     # # XYZ to sRGB [M]-1
                #     r = x * +<float> 3.1338561 + y * -<float> 1.6168667 + z * -<float> 0.4906146
                #     g = x * -<float> 0.9787684 + y * +<float> 1.9161415 + z * +<float> 0.0334540
                #     b = x * +<float> 0.0719453 + y * -<float> 0.2289914 + z * +<float> 1.4052427

                r = r * <float> 0.01
                g = g * <float> 0.01
                b = b * <float> 0.01

                # These linear RGB values are not the final result;
                # gamma correction must still be applied. The following formula transforms
                # the linear values into sRGB:
                if r <= <float>0.0031308:
                    r = <float> 12.92 * r
                else:
                    r = <float> 1.055 * (r ** _1_24) - <float> 0.055

                if g <= <float>0.0031308:
                    g = <float> 12.92 * g
                else:
                    g = <float> 1.055 * (g ** _1_24) - <float> 0.055

                if b <= <float>0.0031308:
                    b = <float> 12.92 * b
                else:
                    b = <float> 1.055 * (b ** _1_24) - <float> 0.055

                r *= <float> 255.0
                g *= <float> 255.0
                b *= <float> 255.0

                # CAP the RGB values 0 .. 255
                if r < 0:
                    r = <float> 0.0
                if r > 255:
                    r = <float> 255.0

                if g < 0:
                    g = <float> 0.0
                if g > 255:
                    g = <float> 255.0

                if b < 0:
                    b = <float> 0.0
                if b > 255:
                    b = <float> 255.0

                rgb_array_new[i, j, 0] = <unsigned char> round_f(r)
                rgb_array_new[i, j, 1] = <unsigned char> round_f(g)
                rgb_array_new[i, j, 2] = <unsigned char> round_f(b)

    return numpy.array(rgb_array_new)



@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cpdef void WhiteBalanceInplace(
        unsigned char[:, :, :] rgb_array,
        float c1=1.0,
        str illuminant='D65',
        bint format_8b = False
):

    """
    The Gray World algorithm for illuminant estimation assumes that
    the average color of the world is gray, or achromatic. Therefore,
    it calculates the scene illuminant as the average RGB value in the image.
    
    e.g
    >>>arr = numpy.empty((640, 480, 3), dtype=numpy.uint8)
    >>>WhiteBalance(arr)
    >>>WhiteBalance(arr, illuminant='d50')
    
    :param rgb_array: numpy.ndarray containing RGB pixels uint8 values in range 0..255
    :param c1 :
    :param illuminant: Illuminant white point; sting can be 'a','c','e','d50', 'd55', 'd65', 'icc'
    :param format_8b : True | False; Default False; Set this variable to True when using 8-bit images/surfaces
        otherwise set it to False for 24 - 32-bit images (float32)
    :return : void
    """

    cdef float [:] illuminant_model

    illuminant = illuminant.upper()
    illuminant_model = cielab_illuminant_c(illuminant)

    cdef Py_ssize_t w, h, dim

    try:
        w, h, dim = rgb_array.shape[:3]
    except Exception as e:
        raise ValueError("adobe98_array argument must be shape (h, w, 3) type uint8.\n %s " % e)

    if dim != 3 and dim != 4:
        raise TypeError(
            "adobe98_array invalid dimensions "
            "for RGB or RGBA array pixels; got (%s, %s, %s).\n" % (w, h, dim))

    cdef:
        int i, j
        float l_, a_, b_
        float r, g, b
        float x, y, z
        float refX, refY, refZ
        float ll = 0
        float tmp_

    # Convert RGB array into CIELAB equivalent
    lab_array = \
        rgb_2_cielab_c(rgb_array, illuminant_model=illuminant_model)

    cdef:
        float avg_l, avg_a, avg_b, pixels_number

    # Returns the mean values for l, a, b (LAB mean values)
    avg_l, avg_a, avg_b, pixels_number = array3d_mean_c(lab_array)
    avg_a -= <float>128.0
    avg_b -= <float>128.0

    cdef float [:, :, :] cielab_array = lab_array

    with nogil:
        for i in prange(w, schedule=SCHEDULE, num_threads=THREADS):
            for j in range(h):

                l_ = cielab_array[i, j, 0]
                a_ = cielab_array[i, j, 1]
                b_ = cielab_array[i, j, 2]


                ll = l_  * _1_255

                a_ = a_ - avg_a * ll * <float> c1
                b_ = b_ - avg_b * ll * <float> c1

                if format_8b:
                    l_ = l_ * _100_255
                    a_ = a_ - <float>128.0
                    b_ = b_ - <float>128.0

                refX = illuminant_model[0] * <float> 100.0
                refY = illuminant_model[1] * <float> 100.0
                refZ = illuminant_model[2] * <float> 100.0

                tmp_ = (l_ + <float> 16.0) * _1_116
                x = refX * inv_f_t(tmp_ + a_ * _1_500)
                y = refY * inv_f_t(tmp_)
                z = refZ * inv_f_t(tmp_ - b_ * _1_200)

                # d65
                r = x * +<float> 3.2404542 + y * -<float> 1.5371385 + z * -<float> 0.4985314
                g = x * -<float> 0.9692660 + y * +<float> 1.8760108 + z * +<float> 0.0415560
                b = x * +<float> 0.0556434 + y * -<float> 0.2040259 + z * +<float> 1.0572252

                r = r * <float> 0.01
                g = g * <float> 0.01
                b = b * <float> 0.01

                # These linear RGB values are not the final result;
                # gamma correction must still be applied. The following formula transforms
                # the linear values into sRGB:
                if r <= <float>0.0031308:
                    r = <float> 12.92 * r
                else:
                    r = <float> 1.055 * (r ** _1_24) - <float> 0.055

                if g <= <float>0.0031308:
                    g = <float> 12.92 * g
                else:
                    g = <float> 1.055 * (g ** _1_24) - <float> 0.055

                if b <= <float>0.0031308:
                    b = <float> 12.92 * b
                else:
                    b = <float> 1.055 * (b ** _1_24) - <float> 0.055

                r *= <float> 255.0
                g *= <float> 255.0
                b *= <float> 255.0

                # CAP the RGB values 0 .. 255
                if r < 0:
                    r = <float> 0.0
                if r > 255:
                    r = <float> 255.0

                if g < 0:
                    g = <float> 0.0
                if g > 255:
                    g = <float> 255.0

                if b < 0:
                    b = <float> 0.0
                if b > 255:
                    b = <float> 255.0

                rgb_array[i, j, 0] = <unsigned char> round_f(r)
                rgb_array[i, j, 1] = <unsigned char> round_f(g)
                rgb_array[i, j, 2] = <unsigned char> round_f(b)





@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cpdef white_balance_grayworld(rgb_array):

    """
    The Gray World algorithm for illuminant estimation assumes that
    the average color of the world is gray, or achromatic. Therefore,
    it calculates the scene illuminant as the average RGB value in the image.
    
    :param rgb_array: numpy.ndarray containing RGB pixels uint8 values in range 0..255
    :return : void
    """


    """
    cdef Py_ssize_t w, h, dim

    try:
        w, h, dim = rgb_array.shape[:3]
    except Exception as e:
        raise ValueError("adobe98_array argument must be shape (h, w, 3) type uint8.\n %s " % e)

    if dim != 3 and dim != 4:
        raise TypeError(
            "adobe98_array invalid dimensions "
            "for RGB or RGBA array pixels; got (%s, %s, %s).\n" % (w, h, dim))

    cdef:
        int i, j
        float r, g, b, alpha, beta
        float avg_r, avg_g, avg_b, pixels_number

    avg_r, avg_g, avg_b, pixels_number = array3d_mean_c(rgb_array)

    cdef unsigned char [:, :, ::1 ] rgb_array = rgb_array

    avg_r *= _1_255
    avg_g *= _1_255
    avg_b *= _1_255
    alpha = avg_g / avg_r
    beta = avg_g / avg_b

    with nogil:
        for i in prange(w):
            for j in range(h):
                r, g, b = rgb_array[i, j, 0], rgb_array[i, j, 1], rgb_array[i, j, 2]
                rgb_array[i, j, 0] = <unsigned char>(min(r * alpha, 255))
                rgb_array[i, j, 1] = <unsigned char>g
                rgb_array[i, j, 2] = <unsigned char>(min(b * beta, 255))

    return numpy.array(rgb_array)
    """

    # Not fully tested
    raise NotImplemented




@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cpdef white_balance_SDWGW(rgb_array):

    """
    The Standard Deviation Weighted Gray World (SDWGW) is an extension of gray-world. It subdivides the
    image into n blocks and for each one of them calculates standard deviations and means of the R, G, and
    B channels. SDWGD defines standard deviation-weighted averages of each colour channels

    :param rgb_array: numpy.ndarray containing RGB pixels uint8 values in range 0..255
   
    """

    """
    cdef Py_ssize_t w, h, dim

    try:
        w, h, dim = rgb_array.shape[:3]
    except Exception as e:
        raise ValueError("adobe98_array argument must be shape (h, w, 3) type uint8.\n %s " % e)

    if dim != 3 and dim != 4:
        raise TypeError(
            "adobe98_array invalid dimensions "
            "for RGB or RGBA array pixels; got (%s, %s, %s).\n" % (w, h, dim))

    cdef:
        int i, j
        float l_, a_, b_
        float r, g, b
        float x, y, z
        float refX, refY, refZ
        float ll = 0
        float tmp_

    cdef:
        float avg_r, avg_g, avg_b, pixels_number

    avg_r, std_r, avg_g, std_g, avg_b, std_b = array3d_stats(rgb_array)

    cdef unsigned char [:, :, ::1 ] rgb_array = rgb_array

    avg_r *= _1_255
    avg_g *= _1_255
    avg_b *= _1_255
    alpha = avg_g / avg_r
    beta = avg_g / avg_b

    with nogil:
        for i in range(w):
            for j in range(h):
                r, g, b = rgb_array[i, j, 0], rgb_array[i, j, 1], rgb_array[i, j, 2]
                rgb_array[i, j, 0] = <unsigned char>(min(r * alpha, 255))
                rgb_array[i, j, 1] = <unsigned char>g
                rgb_array[i, j, 2] = <unsigned char>(min(b * beta, 255))

    return numpy.array(rgb_array)
    """
    # Not fully tested
    raise NotImplemented

@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cpdef adobe2rgb(float [:, :, :] adobe98_array, str ref='D65'):
    """
    Convert an ADOBE 98 array (float) into an SRGB array(uint8) 
    Works with D50 & D65 illuminant only
    
    e.g: 
    
    >>> arrf = numpy.empty((640, 480, 3), dtype=numpy.float32)
    >>> adobe_rgb = adobe2rgb(arrf)

    :param adobe98_array : numpy.ndarray shape (w, h, 3|4) containing RGB pixel values (float) 
    :param ref: str; Illuminant white point; sting can be 'd50', 'd65' 
    :return : New array containing RGB equivalent values after conversion. 
     Array shape (w, h, 3) of type uint8
    """
    return adobe2rgb_c(adobe98_array, ref)


@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cdef adobe2rgb_c(float[:, :, :] adobe98_array, str ref='D65'):

    """
    Convert an ADOBE 98 array (float) into an SRGB array(uint8) 
    Works with D50 & D65 illuminant only

    >>> arrf = numpy.empty((640, 480, 3), dtype=numpy.float32)
    >>> adobe_rgb = adobe2rgb_c(arrf)
    
    :param adobe98_array : numpy.ndarray shape (w, h, 3|4) containing RGB pixel values (float) 
    :param ref: str; Illuminant white point; sting can be 'd50', 'd65' 
    :return : New array containing RGB equivalent values after conversion. 
     Array shape (w, h, 3) of type uint8
    """
    cdef Py_ssize_t w, h, dim

    try:
       w, h, dim = adobe98_array.shape[:3]
    except Exception as e:
       raise ValueError(
           f"\nadobe98_array argument must be shape (w, h, 3|4) type float.\n {e}")


    ref = ref.upper()

    if ref != 'D50' and ref != 'D65':
        raise ValueError('\nAttribute ref must be D50 or D65 got %s' % ref)

    if dim != 3 and dim != 4:
       raise TypeError(
           'adobe98_array invalid dimensions '
           'for RGB or RGBA array pixels; got ({}, {}, {}).\n'.format(w, h, dim))
    cdef:
       int i, j
       unsigned char [:, :, ::1 ] rgb_array = empty((w, h, 3), dtype=numpy.uint8)
       float r, g, b, x, y, z

    with nogil:
        for i in prange(w, schedule=SCHEDULE, num_threads=THREADS):
            for j in range(h):

                r = pow(adobe98_array[i, j, 0] * _1_255, <float> 2.199)
                g = pow(adobe98_array[i, j, 1] * _1_255, <float> 2.199)
                b = pow(adobe98_array[i, j, 2] * _1_255, <float> 2.199)

                # Adobe 1998 Calibration D65
                if ref == 'D65':
                    # ADOBE RGB to XYZ [M]
                    x = r * <float> 0.5767309 + g * <float> 0.1855540 + b * <float> 0.1881852
                    y = r * <float> 0.2973769 + g * <float> 0.6273491 + b * <float> 0.0752741
                    z = r * <float> 0.0270343 + g * <float> 0.0706872 + b * <float> 0.9911085

                    # XYZ to sRGB [M]-1
                    r = x * +<float> 3.2404542 + y * -<float> 1.5371385 + z * -<float> 0.4985314
                    g = x * -<float> 0.9692660 + y * +<float> 1.8760108 + z * +<float> 0.0415560
                    b = x * +<float> 0.0556434 + y * -<float> 0.2040259 + z * +<float> 1.0572252

                if ref == 'D50':
                    # ADOBE RGB to XYZ [M]
                    x = r * <float> 0.6097559 + g * <float> 0.2052401 + b * <float> 0.1492240
                    y = r * <float> 0.3111242 + g * <float> 0.6256560 + b * <float> 0.0632197
                    z = r * <float> 0.0194811 + g * <float> 0.0608902 + b * <float> 0.7448387

                    # # XYZ to sRGB [M]-1
                    r = x * +<float> 3.1338561 + y * -<float> 1.6168667 + z * -<float> 0.4906146
                    g = x * -<float> 0.9787684 + y * +<float> 1.9161415 + z * +<float> 0.0334540
                    b = x * +<float> 0.0719453 + y * -<float> 0.2289914 + z * +<float> 1.4052427

                # These linear RGB values are not the final result;
                # gamma correction must still be applied. The following formula transforms
                # the linear values into sRGB:
                if r <= <float> 0.0031308:
                    r = <float> 12.92 * r
                else:
                    r = <float> 1.055 * (r ** _1_24) - <float> 0.055

                if g <= <float> 0.0031308:
                    g = <float> 12.92 * g
                else:
                    g = <float> 1.055 * (g ** _1_24) - <float> 0.055

                if b <= <float> 0.0031308:
                    b = <float> 12.92 * b
                else:
                    b = <float> 1.055 * (b ** _1_24) - <float> 0.055

                r *= <float> 255.0
                g *= <float> 255.0
                b *= <float> 255.0

                # CAP the RGB values 0 .. 255
                if r < 0:
                    r = <float> 0.0
                if r > 255:
                    r = <float> 255.0

                if g < 0:
                    g = <float> 0.0
                if g > 255:
                    g = <float> 255.0

                if b < 0:
                    b = <float> 0.0
                if b > 255:
                    b = <float> 255.0

                rgb_array[i, j, 0] = <unsigned char> r
                rgb_array[i, j, 1] = <unsigned char> g
                rgb_array[i, j, 2] = <unsigned char> b

    return numpy.asarray(rgb_array)

@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cpdef rgb2adobe(unsigned char[:, :, :] rgb_array, str ref='D65'):
    """
    Convert an RGB array into an ADOBE 98 equivalent array
    Works with D50 & D65 illuminant only
    
    e.g
    >>> arr_u = numpy.empty((640, 480, 3), dtype=numpy.uint8)
    >>> adobe_arr = rgb2adobe(arr_u)
    
    :param rgb_array : numpy.ndarray shape (w, h, 3|4) containing RGB pixel values (uint8) 
    :param ref:str; Illuminant white point; sting can be 'd50', 'd65'
        see below for more details about the illuminant argument.
    :return : New array containing ADOBE 98 values, array shape (w, h, 3) of type float

    """
    return rgb2adobe_c(rgb_array, ref)

@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cdef rgb2adobe_c(unsigned char[:, :, :] rgb_array, str ref='D65'):

    """
    Convert an RGB array into an ADOBE 98 equivalent array
    Works with D50 & D65 illuminant only
    
     e.g
    >>> arr_u = numpy.empty((640, 480, 3), dtype=numpy.uint8)
    >>> adobe_arr = rgb2adobe_c(arr_u)
    
    :param rgb_array : numpy.ndarray shape (w, h, 3|4) containing RGB pixel values (uint8) 
    :param ref:str; Illuminant white point; sting can be 'd50', 'd65'
        see below for more details about the illuminant argument.
    :return : New array containing ADOBE 98 values, array shape (w, h, 3) of type float 
    """

    cdef Py_ssize_t w, h, dim

    try:
        w, h, dim = rgb_array.shape[:3]
    except Exception as e:
        raise ValueError(
            "\nrgb_array argument must be shape (w, h, 3|4) type uint8.\n %s " % e)

    ref = ref.upper()

    if dim != 3 and dim != 4:
        raise TypeError(
            "rgb_array invalid dimensions "
            "for RGB or RGBA array pixels; got (%s, %s, %s).\n" % (w, h, dim))

    cdef:
        int i, j
        float [:, :, ::1 ] tmp = empty((w, h, 3), dtype=numpy.float32)
        float r, g, b, x, y, z, k0, k1, k2, xa, ya, za

    k0 = yw - yk
    k1 = xw - xk
    k2 = zw - zk


    with nogil:
        for i in prange(w, schedule=SCHEDULE, num_threads=THREADS):
            for j in range(h):

                # ------------- RGB TO XYZ --------------------
                # RGB values in nominal range [0, 1].
                r = <float>(rgb_array[i, j, 0] * _1_255)
                g = <float>(rgb_array[i, j, 1] * _1_255)
                b = <float>(rgb_array[i, j, 2] * _1_255)

                if r > 0.04045:
                    r = ((r + <float> 0.055) / <float> 1.055) ** <float> 2.4
                else:
                    r = r / <float> 12.92

                if g > 0.04045:
                    g = ((g + <float> 0.055) / <float> 1.055) ** <float> 2.4
                else:
                    g = g / <float> 12.92

                if b > 0.04045:
                    b = ((b + <float> 0.055) / <float> 1.055) ** <float> 2.4
                else:
                    b = b / <float> 12.92

                if ref == 'D65':
                    # d65
                    x = r * <float> 0.4124564 + g * <float> 0.3575761 + b * <float> 0.1804375
                    y = r * <float> 0.2126729 + g * <float> 0.7151522 + b * <float> 0.0721750
                    z = r * <float> 0.0193339 + g * <float> 0.1191920 + b * <float> 0.9503041

                if ref == 'D50':
                    # d50
                    x = r * <float> 0.4360747 + g * <float> 0.3850649 + b * <float> 0.1430804
                    y = r * <float> 0.2225045 + g * <float> 0.7168786 + b * <float> 0.0606169
                    z = r * <float> 0.0139322 + g * <float> 0.0971045 + b * <float> 0.7141733

                # ------------- RGB TO XYZ END --------------------

                # ------------- XYZ TO ADOBE98 --------------------

                xa = x * k1 * (yw / xw) + xk
                ya = y * k0 + yk
                za = z * k2 * (yw / zw) + zk

                x = (xa - xk) / k1 * (xw / yw)
                y = (ya - yk) / k0
                z = (za - zk) / k2 * (zw / yw)

                if ref == 'D65':
                    # Adobe 1998 Calibration D65
                    r = x * +<float> 2.0413690 + y * -<float> 0.5649464 + z * -<float> 0.3446944
                    g = x * -<float> 0.9692660 + y * +<float> 1.8760108 + z * +<float> 0.0415560
                    b = x * +<float> 0.0134474 + y * -<float> 0.1183897 + z * +<float> 1.0154096

                if ref == 'D50':
                    # D50
                    r = x * +<float> 1.9624274 + y * -<float> 0.6105343 + z * -<float> 0.3413404
                    g = x * -<float> 0.9787684 + y * +<float> 1.9161415 + z * +<float> 0.0334540
                    b = x * +<float> 0.0286869 + y * -<float> 0.1406752 + z * +<float> 1.3487655

                if r < 0.0:
                    r = <float> 0.0
                else:
                    r = <float> 255.0 * pow(r, ADOBE_GAMMA)

                if g < 0.0:
                    g = <float> 0.0
                else:
                    g = <float> 255.0 * pow(g, ADOBE_GAMMA)

                if b < 0.0:
                    b = <float> 0.0
                else:
                    b = <float> 255.0 * pow(b, ADOBE_GAMMA)

                # CAP the RGB values 0 .. 255
                if r > 255:
                    r = <float> 255.0

                if g > 255:
                    g = <float> 255.0

                if b > 255:
                    b = <float> 255.0

                tmp[i, j, 0] = <float>r
                tmp[i, j, 1] = <float>g
                tmp[i, j, 2] = <float>b

    return numpy.asarray(tmp)



@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cpdef void rgb2adobe_inplace(unsigned char[:, :, :] rgb_array, str ref='D65'):
    """
    Convert an RGB array into an ADOBE 98 equivalent array (INPLACE)
    Works with D50 & D65 illuminant only
    
     e.g
    >>> arr_u = numpy.empty((640, 480, 3), dtype=numpy.uint8)
    >>> rgb2adobe_inplace(arr_u)
    
    :param rgb_array : numpy.ndarray shape (w, h, 3|4) containing RGB pixel values (uint8) 
    :param ref:str; Illuminant white point; sting can be 'd50', 'd65'
        see below for more details about the illuminant argument.
    :return : void

    """
    rgb2adobe_inplace_c(rgb_array, ref)

@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cdef void rgb2adobe_inplace_c(unsigned char[:, :, :] rgb_array, str ref='D65'):

    """
    Convert an RGB array into an ADOBE 98 equivalent array (INPLACE)
    Works with D50 & D65 illuminant only
    
    >>> arr_u = numpy.empty((640, 480, 3), dtype=numpy.uint8)
    >>> rgb2adobe_inplace_c(arr_u)
    
    :param rgb_array : numpy.ndarray shape (w, h, 3|4) containing RGB pixel values (uint8) 
    :param ref:str; Illuminant white point; sting can be 'd50', 'd65'
        see below for more details about the illuminant argument.
    :return : void
    """

    cdef Py_ssize_t w, h, dim

    try:
        w, h, dim = rgb_array.shape[:3]
    except Exception as e:
        raise ValueError("\nadobe98_array argument must be shape (h, w, 3) type uint8.\n %s " % e)

    ref = ref.upper()

    if dim != 3 and dim != 4:
        raise TypeError(
            "adobe98_array invalid dimensions "
            "for RGB or RGBA array pixels; got (%s, %s, %s).\n" % (w, h, dim))

    cdef:
        int i, j
        float r, g, b, x, y, z, k0, k1, k2, xa, ya, za

    k0 = yw - yk
    k1 = xw - xk
    k2 = zw - zk

    with nogil:
        for j in prange(h, schedule=SCHEDULE, num_threads=THREADS):
            for i in range(w):

                # RGB values in nominal range [0, 1].
                r = <float>rgb_array[i, j, 0] * _1_255
                g = <float>rgb_array[i, j, 1] * _1_255
                b = <float>rgb_array[i, j, 2] * _1_255

                r = pow(r, <float>2.2)
                g = pow(g, <float>2.2)
                b = pow(b, <float>2.2)

                if r > 0.04045:
                    r = ((r + <float> 0.055) / <float> 1.055) ** <float> 2.4
                else:
                    r = r / <float> 12.92

                if g > 0.04045:
                    g = ((g + <float> 0.055) / <float> 1.055) ** <float> 2.4
                else:
                    g = g / <float> 12.92

                if b > 0.04045:
                    b = ((b + <float> 0.055) / <float> 1.055) ** <float> 2.4
                else:
                    b = b / <float> 12.92

                if ref == 'D65':
                    # d65
                    x = r * <float> 0.4124564 + g * <float> 0.3575761 + b * <float> 0.1804375
                    y = r * <float> 0.2126729 + g * <float> 0.7151522 + b * <float> 0.0721750
                    z = r * <float> 0.0193339 + g * <float> 0.1191920 + b * <float> 0.9503041

                if ref == 'D50':
                    # d50
                    x = r * <float> 0.4360747 + g * <float> 0.3850649 + b * <float> 0.1430804
                    y = r * <float> 0.2225045 + g * <float> 0.7168786 + b * <float> 0.0606169
                    z = r * <float> 0.0139322 + g * <float> 0.0971045 + b * <float> 0.7141733

                xa = x * k1 * (yw / xw) + xk
                ya = y * k0 + yk
                za = z * k2 * (yw / zw) + zk

                x = (xa - xk) / k1 * (xw / yw)
                y = (ya - yk) / k0
                z = (za - zk) / k2 * (zw / yw)

                if ref == 'D65':
                    # Adobe 1998 Calibration D65
                    r = x * +<float> 2.0413690 + y * -<float> 0.5649464 + z * -<float> 0.3446944
                    g = x * -<float> 0.9692660 + y * +<float> 1.8760108 + z * +<float> 0.0415560
                    b = x * +<float> 0.0134474 + y * -<float> 0.1183897 + z * +<float> 1.0154096

                if ref == 'D50':
                    # D50
                    r = x * +<float> 1.9624274 + y * -<float> 0.6105343 + z * -<float> 0.3413404
                    g = x * -<float> 0.9787684 + y * +<float> 1.9161415 + z * +<float> 0.0334540
                    b = x * +<float> 0.0286869 + y * -<float> 0.1406752 + z * +<float> 1.3487655

                if r < 0.0:
                    r = <float> 0.0
                else:
                    r = <float> 255.0 * pow(r, ADOBE_GAMMA)

                if g < 0.0:
                    g = <float> 0.0
                else:
                    g = <float> 255.0 * pow(g, ADOBE_GAMMA)

                if b < 0.0:
                    b = <float> 0.0
                else:
                    b = <float> 255.0 * pow(b, ADOBE_GAMMA)

                # CAP the RGB values 0 .. 255
                if r > 255:
                    r = <float> 255.0

                if g > 255:
                    g = <float> 255.0

                if b > 255:
                    b = <float> 255.0

                rgb_array[i, j, 0] = <unsigned char>r
                rgb_array[i, j, 1] = <unsigned char>g
                rgb_array[i, j, 2] = <unsigned char>b


@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cpdef void adobe2rgb_inplace(unsigned char [:, :, :] adobe98_array, str ref='D65'):
    """
    Convert an ADOBE 98 array (float) into an SRGB array(uint8) 
    Works with D50 & D65 illuminant only
    
    >>> arr_u = numpy.empty((640, 480, 3), dtype=numpy.uint8)
    >>> adobe2rgb_inplace(arr_u)

    :param adobe98_array : numpy.ndarray shape (w, h, 3|4) containing RGB pixel values (float) 
    :param ref: str; Illuminant white point; sting can be 'd50', 'd65' 
    :return : New array containing RGB equivalent values after conversion. 
     Array shape (w, h, 3) of type uint8
    :return : void

    """
    adobe2rgb_inplace_c(adobe98_array, ref)



@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cdef void adobe2rgb_inplace_c(unsigned char [:, :, :] adobe98_array, str ref='D65'):

    """

    Convert an ADOBE 98 array (float) into an SRGB array(uint8) 
    Works with D50 & D65 illuminant only

    >>> arr_u = numpy.empty((640, 480, 3), dtype=numpy.uint8)
    >>> adobe2rgb_inplace_c(arr_u)
    
    :param adobe98_array : numpy.ndarray shape (w, h, 3|4) containing RGB pixel values (float) 
    :param ref: str; Illuminant white point; sting can be 'd50', 'd65' 
    :return : New array containing RGB equivalent values after conversion. 
     Array shape (w, h, 3) of type uint8
    :return : void 
    """
    cdef Py_ssize_t w, h, dim

    try:
       w, h, dim = adobe98_array.shape[:3]
    except Exception as e:
       raise ValueError(
           f"\nadobe98_array argument must be shape (h, w, 3) type float.\n {e}")


    ref = ref.upper()

    if ref != 'D50' and ref != 'D65':
        raise ValueError('\nAttribute ref must be D50 or D65 got %s' % ref)

    if dim != 3 and dim != 4:
       raise TypeError(
           'adobe98_array invalid dimensions '
           'for RGB or RGBA array pixels; got ({}, {}, {}).\n'.format(w, h, dim))
    cdef:
       int i, j
       float r, g, b, x, y, z

    with nogil:
        for j in prange(h, schedule=SCHEDULE, num_threads=THREADS):
            for i in range(w):

                r = pow(adobe98_array[i, j, 0] * _1_255, <float> 2.199)
                g = pow(adobe98_array[i, j, 1] * _1_255, <float> 2.199)
                b = pow(adobe98_array[i, j, 2] * _1_255, <float> 2.199)

                # Adobe 1998 Calibration D65
                if ref == 'D65':
                    x = r * <float> 0.5767309 + g * <float> 0.1855540 + b * <float> 0.1881852
                    y = r * <float> 0.2973769 + g * <float> 0.6273491 + b * <float> 0.0752741
                    z = r * <float> 0.0270343 + g * <float> 0.0706872 + b * <float> 0.9911085

                    # Calibration D65
                    r = x * +<float> 3.2404542 + y * -<float> 1.5371385 + z * -<float> 0.4985314
                    g = x * -<float> 0.9692660 + y * +<float> 1.8760108 + z * +<float> 0.0415560
                    b = x * +<float> 0.0556434 + y * -<float> 0.2040259 + z * +<float> 1.0572252


                if ref == 'D50':
                    x = r * <float> 0.6097559 + g * <float> 0.2052401 + b * <float> 0.1492240
                    y = r * <float> 0.3111242 + g * <float> 0.6256560 + b * <float> 0.0632197
                    z = r * <float> 0.0194811 + g * <float> 0.0608902 + b * <float> 0.7448387

                    # d50
                    r = x * +<float> 3.1338561 + y * -<float> 1.6168667 + z * -<float> 0.4906146
                    g = x * -<float> 0.9787684 + y * +<float> 1.9161415 + z * +<float> 0.0334540
                    b = x * +<float> 0.0719453 + y * -<float> 0.2289914 + z * +<float> 1.4052427


                # These linear RGB values are not the final result;
                # gamma correction must still be applied. The following formula transforms
                # the linear values into sRGB:
                if r <= <float> 0.0031308:
                    r = <float> 12.92 * r
                else:
                    r = <float> 1.055 * (r ** _1_24) - <float> 0.055

                if g <= <float> 0.0031308:
                    g = <float> 12.92 * g
                else:
                    g = <float> 1.055 * (g ** _1_24) - <float> 0.055

                if b <= <float> 0.0031308:
                    b = <float> 12.92 * b
                else:
                    b = <float> 1.055 * (b ** _1_24) - <float> 0.055

                r *= <float> 255.0
                g *= <float> 255.0
                b *= <float> 255.0

                # CAP the RGB values 0 .. 255
                if r < 0:
                    r = <float> 0.0
                if r > 255:
                    r = <float> 255.0

                if g < 0:
                    g = <float> 0.0
                if g > 255:
                    g = <float> 255.0

                if b < 0:
                    b = <float> 0.0
                if b > 255:
                    b = <float> 255.0

                adobe98_array[i, j, 0] = <unsigned char> r
                adobe98_array[i, j, 1] = <unsigned char> g
                adobe98_array[i, j, 2] = <unsigned char> b

@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cpdef rgb2xyz(unsigned char[:, :, :] rgb_array, str ref='D65'):
    """
    Convert an RGB array into an XYZ equivalent array/image
    Works with D50 & D65 illuminant only
    
    e.g:
    >>>arr_u = numpy.empty((640, 480, 3), dtype=numpy.uint8)
    >>>xyz_array = rgb2xyz(arr_u)

    :param rgb_array : numpy.ndarray shape (w, h, 3|4) containing RGB pixel values (uint8) 
    :param ref:str; Illuminant white point; sting can be 'd50', 'd65'
        see below for more details about the illuminant argument.
    :return : New array containing ZYZ values, array shape (w, h, 3) of type float 
    """
    return rgb2xyz_c(rgb_array, ref)

@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cdef rgb2xyz_c(unsigned char[:, :, :] rgb_array, str ref='D65'):

    """
    Convert an RGB array into an XYZ equivalent array/image
    Works with D50 & D65 illuminant only
    
    >>>arr_u = numpy.empty((640, 480, 3), dtype=numpy.uint8)
    >>>xyz_array = rgb2xyz_c(arr_u)
    
    :param rgb_array : numpy.ndarray shape (w, h, 3|4) containing RGB pixel values (uint8) 
    :param ref:str; Illuminant white point; sting can be 'd50', 'd65'
        see below for more details about the illuminant argument.
    :return : New array containing ZYZ values, array shape (w, h, 3) of type float 
    """

    cdef Py_ssize_t w, h, dim

    try:
        w, h, dim = rgb_array.shape[:3]
    except Exception as e:
        raise ValueError(
            "\nrgb_array argument must be shape (w, h, 3|4) type uint8.\n %s " % e)

    ref = ref.upper()

    if dim != 3 and dim != 4:
        raise TypeError(
            "rgb_array invalid dimensions "
            "for RGB or RGBA array pixels; got (%s, %s, %s).\n" % (w, h, dim))

    cdef:
        int i, j
        float [:, :, ::1 ] tmp = empty((w, h, 3), dtype=numpy.float32)
        float r, g, b, x, y, z, k0, k1, k2, xa, ya, za

    k0 = yw - yk
    k1 = xw - xk
    k2 = zw - zk


    with nogil:
        for i in prange(w, schedule=SCHEDULE, num_threads=THREADS):
            for j in range(h):

                # ------------- RGB TO XYZ --------------------
                # RGB values in nominal range [0, 1].
                r = <float>(rgb_array[i, j, 0] * _1_255)
                g = <float>(rgb_array[i, j, 1] * _1_255)
                b = <float>(rgb_array[i, j, 2] * _1_255)

                if r > 0.04045:
                    r = ((r + <float> 0.055) / <float> 1.055) ** <float> 2.4
                else:
                    r = r / <float> 12.92

                if g > 0.04045:
                    g = ((g + <float> 0.055) / <float> 1.055) ** <float> 2.4
                else:
                    g = g / <float> 12.92

                if b > 0.04045:
                    b = ((b + <float> 0.055) / <float> 1.055) ** <float> 2.4
                else:
                    b = b / <float> 12.92

                if ref == 'D65':
                    # d65
                    x = r * <float> 0.4124564 + g * <float> 0.3575761 + b * <float> 0.1804375
                    y = r * <float> 0.2126729 + g * <float> 0.7151522 + b * <float> 0.0721750
                    z = r * <float> 0.0193339 + g * <float> 0.1191920 + b * <float> 0.9503041

                if ref == 'D50':
                    # d50
                    x = r * <float> 0.4360747 + g * <float> 0.3850649 + b * <float> 0.1430804
                    y = r * <float> 0.2225045 + g * <float> 0.7168786 + b * <float> 0.0606169
                    z = r * <float> 0.0139322 + g * <float> 0.0971045 + b * <float> 0.7141733

                # ------------- RGB TO XYZ END --------------------

                tmp[i, j, 0] = <float>x * 255.0
                tmp[i, j, 1] = <float>y * 255.0
                tmp[i, j, 2] = <float>z * 255.0

    return numpy.asarray(tmp)
