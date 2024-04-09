# cython: binding=False, boundscheck=False, wraparound=False, nonecheck=False, cdivision=True,
# profile=False, initializedcheck=False
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

cdef extern from 'Include/Cielab_c.c':

    struct im_stats:
        float red_mean;
        float red_std_dev;
        float green_mean;
        float green_std_dev;
        float blue_mean;
        float blue_std_dev;

    struct lab:
        float l;
        float a;
        float b;

    struct xyz:
        float x;
        float y;
        float z;

    struct rgb:
        float r
        float g
        float b




cpdef tuple array3d_mean(object array)
cdef tuple array3d_mean_c(object array)

cpdef im_stats array3d_stats(object array)

# --------------------- TRANSFORMATION AT THE PIXEL LEVEL -------------------------------------------------------------

# CONVERT XYZ to RGB (ADOBE 1998) D65/2 & D50
cpdef rgb xyz_adobe98(float x, float y, float z, str ref=*)nogil
cdef rgb xyz_adobe98_c(float x, float y, float z, str ref=*)nogil

# CONVERT RGB (ADOBE 1998) TO XYZ D65/2 & D50
cpdef xyz adobe98_xyz(float r, float g, float b, str ref=*)nogil
cdef xyz adobe98_xyz_c(float r, float g, float b, str ref=*)nogil


# CONVERT COLOR RGB -> XYZ D65 & D50
cpdef xyz rgb_to_xyz(float r, float g, float b, str ref=*)nogil
cdef xyz rgb_to_xyz_c(float r, float g, float b, str ref=*)nogil

# CONVERT XYZ -> RGB COLOR D65 & D50
cpdef rgb xyz_to_rgb(float x, float y, float z, str ref=*)nogil
cdef rgb xyz_to_rgb_c(float x, float y, float z, str ref=*)nogil

# CONVERT XYZ to CIELAB
cpdef lab xyz_to_cielab(
        float x,
        float y,
        float z,
        const float [:] model=*,
        bint format_8b = *
)nogil
cdef lab xyz_to_cielab_c(
        float x,
        float y,
        float z,
        const float [:] model=*,
        bint format_8b = *
)nogil

# CONVERT CIELAB TO XYZ
cpdef xyz cielab_to_xyz(
        float l ,
        float a,
        float b,
        const float [:] model=*,
        bint format_8b = *
)nogil
cdef xyz cielab_to_xyz_c(
        float l ,
        float a,
        float b,
        const float [:] model=*,
        bint format_8b = *
)nogil

# CONVERT RGB TO CIELAB
cpdef lab rgb_to_cielab(
        float r,
        float g,
        float b,
        const float [:] model=*,
        bint format_8b=*
)nogil
cdef lab rgb_to_cielab_c(
        float r,
        float g,
        float b,
        const float [:] model=*,
        bint format_8b=*
)nogil


# CONVERT CIELAB TO RGB
cpdef rgb cielab_to_rgb(
        float l,
        float a,
        float b,
        const float [:] model=*,
        bint format_8b=*
)nogil
cdef rgb cielab_to_rgb_c(
        float l,
        float a,
        float b,
        const float [:] model=*,
        bint format_8b=*
)nogil

# --------------------------------------------------------------------------------------------------------------------

# -------------------------------TRANSFORMATION AT THE IMAGE / ARRAY LEVEL -------------------------------------------
# CONVERT IMAGE RGB TO IMAGE CIELAB
cpdef rgb_2_cielab(
        unsigned char[:, :, :] rgb_array,
        str illuminant =*,
        bint format_8b=*
)
cdef rgb_2_cielab_c(
        unsigned char[:, :, :] rgb_array,
        float [:] illuminant_model = *,
        bint format_8b = *
)

cpdef cielab_2_rgb(
        float [:, :, :] lab_array,
        str illuminant = *,
        bint format_8b = *
)
cdef cielab_2_rgb_c(
        float [:, :, :] lab_array,
        const float [:] illuminant_model = *,
        bint format_8b = *
)

cpdef WhiteBalance(
        unsigned char[:, :, :] rgb_array,
        float c1= *,
        str illuminant= *,
        bint format_8b = *
)

cpdef void WhiteBalanceInplace(
        unsigned char[:, :, :] rgb_array,
        float c1= *,
        str illuminant= *,
        bint format_8b = *
)

# Not fully tested in ver 1.0.11
cpdef white_balance_grayworld(rgb_array)
# Not fully tested in ver 1.0.11
cpdef white_balance_SDWGW(rgb_array)


cpdef rgb2adobe(unsigned char[:, :, :] rgb_array, str ref=*)
cdef rgb2adobe_c(unsigned char[:, :, :] rgb_array, str ref=*)

cpdef adobe2rgb(float [:, :, :] adobe98_array, str ref=*)
cdef adobe2rgb_c(float [:, :, :] adobe98_array, str ref=*)


cpdef void rgb2adobe_inplace(unsigned char[:, :, :] rgb_array, str ref=*)
cdef void rgb2adobe_inplace_c(unsigned char[:, :, :] rgb_array, str ref=*)

cpdef void adobe2rgb_inplace(unsigned char [:, :, :] adobe98_array, str ref=*)
cdef void adobe2rgb_inplace_c(unsigned char [:, :, :] adobe98_array, str ref=*)

cpdef rgb2xyz(unsigned char[:, :, :] rgb_array, str ref=*)
cdef rgb2xyz_c(unsigned char[:, :, :] rgb_array, str ref=*)
