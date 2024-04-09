# Cielab conversion tools 

CIELAB is a free library coded with python and cython and offers 
fast conversion methods between various colour spaces such as
 sRGB, CIELAB, XYZ, ADOBE 98 and contains all the necessary methods
 to convert a color space domain from one to another and reciprocally.

This library is built with methods that can be call for a given pixel (RGB), 
LAB, XYZ tristimulus values, or used directly to an entire 3d arrays with 
similar color space.

Most of the image format such as PNG, JPEG, JPG, BMP etc will work with
this library as long as you can provide a valid 3d array (refers to the 
methods arguments and documentations to pass a valid array shape and type).
Check also the examples below that shows how to extract a 3d array from
the most common known image processing library such as (OPENCV, PIL, scikit) or 
Pygame that provides very fast and efficient methods for image editing.

You can directly call Cielab conversion methods from your favourite Python editor
 or choose to call the `cythonized` versions instead (only if you are familiar 
with `Cython`). This will provide the best performances for your project(s).
Nevertheless, Cielab library offers both methods, Cython `Cpdef` hooks for 
Python and `cdef` version for cython external library. 

Most of the methods can be used with D50 and D65 illuminant used in the vast majority 
of industries and applications when using the Cielab pixel conversion methods.

<!--However, you can choose any of the following illuminant models 'a', 'c', 'e', 'd50', 
'd55', 'd65', 'icc' when converting a 3d array (except for ADOBE 98) -->

<p align="left">
    <img src="https://github.com/yoyoberenguer/Cielab/blob/main/Cielab/Assets/demo.png?raw=true">
</p>

---

## Installation from pip
Check the link for newest version https://pypi.org/project/Cielab/

From the command line 
```cmd
C:\>pip install Cielab
```
---

## Some useful color space definitions 
### What is Cielab (definition from Wikipedia)

The CIELAB color space, also referred to as L*a*b*, is a color space defined by the International
Commission on Illumination (abbreviated CIE) in 1976. It expresses color as three values: 
L* for perceptual lightness and a* and b* for the four unique colors of human vision: red, green, blue and yellow. 

The CIELAB space is three-dimensional and covers the entire gamut (range) of human color perception.
It is based on the opponent color model of human vision, where red and green form an opponent pair and
blue and yellow form an opponent pair. The lightness value, L*, also referred to as "Lstar",
defines black at 0 and white at 100. The a* axis is relative to the green–red opponent colors, 
with negative values toward green and positive values toward red. The b* axis represents the blue–yellow
opponents, with negative numbers toward blue and positive toward yellow.

The a* and b* axes are unbounded and depending on the reference white they can easily exceed ±150 to cover
the human gamut. Nevertheless, software implementations often clamp these values for practical reasons.
For instance, if integer math is being used it is common to clamp a* and b* in the range of −128 to 127.

CIELAB is calculated relative to a reference white, for which the CIE recommends the use of CIE Standard
illuminant D65. D65 is used in the vast majority of industries and applications, with the notable 
exception being the printing industry which uses D50. The International Color Consortium largely supports
the printing industry and uses D50 with either CIEXYZ or CIELAB in the Profile Connection Space, 
for v2 and v4 ICC profiles.

### Meaning of X, Y and Z

The CIE 1931 RGB color space and CIE 1931 XYZ color space were created by the International 
Commission on Illumination (CIE) in 1931.They resulted from a series of experiments done
in the late 1920s by William David Wright using ten observers and John Guild using seven 
observers. The experimental results were combined into the specification of the CIE RGB color
space, from which the CIE XYZ color space was derived.

The CIE 1931 color spaces are still widely used, as is the 1976 CIELUV color space.
Tristimulus values

The normalized spectral sensitivity of human cone cells of short-, middle- and long-wavelength types.
The human eye with normal vision has three kinds of cone cells that sense light, having peaks of spectral
sensitivity in short ("S", 420 nm – 440 nm), middle ("M", 530 nm – 540 nm), and long ("L", 560 nm – 580 nm)
wavelengths. These cone cells underlie human color perception in conditions of medium and high brightness;
in very dim light color vision diminishes, and the low-brightness, monochromatic "night vision" receptors,
denominated "rod cells", become effective. Thus, three parameters corresponding to levels of stimulus of 
the three kinds of cone cells, in principle describe any human color sensation. Weighting a total light
power spectrum by the individual spectral sensitivities of the three kinds of cone cells renders three
effective values of stimulus; these three values compose a tristimulus specification of the objective 
color of the light spectrum. The three parameters, denoted "S", "M", and "L", are indicated using a
3-dimensional space denominated the "LMS color space", which is one of many color spaces devised to quantify
human color vision.

A color space maps a range of physically produced colors from mixed light, pigments, etc. to an objective
description of color sensations registered in the human eye, typically in terms of tristimulus values, 
but not usually in the LMS color space defined by the spectral sensitivities of the cone cells. 
The tristimulus values associated with a color space can be conceptualized as amounts of three primary colors
in a tri-chromatic, additive color model. In some color spaces, including the LMS and XYZ spaces, 
the primary colors used are not real colors in the sense that they cannot be generated in any light spectrum.

The CIE XYZ color space encompasses all color sensations that are visible to a person with average eyesight. 
That is why CIE XYZ (Tristimulus values) is a device-invariant representation of color. It serves as a
standard reference against which many other color spaces are defined. A set of color-matching functions,
like the spectral sensitivity curves of the LMS color space, but not restricted to non-negative sensitivities,
associates physically produced light spectra with specific tristimulus values.

Consider two light sources composed of different mixtures of various wavelengths. Such light sources may 
appear to be the same color; this effect is called "metamerism." Such light sources have the same apparent
color to an observer when they produce the same tristimulus values, regardless of the spectral power 
distributions of the sources.

Most wavelengths stimulate two or all three kinds of cone cell because the spectral sensitivity curves of 
the three kinds overlap. Certain tristimulus values are thus physically impossible: e.g. LMS tristimulus
values that are non-zero for the M component and zero for both the L and S components. Furthermore pure
spectral colors would, in any normal trichromatic additive color space, e.g., the RGB color spaces, 
imply negative values for at least one of the three primaries because the chromaticity would be outside 
the color triangle defined by the primary colors. To avoid these negative RGB values, and to have one
component that describes the perceived brightness, "imaginary" primary colors and corresponding color-matching
functions were formulated. The CIE 1931 color space defines the resulting tristimulus values, in which they
are denoted by "X", "Y", and "Z".In XYZ space, all combinations of non-negative coordinates are meaningful,
but many, such as the primary locations [1, 0, 0], [0, 1, 0], and [0, 0, 1], correspond to imaginary colors 
outside the space of possible LMS coordinates; imaginary colors do not correspond to any spectral distribution
of wavelengths and therefore have no physical reality.

### Adobe RGB color space
The Adobe RGB (1998) color space or opRGB is a color space developed by Adobe Inc. in 1998. 
It was designed to encompass most of the colors achievable on CMYK color printers, but by using
RGB primary colors on a device such as a computer display. The Adobe RGB (1998) color space 
encompasses roughly 30% of the visible colors specified by the CIELAB color space – improving
upon the gamut of the sRGB color space, primarily in cyan-green hues. It was subsequently 
standardized by the IEC as IEC 61966-2-5:1999 with a name opRGB (optional RGB color space)
and is used in HDMI.

### SRGB
sRGB is a standard RGB (red, green, blue) color space that HP and Microsoft created cooperatively 
in 1996 to use on monitors, printers, and the World Wide Web. It was subsequently standardized
by the International Electrotechnical Commission (IEC) as IEC 61966-2-1:1999. sRGB is the
current defined standard colorspace for the web, and it is usually the assumed colorspace for 
images that are neither tagged for a colorspace nor have an embedded color profile.

sRGB essentially codifies the display specifications for the computer monitors in use at that time,
which greatly aided its acceptance. sRGB uses the same color primaries and white point as ITU-R BT.709
standard for HDTV, a transfer function (or gamma) compatible with the era's CRT displays, 
and a viewing environment designed to match typical home and office viewing conditions.

---
*In python Idle*
```
from Cielab import *
```

---
## Getting started:

### Examples showing how to convert images

Pygame (convert RGB image to CIELAB)
```python

WIDTH = 1280
HEIGHT = 1024
SCREEN = pygame.display.set_mode((WIDTH, HEIGHT))

# load an image
image = pygame.image.load("../Assets/background2.png")
rgb_array_ = pygame.surfarray.pixels3d(image)

# Transform an RGB array into CIELAB equivalent array using 
# d65 illuminant 
cielab_array = rgb_2_cielab(rgb_array_, illuminant ='d65', format_8b=False)
# Create a surface from the cielab array 
image_cielab = pygame.surfarray.make_surface(cielab_array)


```

PIL (convert RGB image to CIELAB)
```python
from PIL import Image
# load image
im = Image.open("../Assets/background2.png")
im_bytes = im.tobytes()

# create 3d array (682, 1024, 3) of type int8
rgb_array_ = numpy.frombuffer(im_bytes, dtype=numpy.uint8).reshape((682, 1024, 3))
# transpose width and height
rgb_array_ = rgb_array_.transpose(1, 0, 2).copy()
# Convert rgb array into cielab model 
cielab_array = rgb_2_cielab(rgb_array_, illuminant ='d65', format_8b=False)

cielab_array = cielab_array.transpose(1, 0, 2)
numpy.ndarray.flatten(cielab_array)

image_str = (cielab_array.astype(numpy.uint8)).tobytes()
image = Image.frombytes('RGB', (1024, 682), image_str)
image.show()
```
PIL (convert RGB image to ADOBE 98)
```python
from PIL import Image
# load image
im = Image.open("../Assets/background2.png")
im_bytes = im.tobytes()

# create 3d array (682, 1024, 3) of type int8
rgb_array_ = numpy.frombuffer(im_bytes, dtype=numpy.uint8).reshape((682, 1024, 3))
# transpose width and height
rgb_array_ = rgb_array_.transpose(1, 0, 2).copy()
# Convert rgb array into adobe98 array using d65 illuminant 
adobe_array = rgb2adobe(rgb_array_, ref ='d65')

adobe_array = adobe_array.transpose(1, 0, 2)
numpy.ndarray.flatten(adobe_array)

image_str = (adobe_array.astype(numpy.uint8)).tobytes()
image = Image.frombytes('RGB', (1024, 682), image_str)
image.show()

```

OpenCv (convert RGB image to CIELAB)
```python
import cv2
img = cv2.imread("../Assets/background2.png")
cielab_array = rgb_2_cielab(img, illuminant ='d65', format_8b=False)
cielab_array = cv2.cvtColor(cielab_array, cv2.COLOR_BGR2RGB)
cv2.imshow('image', cielab_array.astype(numpy.uint8))
cv2.waitKey(0)
cv2.destroyAllWindows()
```

scikit-image (convert RGB image to CIELAB)
```python
import skimage as ski
import os
import matplotlib
from matplotlib import pyplot as plt
matplotlib.use('TkAgg')
filename = os.path.join(ski.data_dir, '../Assets/background2.png')
rgb_array = ski.io.imread(filename)
rgb_array = rgb_array.transpose(1, 0, 2)
cielab_array = rgb_2_cielab(rgb_array, illuminant ='d50', format_8b=False)
image = cielab_array.transpose(1, 0, 2).astype(numpy.uint8)
plt.imshow(image)
plt.show()
```



### Example showing how to convert color information

```python

# Color definition 
# RGB 255.0 0 0 (RED)
# XYZ 41.246 | 21.267 | 1.933
# Adobe 218.946 0 0.048
x, y, z = 41.246, 21.267, 1.933

# XYZ to ADOBE 98
r, g, b = xyz_adobe98(x, y, z, ref='D65').values()
# RGB to XYZ
x, y, z = rgb_to_xyz(255.0, 0, 0).values()  # default d65
# XYZ to CIELAB 
D65 = numpy.array([0.9504, 1.0000, 1.0888], dtype=float32)
l, a, b = xyz_to_cielab(
   41.24563980102539, 
   21.267290115356445, 
   1.9333901405334473, model=D65).values()


```
### Array mean and standard deviation 
```python
import skimage as ski
import os
import matplotlib
from matplotlib import pyplot as plt
matplotlib.use('TkAgg')
filename = os.path.join(ski.data_dir, '../Assets/background2.png')
rgb_array = ski.io.imread(filename)
# Get RGB mean and standard deviation for each channels
red_mean, red_dev, \
green_mean, green_dev,\
blue_mean, blue_dev = array3d_stats(rgb_array).values()

```


---
## PIXEL TRANSFORMATION

#### Convert XYZ tristimulus to ADOBE98 and back (D65 or D50 illuminant)
```text
xyz_adobe98(x, y, z, ref='D65')

adobe98_xyz(r, g, b, ref='D65')
```

#### Convert RGB to XYZ tristimulus and back (D65 or D50 illuminant)
```text
rgb_to_xyz(r, g, b, ref='D65')

xyz_to_rgb( x, y, z, ref='D65')
```

#### Convert XYZ to CIELAB and back; models ('a', 'c', 'e', 'd50', 'd55', 'd65', 'icc')
```text
xyz_to_cielab(x, y, z, model=model_d65, format_8b = False)

cielab_to_xyz(l , a, b, model=model_d65, format_8b = False)
```

#### Convert RGB to CIELAB and back; models ('a', 'c', 'e', 'd50', 'd55', 'd65', 'icc')
```
rgb_to_cielab(r, g, b, model=model_d65, format_8b = False)
        
cielab_to_rgb(l, a, b, model=model_d65, format_8b = False)

```
---
### ARRAY TRANSFORMATION

#### Convert ARRAY RGB to ARRAY CIELAB; models ('a', 'c', 'e', 'd50', 'd55', 'd65', 'icc')

```text
rgb_2_cielab(rgb_array_, illuminant_ ='d65', format_8b=False)
        
cielab_2_rgb(lab_array_, illuminant_ ='d65', format_8b=False)
```


#### White balance ; models ('a', 'c', 'e', 'd50', 'd55', 'd65', 'icc')
```text

WhiteBalance(rgb_array_, c1=1.0, illuminant_='D65', format_8b = False)

WhiteBalanceInplace(rgb_array_,  c1=1.0, illuminant_='D65', format_8b = False)

```


#### ADOBE98 ARRAY to RGB ARRAY and back ; models ('d50', 'd65')
```text
adobe2rgb(adobe98_array_, ref='D65')

rgb2adobe(rgb_array_, ref='D65')

rgb2adobe_inplace(rgb_array_, ref='D65')

adobe2rgb_inplace(adobe98_array_, ref='D65')

```

#### RGB array to XYZ array ; models ('d50', 'd65')
```text
rgb2xyz(rgb_array_, ref='D65')
    

```


## Credit
Yoann Berenguer 

## Dependencies :
```
numpy >= 1.19.5
pygame ==2.5.2
cython =>3.0.2
setuptools~=54.1.1
```

## License :

GNU GENERAL PUBLIC LICENSE Version 3

Copyright (c) 2019 Yoann Berenguer

Copyright (C) 2007 Free Software Foundation, Inc. <https://fsf.org/>
Everyone is permitted to copy and distribute verbatim copies
of this license document, but changing it is not allowed.


## Testing: 
```python
>>> import Cielab
>>> from Cielab.tests.test_cielab import run_testsuite
>>> run_testsuite()
```

## Performances:
From the command line 
```cmd
C:\>cd tests
C:\>python profiler.py
```
