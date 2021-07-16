# -*- coding: utf-8 -*-

import numpy as np
from PIL import Image

# /////////////// Corruption Helpers ///////////////

import skimage as sk
from torchvision import transforms
import torchvision.transforms.functional as F
from skimage.filters import gaussian
from io import BytesIO
from wand.image import Image as WandImage
from wand.api import library as wandlibrary
import wand.color as WandColor
import ctypes
from PIL import Image as PILImage
import cv2
from scipy.ndimage import zoom as scizoom
from scipy.ndimage.interpolation import map_coordinates
import warnings
import os
from pkg_resources import resource_filename

warnings.simplefilter("ignore", UserWarning)


def disk(radius, alias_blur=0.1, dtype=np.float32):
    # 17 x 17 kernel causes seg fault in opencv
    # if radius <= 8:
    #     L = np.arange(-8, 8 + 1)
    #     ksize = (3, 3) 
    if radius <= 5:
        L = np.arange(-5, 5 + 1)
        ksize = (3, 3)
    else:
        L = np.arange(-radius, radius + 1)
        ksize = (5, 5)
    X, Y = np.meshgrid(L, L)
    aliased_disk = np.array((X ** 2 + Y ** 2) <= radius ** 2, dtype=dtype)
    aliased_disk /= np.sum(aliased_disk)
    # supersample disk to antialias
    return cv2.GaussianBlur(aliased_disk, ksize=ksize, sigmaX=alias_blur)


# Tell Python about the C method
wandlibrary.MagickMotionBlurImage.argtypes = (ctypes.c_void_p,  # wand
                                              ctypes.c_double,  # radius
                                              ctypes.c_double,  # sigma
                                              ctypes.c_double)  # angle


# Extend wand.image.Image class to include method signature
class MotionImage(WandImage):
    def motion_blur(self, radius=0.0, sigma=0.0, angle=0.0):
        wandlibrary.MagickMotionBlurImage(self.wand, radius, sigma, angle)


# modification of https://github.com/FLHerne/mapgen/blob/master/diamondsquare.py
def plasma_fractal(mapsize=512, wibbledecay=3):
    """
    Generate a heightmap using diamond-square algorithm.
    Return square 2d array, side length 'mapsize', of floats in range 0-255.
    'mapsize' must be a power of two.
    """
    assert (mapsize & (mapsize - 1) == 0)
    maparray = np.empty((mapsize, mapsize), dtype=np.float_)
    maparray[0, 0] = 0
    stepsize = mapsize
    wibble = 100

    def wibbledmean(array):
        return array / 4 + wibble * np.random.uniform(-wibble, wibble, array.shape)

    def fillsquares():
        """For each square of points stepsize apart,
           calculate middle value as mean of points + wibble"""
        cornerref = maparray[0:mapsize:stepsize, 0:mapsize:stepsize]
        squareaccum = cornerref + np.roll(cornerref, shift=-1, axis=0)
        squareaccum += np.roll(squareaccum, shift=-1, axis=1)
        maparray[stepsize // 2:mapsize:stepsize,
        stepsize // 2:mapsize:stepsize] = wibbledmean(squareaccum)

    def filldiamonds():
        """For each diamond of points stepsize apart,
           calculate middle value as mean of points + wibble"""
        mapsize = maparray.shape[0]
        drgrid = maparray[stepsize // 2:mapsize:stepsize, stepsize // 2:mapsize:stepsize]
        ulgrid = maparray[0:mapsize:stepsize, 0:mapsize:stepsize]
        ldrsum = drgrid + np.roll(drgrid, 1, axis=0)
        lulsum = ulgrid + np.roll(ulgrid, -1, axis=1)
        ltsum = ldrsum + lulsum
        maparray[0:mapsize:stepsize, stepsize // 2:mapsize:stepsize] = wibbledmean(ltsum)
        tdrsum = drgrid + np.roll(drgrid, 1, axis=1)
        tulsum = ulgrid + np.roll(ulgrid, -1, axis=0)
        ttsum = tdrsum + tulsum
        maparray[stepsize // 2:mapsize:stepsize, 0:mapsize:stepsize] = wibbledmean(ttsum)

    while stepsize >= 2:
        fillsquares()
        filldiamonds()
        stepsize //= 2
        wibble /= wibbledecay

    maparray -= maparray.min()
    return maparray / maparray.max()


def clipped_zoom(img, zoom_factor):
    h = img.shape[0]
    # ceil crop height(= crop width)
    ch = int(np.ceil(h / float(zoom_factor)))

    top = (h - ch) // 2
    img = scizoom(img[top:top + ch, top:top + ch], (zoom_factor, zoom_factor, 1), order=1)
    # trim off any extra pixels
    trim_top = (img.shape[0] - h) // 2

    return img[trim_top:trim_top + h, trim_top:trim_top + h]


# /////////////// End Corruption Helpers ///////////////


# /////////////// Corruptions ///////////////

def gaussian_noise(x, severity=1):
    # c = [.08, .12, 0.18, 0.26, 0.38][severity - 1]
    c = [0.04, 0.06, .08, .09, .10][severity - 1]

    x = np.array(x) / 255.
    return np.clip(x + np.random.normal(size=x.shape, scale=c), 0, 1) * 255


def shot_noise(x, severity=1):
    # c = [60, 25, 12, 5, 3][severity - 1]
    c = [500, 250, 100, 75, 50][severity - 1]

    x = np.array(x) / 255.
    return np.clip(np.random.poisson(x * c) / float(c), 0, 1) * 255


def impulse_noise(x, severity=1):
    # c = [.03, .06, .09, 0.17, 0.27][severity - 1]
    c = [.01, .02, .03, .05, .07][severity - 1]

    x = sk.util.random_noise(np.array(x) / 255., mode='s&p', amount=c)
    return np.clip(x, 0, 1) * 255


def speckle_noise(x, severity=1):
    # c = [.15, .2, 0.35, 0.45, 0.6][severity - 1]
    c = [.06, .1, .12, .16, .2][severity - 1]

    x = np.array(x) / 255.
    return np.clip(x + x * np.random.normal(size=x.shape, scale=c), 0, 1) * 255


def gaussian_blur(x, severity=1):
    # c = [1, 2, 3, 4, 6][severity - 1]
    c = [.4, .6, 0.7, .8, 1][severity - 1]

    x = gaussian(np.array(x) / 255., sigma=c, multichannel=True)
    return np.clip(x, 0, 1) * 255


def glass_blur(x, severity=1):
    # sigma, max_delta, iterations
    # c = [(0.7, 1, 2), (0.9, 2, 1), (1, 2, 3), (1.1, 3, 2), (1.5, 4, 2)][severity - 1]
    c = [(0.05,1,1), (0.25,1,1), (0.4,1,1), (0.25,1,2), (0.4,1,2)][severity - 1]

    x = np.uint8(gaussian(np.array(x) / 255., sigma=c[0], multichannel=True) * 255)
    size = x.shape[0]
    # locally shuffle pixels
    for i in range(c[2]):
        for h in range(size - c[1], c[1], -1):
            for w in range(size - c[1], c[1], -1):
                dx, dy = np.random.randint(-c[1], c[1], size=(2,))
                h_prime, w_prime = h + dy, w + dx
                # swap
                x[h, w], x[h_prime, w_prime] = x[h_prime, w_prime], x[h, w]

    return np.clip(gaussian(x / 255., sigma=c[0], multichannel=True), 0, 1) * 255


def defocus_blur(x, severity=1):
    # c = [(3, 0.1), (4, 0.5), (6, 0.5), (8, 0.5), (10, 0.5)][severity - 1]
    c = [(0.3, 0.4), (0.4, 0.5), (0.5, 0.6), (1, 0.2), (1.5, 0.1)][severity - 1]

    x = np.array(x) / 255.
    kernel = disk(radius=c[0], alias_blur=c[1])

    channels = []
    for d in range(3):
        channels.append(cv2.filter2D(x[:, :, d].astype(np.float32), -1, kernel))
    channels = np.array(channels).transpose((1, 2, 0))  # 3x32x32 -> 32x32x3

    return np.clip(channels, 0, 1) * 255

def motion_blur(x, severity=1):
    # c = [(10, 3), (15, 5), (15, 8), (15, 12), (20, 15)][severity - 1]
    c = [(6,1), (6,1.5), (6,2), (8,2), (9,2.5)][severity - 1]

    output = BytesIO()
    Image.fromarray(x).save(output, format='PNG')
    x = MotionImage(blob=output.getvalue())

    x.motion_blur(radius=c[0], sigma=c[1], angle=np.random.uniform(-45, 45))

    x = cv2.imdecode(np.fromstring(x.make_blob(), np.uint8),
                     cv2.IMREAD_UNCHANGED)

    if x.shape != (512, 512):
        return np.clip(x[..., [2, 1, 0]], 0, 255)  # BGR to RGB
    else:  # greyscale to RGB
        return np.clip(np.array([x, x, x]).transpose((1, 2, 0)), 0, 255)


def zoom_blur(x, severity=1):
    # c = [np.arange(1, 1.11, 0.01),
    #      np.arange(1, 1.16, 0.01),
    #      np.arange(1, 1.21, 0.02),
    #      np.arange(1, 1.26, 0.02),
    #      np.arange(1, 1.31, 0.03)][severity - 1]
    c = [np.arange(1, 1.06, 0.01), np.arange(1, 1.11, 0.01), np.arange(1, 1.16, 0.01),
         np.arange(1, 1.21, 0.01), np.arange(1, 1.26, 0.01)][severity - 1]

    x = (np.array(x) / 255.).astype(np.float32)
    out = np.zeros_like(x)
    for zoom_factor in c:
        out += clipped_zoom(x, zoom_factor)

    x = (x + out) / (len(c) + 1)
    return np.clip(x, 0, 1) * 255


def fog(x, severity=1):
    # c = [(1.5, 2), (2., 2), (2.5, 1.7), (2.5, 1.5), (3., 1.4)][severity - 1]
    c = [(.2,3), (.5,3), (0.75,2.5), (1,2), (1.5,1.75)][severity - 1]
    size = x.shape[0]
    x = np.array(x) / 255.
    max_val = x.max()
    x += c[0] * plasma_fractal(wibbledecay=c[1])[:size, :size][..., np.newaxis]
    return np.clip(x * max_val / (max_val + c[0]), 0, 1) # *255


def frost(x, severity=1):
    size = x.shape[0]
    # c = [(1, 0.4),
    #      (0.8, 0.6),
    #      (0.7, 0.7),
    #      (0.65, 0.7),
    #      (0.6, 0.75)][severity - 1]
    c = [(1, 0.2), (1, 0.3), (0.9, 0.4), (0.85, 0.4), (0.75, 0.45)][severity - 1]
    idx = np.random.randint(5)
    filename = [resource_filename(__name__, 'frost/frost1.png'),
                resource_filename(__name__, 'frost/frost2.png'),
                resource_filename(__name__, 'frost/frost3.png'),
                resource_filename(__name__, 'frost/frost4.jpg'),
                resource_filename(__name__, 'frost/frost5.jpg'),
                resource_filename(__name__, 'frost/frost6.jpg')][idx]
    frost = cv2.imread(filename)
    # randomly crop and convert to rgb

    x_start, y_start = np.random.randint(0, frost.shape[0] - size), np.random.randint(0, frost.shape[1] - size)

    frost = frost[x_start:x_start + size, y_start:y_start + size][..., [2, 1, 0]]

    return np.clip(c[0] * np.array(x) + c[1] * frost, 0, 255) / 255


def snow(x, severity=1):
    # c = [(0.1, 0.3, 3, 0.5, 10, 4, 0.8),
    #      (0.2, 0.3, 2, 0.5, 12, 4, 0.7),
    #      (0.55, 0.3, 4, 0.9, 12, 8, 0.7),
    #      (0.55, 0.3, 4.5, 0.85, 12, 8, 0.65),
    #      (0.55, 0.3, 2.5, 0.85, 12, 12, 0.55)][severity - 1]

    c = [(0.1,0.2,1,0.6,8,3,0.95),
         (0.1,0.2,1,0.5,10,4,0.9),
         (0.15,0.3,1.75,0.55,10,4,0.9),
         (0.25,0.3,2.25,0.6,12,6,0.85),
         (0.3,0.3,1.25,0.65,14,12,0.8)][severity - 1]

    size = x.shape[0]
    x = np.array(x, dtype=np.float32) / 255.
    snow_layer = np.random.normal(size=x.shape[:2], loc=c[0], scale=c[1])  # [:2] for monochrome

    snow_layer = clipped_zoom(snow_layer[..., np.newaxis], c[2])
    snow_layer[snow_layer < c[3]] = 0

    snow_layer = PILImage.fromarray((np.clip(snow_layer.squeeze(), 0, 1) * 255).astype(np.uint8), mode='L')
    output = BytesIO()
    snow_layer.save(output, format='PNG')
    snow_layer = MotionImage(blob=output.getvalue())

    snow_layer.motion_blur(radius=c[4], sigma=c[5], angle=np.random.uniform(-135, -45))

    snow_layer = cv2.imdecode(np.fromstring(snow_layer.make_blob(), np.uint8),
                              cv2.IMREAD_UNCHANGED) / 255.
    snow_layer = snow_layer[..., np.newaxis]

    x = c[6] * x + (1 - c[6]) * np.maximum(x, cv2.cvtColor(x, code = cv2.COLOR_RGB2GRAY).reshape(size, size, 1) * 1.5 + 0.5)
    return np.clip(x + snow_layer + np.rot90(snow_layer, k=2), 0, 1) * 255


def spatter(x, severity=1):
    # c = [(0.65, 0.3, 4, 0.69, 0.6, 0),
    #      (0.65, 0.3, 3, 0.68, 0.6, 0),
    #      (0.65, 0.3, 2, 0.68, 0.5, 0),
    #      (0.65, 0.3, 1, 0.65, 1.5, 1),
    #      (0.67, 0.4, 1, 0.65, 1.5, 1)][severity - 1]
    c = [(0.62,0.1,0.7,0.7,0.5,0),
         (0.65,0.1,0.8,0.7,0.5,0),
         (0.65,0.3,1,0.69,0.5,0),
         (0.65,0.1,0.7,0.69,0.6,1),
         (0.65,0.1,0.5,0.68,0.6,1)][severity - 1]

    x = np.array(x, dtype=np.float32) / 255.

    liquid_layer = np.random.normal(size=x.shape[:2], loc=c[0], scale=c[1])

    liquid_layer = gaussian(liquid_layer, sigma=c[2])
    liquid_layer[liquid_layer < c[3]] = 0
    if c[5] == 0:
        liquid_layer = (liquid_layer * 255).astype(np.uint8)
        dist = 255 - cv2.Canny(liquid_layer, 50, 150)
        dist = cv2.distanceTransform(dist, cv2.DIST_L2, 5)
        _, dist = cv2.threshold(dist, 20, 20, cv2.THRESH_TRUNC)
        dist = cv2.blur(dist, (3, 3)).astype(np.uint8)
        dist = cv2.equalizeHist(dist)
        ker = np.array([[-2, -1, 0], [-1, 1, 1], [0, 1, 2]])
        dist = cv2.filter2D(dist, cv2.CV_8U, ker)
        dist = cv2.blur(dist, (3, 3)).astype(np.float32)

        m = cv2.cvtColor(liquid_layer * dist, cv2.COLOR_GRAY2BGRA)
        m /= np.max(m, axis=(0, 1))
        m *= c[4]

        # water is pale turqouise
        color = np.concatenate((175 / 255. * np.ones_like(m[..., :1]),
                                238 / 255. * np.ones_like(m[..., :1]),
                                238 / 255. * np.ones_like(m[..., :1])), axis=2)

        color = cv2.cvtColor(color, cv2.COLOR_BGR2BGRA)
        x = cv2.cvtColor(x, cv2.COLOR_BGR2BGRA)

        return cv2.cvtColor(np.clip(x + m * color, 0, 1), cv2.COLOR_BGRA2BGR) * 255
    else:
        m = np.where(liquid_layer > c[3], 1, 0)
        m = gaussian(m.astype(np.float32), sigma=c[4])
        m[m < 0.8] = 0

        # mud brown
        color = np.concatenate((63 / 255. * np.ones_like(x[..., :1]),
                                42 / 255. * np.ones_like(x[..., :1]),
                                20 / 255. * np.ones_like(x[..., :1])), axis=2)

        color *= m[..., np.newaxis]
        x *= (1 - m[..., np.newaxis])

        return np.clip(x + color, 0, 1) * 255


def contrast(x, severity=1):
    # c = [0.4, .3, .2, .1, .05][severity - 1]
    c = [.75, .5, .4, .3, 0.15][severity - 1]

    x = np.array(x) / 255.
    means = np.mean(x, axis=(0, 1), keepdims=True)
    return np.clip((x - means) * c + means, 0, 1) * 255


def generate_random_lines(imshape,slant,drop_length,rain_type):
    drops=[]
    area=imshape[0]*imshape[1]
    no_of_drops=area//600

    if rain_type.lower()=='drizzle':
        no_of_drops=area//770
        drop_length=10
    elif rain_type.lower()=='heavy':
        drop_length=30
    elif rain_type.lower()=='torrential':
        no_of_drops=area//500
        drop_length=60

    for i in range(no_of_drops): ## If You want heavy rain, try increasing this
        if slant<0:
            x= np.random.randint(slant,imshape[1])
        else:
            x= np.random.randint(0,imshape[1]-slant)
        y= np.random.randint(0,imshape[0]-drop_length)
        drops.append((x,y))
    return drops,drop_length


def rain_process(image,slant,drop_length,drop_color,drop_width,rain_drops):
    imshape = image.shape  
    image_t = image.copy()
    for rain_drop in rain_drops:
        cv2.line(image_t,(rain_drop[0],rain_drop[1]),(rain_drop[0]+slant,rain_drop[1]+drop_length),drop_color,drop_width)
    image= cv2.blur(image_t,(7,7)) ## rainy view are blurry
    brightness_coefficient = 0.7 ## rainy days are usually shady 
    image_HLS = hls(image) ## Conversion to HLS
    image_HLS[:,:,1] = image_HLS[:,:,1]*brightness_coefficient ## scale pixel values down for channel 1(Lightness)
    image_RGB= rgb(image_HLS,'hls') ## Conversion to RGB
    return image_RGB


def hls(image,src='RGB'):
    image_HLS = eval('cv2.cvtColor(image,cv2.COLOR_'+src.upper()+'2HLS)')
    return image_HLS


def rgb(image, src='BGR'):
    image_RGB= eval('cv2.cvtColor(image,cv2.COLOR_'+src.upper()+'2RGB)')
    return image_RGB


def rain(image, slant=-1,drop_length=20,drop_width=1,drop_color=(200,200,200),rain_type='torrential'): ## (200,200,200) a shade of gray
    # verify_image(image)
    slant_extreme=slant
    # if not(is_numeric(slant_extreme) and (slant_extreme>=-20 and slant_extreme<=20)or slant_extreme==-1):
    #     raise Exception(err_rain_slant)
    # if not(is_numeric(drop_width) and drop_width>=1 and drop_width<=5):
    #     raise Exception(err_rain_width)
    # if not(is_numeric(drop_length) and drop_length>=0 and drop_length<=100):
    #     raise Exception(err_rain_length)

    imshape = image.shape
    if slant_extreme==-1:
        slant= np.random.randint(-10,10) ##generate random slant if no slant value is given
    rain_drops, drop_length= generate_random_lines(imshape,slant,drop_length,rain_type)
    output = rain_process(image,slant_extreme,drop_length,drop_color,drop_width,rain_drops)
    return output


def brightness(x, severity=1):
    # c = [.1, .2, .3, .4, .5][severity - 1]
    c = [.05, .1, .15, .2, .3][severity - 1]

    x = np.array(x) / 255.
    x = sk.color.rgb2hsv(x)
    x[:, :, 2] = np.clip(x[:, :, 2] + c, 0, 1)
    x = sk.color.hsv2rgb(x)

    return np.clip(x, 0, 1) * 255


def saturate(x, severity=1):
    # c = [(0.3, 0), (0.1, 0), (2, 0), (5, 0.1), (20, 0.2)][severity - 1]
    c = [(0.3, 0), (0.1, 0), (1.5, 0), (2, 0.1), (2.5, 0.2)][severity - 1]
    x = np.array(x) / 255.
    x = sk.color.rgb2hsv(x)
    x[:, :, 1] = np.clip(x[:, :, 1] * c[0] + c[1], 0, 1)
    x = sk.color.hsv2rgb(x)

    return np.clip(x, 0, 1) * 255


def jpeg_compression(x, severity=1):
    # c = [25, 18, 15, 10, 7][severity - 1]
    c = [80, 65, 58, 50, 40][severity - 1]

    output = BytesIO()

    Image.fromarray(x).save(output, 'JPEG', quality=c)
    x = np.array(PILImage.open(output))

    return x


def pixelate(x, severity=1):
    size = x.shape[0]
    # c = [0.6, 0.5, 0.4, 0.3, 0.25][severity - 1]
    c = [0.95, 0.9, 0.85, 0.75, 0.65][severity - 1]

    x = Image.fromarray(x)
    x = x.resize((int(size * c), int(size * c)),resample=Image.BILINEAR)
    x = x.resize((size, size),Image.NEAREST)

    return np.array(x)


# mod of https://gist.github.com/erniejunior/601cdf56d2b424757de5
def elastic_transform(image, severity=1):
    c = [(244 * 2, 244 * 0.7, 244 * 0.1),  # 244 should have been 512, but ultimately nothing is incorrect
         (244 * 2, 244 * 0.08, 244 * 0.2),
         (244 * 0.05, 244 * 0.01, 244 * 0.02),
         (244 * 0.07, 244 * 0.01, 244 * 0.02),
         (244 * 0.12, 244 * 0.01, 244 * 0.02)][severity - 1]

    image = np.array(image, dtype=np.float32) / 255.
    shape = image.shape
    shape_size = shape[:2]

    # random affine
    center_square = np.float32(shape_size) // 2
    square_size = min(shape_size) // 3
    pts1 = np.float32([center_square + square_size,
                       [center_square[0] + square_size, center_square[1] - square_size],
                       center_square - square_size])
    pts2 = pts1 + np.random.uniform(-c[2], c[2], size=pts1.shape).astype(np.float32)
    M = cv2.getAffineTransform(pts1, pts2)
    image = cv2.warpAffine(image, M, shape_size[::-1], borderMode=cv2.BORDER_REFLECT_101)

    dx = (gaussian(np.random.uniform(-1, 1, size=shape[:2]),
                   c[1], mode='reflect', truncate=3) * c[0]).astype(np.float32)
    dy = (gaussian(np.random.uniform(-1, 1, size=shape[:2]),
                   c[1], mode='reflect', truncate=3) * c[0]).astype(np.float32)
    dx, dy = dx[..., np.newaxis], dy[..., np.newaxis]

    x, y, z = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]), np.arange(shape[2]))
    indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1)), np.reshape(z, (-1, 1))
    return np.clip(map_coordinates(image, indices, order=1, mode='reflect').reshape(shape), 0, 1) * 255


def blackoutNoise(image, severity=1):
    image = np.zeros(image.shape, dtype=np.uint8)
    m = (severity, severity, severity)
    s = (severity, severity, severity)

    image = np.clip(cv2.randn(image, m, s), 0, 255)

    return image


def additiveGaussianNoise(image, severity=1):
    m = (severity, severity, severity)
    s = (severity, severity, severity)
    corr = cv2.randn(np.zeros(image.shape, dtype=np.uint8), m, s)

    image = np.clip(image.copy() + corr, 0, 255)
    return image


def occlusion(image, severity=1):
    mask = np.ones(image.shape, dtype=np.uint8)

    x = int(image.shape[0] * np.random.rand())
    y = int(image.shape[1] * np.random.rand())
    r = int((min(image.shape[:2]) / 4) * np.random.rand() + (min(image.shape[:2]) / 4))

    cv2.circle(mask, (x, y), r, 0, -1)

    image = np.clip(image.copy() * mask, 0, 255)
    return image
