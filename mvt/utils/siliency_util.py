#!/usr/bin/python
# -*- coding: utf-8 -*-
# saliency util

import math
import itertools
from mtl.utils.geometric_util import imresize
import numpy as np
import cv2


class ImgNormalize:
    def normalize_range(self, src, begin=0, end=255):
        # normalize with range
        dst = np.zeros((len(src), len(src[0])))
        amin, amax = np.amin(src), np.amax(src)
        for y, x in itertools.product(range(len(src)), range(len(src[0]))):
            if amin != amax:
                dst[y][x] = (src[y][x] - amin) * (end - begin) / (amax - amin) + begin
            else:
                dst[y][x] = (end + begin) / 2
        return dst

    def normalize(self, src):
        # normalize with strethcing or smoothing
        src = self.normalize_range(src, 0., 1.)  # normalize to [0,1]
        amax = np.amax(src) # max value of array
        maxs = []
        for y in range(1, len(src) - 1):
            for x in range(1, len(src[0]) - 1):
                val = src[y][x]
                if val == amax:
                    continue
                if val > src[y - 1][x] and val > src[y + 1][x] and val > src[y][x - 1] and val > src[y][x + 1]:
                    # appedn the ground
                    maxs.append(val)

        if len(maxs) != 0:
            src *= math.pow(amax - (np.sum(maxs) / np.float64(len(maxs))), 2.)
        return src


class GaussianPyramid:
    # gaussian pyramid
    def __init__(self, src):
        self.maps = self.__make_gaussian_pyramid(src)  # generate pyramids

    def __make_gaussian_pyramid(self, src):
        # gaussian pyramids, output a series of features
        maps = {'intensity': [],
                'colors': {'b': [], 'g': [], 'r': [], 'y': []},
                'orientations': {'0': [], '45': [], '90': [], '135': []}}
        amax = np.amax(src)  # max value
        # print(src)
        b, g, r = cv2.split(src)
        for x in range(1, 9):
            b, g, r = map(cv2.pyrDown, [b, g, r])  # gaussian smooth and downsample
            if x < 2:
                continue
            buf_its = np.zeros(b.shape)
            buf_colors = list(map(lambda _: np.zeros(b.shape), range(4)))  # b, g, r, y
            for y, x in itertools.product(range(len(b)), range(len(b[0]))):
                buf_its[y][x] = self.__get_intensity(b[y][x], g[y][x], r[y][x])  # brightness
                # generate for color aberration maps
                buf_colors[0][y][x], buf_colors[1][y][x], buf_colors[2][y][x], buf_colors[3][y][x] = self.__get_colors(b[y][x], g[y][x], r[y][x], buf_its[y][x], amax)
            maps['intensity'].append(buf_its)
            for (color, index) in zip(sorted(maps['colors'].keys()), range(4)):
                maps['colors'][color].append(buf_colors[index])
            for (orientation, index) in zip(sorted(maps['orientations'].keys()), range(4)):
                maps['orientations'][orientation].append(self.__conv_gabor(buf_its, np.pi * index / 4))  # orientation feature maps
        return maps

    def __get_intensity(self, b, g, r):
        # get the brightness
        return (np.float64(b) + np.float64(g) + np.float64(r)) / 3.

    def __get_colors(self, b, g, r, i, amax):
        # get color aberration
        b, g, r = list(map(lambda x: np.float64(x) if (x > 0.1 * amax) else 0., [b, g, r]))  # 将小于最大值的十分之一的数据置零
        nb, ng, nr = list(map(lambda x, y, z: max(x - (y + z) / 2., 0.), [b, g, r], [r, r, g], [g, b, b]))  # 两两相减形成3个色差图
        ny = max(((r + g) / 2. - math.fabs(r - g) / 2. - b), 0.)  # 第四个色差图
        if i != 0.0:
            return list(map(lambda x: x / np.float64(i), [nb, ng, nr, ny]))
        else:
            return nb, ng, nr, ny

    def __conv_gabor(self, src, theta):
        # Gabor filters
        kernel = cv2.getGaborKernel((8, 8), 4, theta, 8, 1)
        return cv2.filter2D(src, cv2.CV_32F, kernel)


class FeatureMap:
    # difference feature maps
    def __init__(self,srcs):
        self.maps = self.__make_feature_map(srcs)

    def __make_feature_map(self, srcs):
        cs_index = ((0, 3), (0, 4), (1, 4), (1, 5), (2, 5), (2, 6))  # center-surround index
        maps = {'intensity': [],
                'colors': {'bg': [], 'ry': []},
                'orientations': {'0': [], '45': [], '90': [], '135': []}}
        for c, s in cs_index:
            maps['intensity'].append(self.__scale_diff(srcs['intensity'][c], srcs['intensity'][s]))  # difference operator
            for key in maps['orientations'].keys():
                maps['orientations'][key].append(self.__scale_diff(srcs['orientations'][key][c], srcs['orientations'][key][s]))  # difference operator
            for key in maps['colors'].keys():
                maps['colors'][key].append(self.__scale_color_diff(
                    srcs['colors'][key[0]][c], srcs['colors'][key[0]][s],
                    srcs['colors'][key[1]][c], srcs['colors'][key[1]][s]
                ))  # get the difference
        return maps

    def __scale_diff(self, c, s):
        # difference operator
        c_size = tuple(reversed(c.shape))
        return cv2.absdiff(c, cv2.resize(s, c_size, None, 0, 0, cv2.INTER_NEAREST))

    def __scale_color_diff(self,c1,s1,c2,s2):
        # difference operator
        c_size = tuple(reversed(c1.shape))
        return cv2.absdiff(c1 - c2, cv2.resize(s2 - s1, c_size, None, 0, 0, cv2.INTER_NEAREST))


class ConspicuityMap:
    # saliency map
    def __init__(self, srcs):
        self.maps = self.__make_conspicuity_map(srcs)

    def __make_conspicuity_map(self, srcs):
        norm_opt = ImgNormalize()
        intensity = self.__scale_add(list(map(norm_opt.normalize, srcs['intensity'])))  # add accumulately with brightness
        for key in srcs['colors'].keys():
            srcs['colors'][key] = list(map(norm_opt.normalize, srcs['colors'][key]))
        color = self.__scale_add([srcs['colors']['bg'][x] + srcs['colors']['ry'][x] for x in range(len(srcs['colors']['bg']))])  # add accumulately with color
        orientation = np.zeros(intensity.shape)
        for key in srcs['orientations'].keys():
            orientation += self.__scale_add(list(map(norm_opt.normalize, srcs['orientations'][key])))  # add accumulately with orientation
        return {'intensity': intensity, 'color': color, 'orientation': orientation}  #salient maps with three attributes

    def __scale_add(self, srcs):
        # add accumutely with difference of the same attribute
        buf = np.zeros(srcs[0].shape)
        for x in srcs:
            buf += cv2.resize(x, tuple(reversed(buf.shape)))
        return buf


class SaliencyMap:
    # saliency
    def __init__(self, src):
        self.gp = GaussianPyramid(src)
        self.fm = FeatureMap(self.gp.maps)
        self.cm = ConspicuityMap(self.fm.maps)
        self.map = cv2.resize(self.__make_saliency_map(self.cm.maps), tuple(reversed(src.shape[0:2])))

    def __make_saliency_map(self, srcs):
        util = ImgNormalize()
        srcs = list(map(util.normalize, [srcs[key] for key in srcs.keys()]))
        # add propotionally with each attribute将各个属性下的显著图等比例相加
        return srcs[0] / 3. + srcs[1] / 3. + srcs[2] / 3.


def get_sr_map(image):
    # initialize OpenCV's static saliency spectral residual detector 
    # and compute the saliency map
    sr_ops = cv2.saliency.StaticSaliencySpectralResidual_create()
    _, saliency_map_sr = sr_ops.computeSaliency(image)
    saliency_map_sr = (saliency_map_sr * 255).astype("uint8")

    return saliency_map_sr


def get_sf_map(image):
    # initialize OpenCV's static fine grained saliency detector and
    # compute the saliency map
    sf_ops = cv2.saliency.StaticSaliencyFineGrained_create()
    _, saliency_map_sf = sf_ops.computeSaliency(image)
    saliency_map_sf = (saliency_map_sf * 255).astype("uint8")

    return saliency_map_sf


def get_itti_map(image, img_normalizer):
    # load the input image
    sm = SaliencyMap(image)
    saliency_itti_map = np.uint8(img_normalizer.normalize_range(sm.map))

    return saliency_itti_map


def get_saliency_map(image,
                     mtype='itti',
                     img_normalizer=None,
                     fixed_size=(320, 320),
                     is_resize=False):
    """Get the saliency map from the image."""

    ori_shape = image.shape
    if is_resize:
        img = imresize(image, fixed_size)
    else:
        img = image

    sd_map = None
    if mtype == 'itti':
        if img_normalizer is None:
            img_normalizer = ImgNormalize()
        sd_map = get_itti_map(img, img_normalizer)
    elif mtype == 'sr':
        sd_map = get_sr_map(img)
    elif mtype == 'sf':
        sd_map = get_sf_map(img)
    else:
        raise TypeError(f"{mtype} is unsupported by the method.")

    if is_resize:
        return imresize(sd_map, (ori_shape[1], ori_shape[0]))

    return sd_map
