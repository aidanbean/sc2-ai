#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function
import numpy as np
import tensorflow as tf
from pysc2.lib import features

_SCREEN_PLAYER_ID = features.SCREEN_FEATURES.player_id.index # just the position of unit-type in screen_feature class
_SCREEN_UNIT_TYPE = features.SCREEN_FEATURES.unit_type.index


def preprocess_screen(screen):
    layers = []
    assert screen.shape[0] == len(features.SCREEN_FEATURES)
    for i in range(len(features.SCREEN_FEATURES)):
        # screen[i:i+ 1] is (1, 64, 64), screen[i] is (64, 64), we do so b/c categorical feature
        # layers are encode as one-hot layers of size (scale, 64, 64), we need to concatenate all of them
        # print(screen[i].shape, screen[i: i + 1].shape)
        if i == _SCREEN_PLAYER_ID or i == _SCREEN_UNIT_TYPE: # player id or unit type has large categories
            layers.append(screen[i:i+1] / features.SCREEN_FEATURES[i].scale)
        elif features.SCREEN_FEATURES[i].type == features.FeatureType.SCALAR:
            layers.append(np.log(screen[i:i+1] + np.finfo(np.float32).eps)) # take log for scule data
        else:
            # create a scale size x screen size x screen size multi-dimension matrix of 0s
            layer = np.zeros([features.SCREEN_FEATURES[i].scale, screen.shape[1], screen.shape[2]], dtype=np.float32)
            for j in range(features.SCREEN_FEATURES[i].scale):
                # iterate each scale level j of feature i
                indy, indx = (screen[i] == j).nonzero() # get coor of all points whose value = j in feature layer i
                layer[j, indy, indx] = 1 # make all these point equal to value 1
            layers.append(layer)
    return np.concatenate(layers, axis=0)


def screen_channel():
    c = 0
    for i in range(len(features.SCREEN_FEATURES)):
        if i == _SCREEN_PLAYER_ID or i == _SCREEN_UNIT_TYPE:
            c += 1
        elif features.SCREEN_FEATURES[i].type == features.FeatureType.SCALAR:
            c += 1
        else:
            c += features.SCREEN_FEATURES[i].scale
    return c
