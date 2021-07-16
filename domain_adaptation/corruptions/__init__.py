from corruptions.corruptions import *

import collections

weather = ['rain', 'fog', 'frost', 'snow']

key2deg = collections.OrderedDict()
key2deg['gaussianNoise'] = gaussian_noise
key2deg['shotNoise'] = shot_noise
key2deg['impulseNoise'] = impulse_noise
key2deg['defocusBlur'] = defocus_blur
key2deg['glassBlur'] = glass_blur
key2deg['zoomBlur'] = zoom_blur
key2deg['snow'] = snow
key2deg['frost'] = frost
key2deg['fog'] = fog
key2deg['brightness'] = brightness
key2deg['contrast'] = contrast
key2deg['elastic'] = elastic_transform
# key2deg['pixelate'] = pixelate
key2deg['saturate'] = saturate
# key2deg['spatter'] = spatter
key2deg['speckleNoise'] = speckle_noise

key2deg['gaussianBlur'] = gaussian_blur
key2deg['motionBlur'] = motion_blur
key2deg['jpeg'] = jpeg_compression
key2deg['blackoutNoise'] = blackoutNoise
key2deg['additiveGaussianNoise'] = additiveGaussianNoise
key2deg['occlusion'] = occlusion
key2deg['rain'] = rain
