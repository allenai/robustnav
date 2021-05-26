"""
PyRobot Noise Models

Adapted from -- https://github.com/facebookresearch/habitat-sim/blob/bf412fc4968c623d04937b412255ec14841c769f/habitat_sim/agent/controls/pyrobot_noisy_controls.py

Parameters contributed from PyRobot
https://pyrobot.org/
https://github.com/facebookresearch/pyrobot

Please cite PyRobot if you use this noise model
"""

from typing import List, Optional, Sequence, Tuple, Union, cast, Dict

import attr
import copy
import math

import numpy as np
import scipy.stats
from numpy import float64, ndarray


class _TruncatedMultivariateGaussian:
    def __init__(self, mean, cov):
        self.mean = mean
        self.cov = cov

    def __attrs_post_init__(self):
        self.mean = np.array(self.mean)
        self.cov = np.array(self.cov)
        if len(self.cov.shape) == 1:
            self.cov = np.diag(self.cov)

        assert (
            np.count_nonzero(self.cov - np.diag(np.diagonal(self.cov))) == 0
        ), "Only supports diagonal covariance"

    def sample(
        self,
        truncation: Optional[
            Union[List[Optional[Tuple[float64, None]]], List[Tuple[float64, None]]]
        ] = None,
    ) -> ndarray:
        if truncation is not None:
            assert len(truncation) == len(self.mean)

        sample = np.zeros_like(self.mean)
        for i in range(len(self.mean)):
            stdev = np.sqrt(self.cov[i])
            mean = self.mean[i]

            # Always truncate to 3 standard deviations
            a, b = -3, 3

            if truncation is not None and truncation[i] is not None:
                trunc = truncation[i]
                if trunc[0] is not None:
                    a = max((trunc[0] - mean) / stdev, a)
                if trunc[1] is not None:
                    b = min((trunc[1] - mean) / stdev, b)

            sample[i] = scipy.stats.truncnorm.rvs(a, b, mean, stdev)

        return sample


class MotionNoiseModel:
    def __init__(self, linear, rotation):
        self.linear = linear
        self.rotation = rotation


class ControllerNoiseModel:
    def __init__(self, linear_motion, rotational_motion):
        self.linear_motion = linear_motion
        self.rotational_motion = rotational_motion


class RobotNoiseModel:
    def __init__(self, ILQR, Proportional, Movebase):
        self.ILQR = ILQR
        self.Proportional = Proportional
        self.Movebase = Movebase

    def __getitem__(self, key: str) -> ControllerNoiseModel:
        return getattr(self, key)


pyrobot_noise_models = {
    "LoCoBot": RobotNoiseModel(
        ILQR=ControllerNoiseModel(
            linear_motion=MotionNoiseModel(
                _TruncatedMultivariateGaussian([0.014, 0.009], [0.006, 0.005]),
                _TruncatedMultivariateGaussian([0.008], [0.004]),
            ),
            rotational_motion=MotionNoiseModel(
                _TruncatedMultivariateGaussian([0.003, 0.003], [0.002, 0.003]),
                _TruncatedMultivariateGaussian([0.023], [0.012]),
            ),
        ),
        Proportional=ControllerNoiseModel(
            linear_motion=MotionNoiseModel(
                _TruncatedMultivariateGaussian([0.017, 0.042], [0.007, 0.023]),
                _TruncatedMultivariateGaussian([0.031], [0.026]),
            ),
            rotational_motion=MotionNoiseModel(
                _TruncatedMultivariateGaussian([0.001, 0.005], [0.001, 0.004]),
                _TruncatedMultivariateGaussian([0.043], [0.017]),
            ),
        ),
        Movebase=ControllerNoiseModel(
            linear_motion=MotionNoiseModel(
                _TruncatedMultivariateGaussian([0.074, 0.036], [0.019, 0.033]),
                _TruncatedMultivariateGaussian([0.189], [0.038]),
            ),
            rotational_motion=MotionNoiseModel(
                _TruncatedMultivariateGaussian([0.002, 0.003], [0.0, 0.002]),
                _TruncatedMultivariateGaussian([0.219], [0.019]),
            ),
        ),
    ),
    "LoCoBot-Lite": RobotNoiseModel(
        ILQR=ControllerNoiseModel(
            linear_motion=MotionNoiseModel(
                _TruncatedMultivariateGaussian([0.142, 0.023], [0.008, 0.008]),
                _TruncatedMultivariateGaussian([0.031], [0.028]),
            ),
            rotational_motion=MotionNoiseModel(
                _TruncatedMultivariateGaussian([0.002, 0.002], [0.001, 0.002]),
                _TruncatedMultivariateGaussian([0.122], [0.03]),
            ),
        ),
        Proportional=ControllerNoiseModel(
            linear_motion=MotionNoiseModel(
                _TruncatedMultivariateGaussian([0.135, 0.043], [0.007, 0.009]),
                _TruncatedMultivariateGaussian([0.049], [0.009]),
            ),
            rotational_motion=MotionNoiseModel(
                _TruncatedMultivariateGaussian([0.002, 0.002], [0.002, 0.001]),
                _TruncatedMultivariateGaussian([0.054], [0.061]),
            ),
        ),
        Movebase=ControllerNoiseModel(
            linear_motion=MotionNoiseModel(
                _TruncatedMultivariateGaussian([0.192, 0.117], [0.055, 0.144]),
                _TruncatedMultivariateGaussian([0.128], [0.143]),
            ),
            rotational_motion=MotionNoiseModel(
                _TruncatedMultivariateGaussian([0.002, 0.001], [0.001, 0.001]),
                _TruncatedMultivariateGaussian([0.173], [0.025]),
            ),
        ),
    ),
}

_X_AXIS = 0
_Y_AXIS = 1
_Z_AXIS = 2


def get_teleport_location(
    curr_state: Dict,
    translate_amount: float,
    rotate_amount: float,
    multiplier: float,
    model: MotionNoiseModel,
    motion_type: str,
) -> Dict:
    """
    PyRobot Noise Models
    ======================
    Habitat uses global coordinates but
    AI2-THOR uses local coordinates

    Beyond different controller variations, PyRobot has the following
    general structure in terms of how noise is applied
    
    Agent X,Y,Z convention 
    - X in agent's heading
    - Y is the yaw axis (for rotation)
    - Z is out of 3rd party frame / to the right in ego-frame

    **Translation**
    - Whenever the agent moves, it always rotates by the noise amount in the
        "agent's" coordinate frame (not camera coordinates or global coordinates)
    - Note that the agent will always rotate (even for translation) by some fixed
        amount (very minimal)

    **Rotation**
    - Whenever the agent rotates, it will always move by the noise amount in the
        "agent's" coordinate frame (not camera coordinates or global coordinates)
    - Note that the agent will always translate (even for rotation) by some fixed
        amount (very minimal)

    """

    if motion_type == "rotational":
        translation_noise = multiplier * model.linear.sample()
    else:
        # The robot will always move a little bit.  This has to be defined based on the intended actuation
        # as otherwise small rotation amounts would be invalid.  However, pretty quickly, we'll
        # get to the truncation of 3 sigma
        translate_trunc = [(-0.95 * np.abs(translate_amount), None), None]

        translation_noise = multiplier * model.linear.sample(translate_trunc)

    # + EPS to make sure 0 is positive.  We multiply by the sign of the translation
    # as otherwise forward would overshoot on average and backward would undershoot, while
    # both should overshoot
    translation_noise *= np.sign(translate_amount + 1e-8)

    if motion_type == "linear":
        rot_noise = multiplier * model.rotation.sample()
    else:
        # The robot will always turn a little bit.  This has to be defined based on the intended actuation
        # as otherwise small rotation amounts would be invalid.  However, pretty quickly, we'll
        # get to the truncation of 3 sigma
        rotate_trunc = [(-0.95 * np.abs(np.deg2rad(rotate_amount)), None)]

        rot_noise = multiplier * model.rotation.sample(rotate_trunc)

    # Same deal with rotation about + EPS and why we multiply by the sign
    rot_noise *= np.sign(rotate_amount + 1e-8)

    # Get teleportation location
    teleport_state = copy.deepcopy(curr_state)
    teleport_state["x"] = (
        curr_state["x"]
        + (translate_amount + translation_noise[0])
        * math.sin(math.radians(curr_state["rotation"]["y"]))
        + translation_noise[1] * math.cos(math.radians(curr_state["rotation"]["y"]))
    )
    teleport_state["z"] = (
        curr_state["z"]
        + (translate_amount + translation_noise[0])
        * math.cos(math.radians(curr_state["rotation"]["y"]))
        - translation_noise[1] * math.sin(math.radians(curr_state["rotation"]["y"]))
    )
    teleport_state["rotation"]["y"] = (
        curr_state["rotation"]["y"] + rotate_amount + rot_noise[0]
    ) % 360.0

    return teleport_state
