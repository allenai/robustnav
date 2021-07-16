import os
import uuid
import argparse
import numpy as np
from PIL import Image as im
from ai2thor.controller import Controller

def sample_action():
    sample = np.random.choice(5, 1)[0]

    if sample == 0:
        return 'MoveAhead'
    elif sample == 1:
        return 'RotateRight'
    elif sample == 2:
        return 'RotateLeft'
    elif sample == 3:
        return 'LookUp'
    else:
        return 'LookDown'


def sample(args):
    fail_count = 0
    controller = Controller(gridSize=0.25, movementGaussianSigma=0, rotateGaussianSigma=0, renderDepthImage=False, renderInstanceSegmentation=False, **args)
    total_success = 0
    total_fail = 0

    for i in range(STEPS):
        action = sample_action()
        event = controller.step(action=action)
        isActSucc = event.metadata['lastActionSuccess']
        frame_arr = event.frame
        image_id = None

        if isActSucc:
            fail_count = 0
            total_success += 1
            image = im.fromarray(frame_arr)
            image_id = uuid.uuid1()
            image.save(f'{SAVE_PATH}/{image_id}.png')
        else:
            total_fail += 1
            fail_count += 1

            if fail_count > FAIL_COUNT:
            controller.reset(args)

        if i % PRINT_EVERY == 0:
            print(f'Actions Succeeded: {total_success}')
            print(f'Actions Failed: {total_fail}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="config")

    parser.add_argument(
        "--scene",
        nargs="?",
        type=str,
        default="FloorPlan_Train1_1",
        help="RoboThor scene to run"
    )

    parser.add_argument(
        "--x_display",
        nargs="?",
        type=str,
        default='10.0',
        help="x display port"
    )

    parser.add_argument(
        "--height",
        nargs="?",
        type=int,
        default=64,
        help="frames height",
    )

    parser.add_argument(
        "--width",
        nargs="?",
        type=int,
        default=64,
        help="frames width",
    )

    parser.add_argument(
        "--fieldOfView",
        nargs="?",
        type=int,
        default=60,
        help="Angle of field of view"
    )

    parser.add_argument(
        "--rotateStepDegrees",
        nargs="?",
        type=int,
        default=30,
        help="How much agent rotates when rotate is called"
    )

    parser.add_argument(
        "--savepath",
        nargs="?",
        type=str,
        default='',
        help="Path to save frames"
    )

    parser.add_argument(
        "--steps",
        nargs="?",
        type=int,
        default=1000,
        help="Number of steps"
    )

    parser.add_argument(
        "--failcount",
        nargs="?",
        type=int,
        default=30,
        help="Number of action fails"
    )

    parser.add_argument(
        "--printevery",
        nargs="?",
        type=int,
        default=100,
        help="Print every count"
    )

    args = parser.parse_args()

    PRINT_EVERY = args.printevery
    SAVE_PATH = args.savepath
    FAIL_COUNT = args.failcount
    STEPS = args.steps

    if SAVE_PATH == '':
        SAVE_PATH = f'./64_frames/{args.scene}'

        if not os.path.exists(SAVE_PATH):
                os.makedirs(SAVE_PATH, exist_ok=True)
    
    del args.printevery
    del args.failcount
    del args.savepath
    del args.steps

    sample(vars(args))