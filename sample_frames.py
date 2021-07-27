import os
import json
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
    samples_data = dict()
    object_counts = None

    for i in range(STEPS):
        action = sample_action()
        event = controller.step(action=action)
        isActSucc = event.metadata['lastActionSuccess']
        image_id = None

        if isActSucc:
            fail_count = 0
            total_success += 1
            frame_arr = event.frame
            image = im.fromarray(frame_arr)
            image_id = str(uuid.uuid1())
            image.save(f'{SAVE_PATH}/{image_id}.png')

            hrzn = event.metadata['agent']['cameraHorizon']
            pos = event.metadata['agent']['position']
            rot = event.metadata['agent']['rotation']
            samples_data[image_id] = {'Horizon' : hrzn, 'Position' : pos, 'Rotation': rot}

            if object_counts is None:
                object_counts = dict()
                for obj in event.metadata['objects']:
                    object_counts[obj['name']] = 1 if obj['visible'] else 0
            else:
                for obj in event.metadata['objects']:
                    if obj['visible']:
                        object_counts[obj['name']] += 1
        else:
            total_fail += 1
            fail_count += 1

            if fail_count > FAIL_COUNT:
                controller.reset(args)

        if i % PRINT_EVERY == 0:
            print(f'Actions Succeeded: {total_success}')
            print(f'Actions Failed: {total_fail}')
    
    object_counts['total_saves'] = total_success

    with open(f"{SAVE_PATH}/samples_meta.json", 'w') as f:
        json.dump(samples_data, f, indent=4)
    
    with open(f"{SAVE_PATH}/objects_meta.json", 'w') as f:      
        json.dumps(object_counts, f, indent=4)


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
        default=480,
        help="frames height",
    )

    parser.add_argument(
        "--width",
        nargs="?",
        type=int,
        default=640,
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
        default=None,
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

    if SAVE_PATH is None:
        SAVE_PATH = f'./unknown_frames/{args.scene}'

    if not os.path.exists(SAVE_PATH):
        os.makedirs(SAVE_PATH, exist_ok=True)
    
    del args.printevery
    del args.failcount
    del args.savepath
    del args.steps

    sample(vars(args))