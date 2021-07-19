"""Entry point to training/validating/testing for a user given experiment
name."""
# Import these first to avoid
# conflict with other imports
# These are required for visual RGB corruptions
import skimage
import scipy

import argparse
import importlib
import inspect
import os
from typing import Dict, Tuple, List, Optional, Type

from setproctitle import setproctitle as ptitle

from allenact import __version__
from allenact.algorithms.onpolicy_sync.runner import OnPolicyRunner
from allenact.base_abstractions.experiment_config import ExperimentConfig
from allenact.utils.system import get_logger, init_logging, HUMAN_LOG_LEVELS


def get_args():
    """Creates the argument parser and parses any input arguments."""

    # noinspection PyTypeChecker
    parser = argparse.ArgumentParser(
        description="allenact", formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "experiment", type=str, help="experiment configuration file name",
    )

    parser.add_argument(
        "-et",
        "--extra_tag",
        type=str,
        default="",
        required=False,
        help="Add an extra tag to the experiment when trying out new ideas (will be used"
        "as a subdirectory of the tensorboard path so you will be able to"
        "search tensorboard logs using this extra tag).",
    )

    parser.add_argument(
        "-o",
        "--output_dir",
        required=False,
        type=str,
        default="experiment_output",
        help="experiment output folder",
    )

    parser.add_argument(
        "-s", "--seed", required=False, default=None, type=int, help="random seed",
    )
    parser.add_argument(
        "-b",
        "--experiment_base",
        required=False,
        default=os.getcwd(),
        type=str,
        help="experiment configuration base folder (default: working directory)",
    )
    parser.add_argument(
        "-c",
        "--checkpoint",
        required=False,
        default=None,
        type=str,
        help="optional checkpoint file name to resume training or test",
    )
    parser.add_argument(
        "--approx_ckpt_steps_count",
        required=False,
        default=None,
        type=float,
        help="if running testing, ",
    )
    parser.add_argument(
        "-r",
        "--restart_pipeline",
        dest="restart_pipeline",
        action="store_true",
        required=False,
        help="for training, if checkpoint is specified, DO NOT continue the training pipeline from where"
        "training had previously ended. Instead restart the training pipeline from scratch but"
        "with the model weights from the checkpoint.",
    )
    parser.set_defaults(restart_pipeline=False)

    parser.add_argument(
        "-d",
        "--deterministic_cudnn",
        dest="deterministic_cudnn",
        action="store_true",
        required=False,
        help="sets CuDNN in deterministic mode",
    )
    parser.set_defaults(deterministic_cudnn=False)

    parser.add_argument(
        "-t",
        "--test_date",
        default=None,
        type=str,
        required=False,
        help="tests the experiment run on specified date (formatted as %%Y-%%m-%%d_%%H-%%M-%%S), assuming it was "
        "previously trained. If no checkpoint is specified, it will run on all checkpoints enabled by "
        "skip_checkpoints",
    )

    parser.add_argument(
        "-k",
        "--skip_checkpoints",
        required=False,
        default=0,
        type=int,
        help="optional number of skipped checkpoints between runs in test if no checkpoint specified",
    )

    parser.add_argument(
        "-m",
        "--max_sampler_processes_per_worker",
        required=False,
        default=None,
        type=int,
        help="maximal number of sampler processes to spawn for each worker",
    )

    parser.add_argument(
        "--gp", default=None, action="append", help="values to be used by gin-config.",
    )

    parser.add_argument(
        "-e",
        "--deterministic_agents",
        dest="deterministic_agents",
        action="store_true",
        required=False,
        help="enable deterministic agents (i.e. always taking the mode action) during validation/testing",
    )
    parser.set_defaults(deterministic_agents=False)

    parser.add_argument(
        "-vc",
        "--visual_corruption",
        default=None,
        type=str,
        required=False,
        help="Visual corruption to be applied to egocentric RGB observation",
    )

    parser.add_argument(
        "-vs",
        "--visual_severity",
        default=0,
        type=int,
        required=False,
        help="Severity of visual corruption to be applied",
    )

    parser.add_argument(
        "-dcr",
        "--dyn_corr_mode",
        dest="dyn_corr_mode",
        required=False,
        action="store_true",
        help="Whether to apply dynamics corruptions",
    )
    parser.set_defaults(dyn_corr_mode=False)

    parser.add_argument(
        "-mf",
        "--motor_failure",
        dest="motor_failure",
        required=False,
        action="store_true",
        help="Whether to apply motor failure as the dynamics corruption",
    )
    parser.set_defaults(motor_failure=False)

    parser.add_argument(
        "-ctr",
        "--const_translate",
        dest="const_translate",
        required=False,
        action="store_true",
        help="Whether to apply constant translation bias as the dynamics corruption",
    )
    parser.set_defaults(const_translate=False)

    parser.add_argument(
        "-crt",
        "--const_rotate",
        dest="const_rotate",
        required=False,
        action="store_true",
        help="Whether to apply constant rotation bias as the dynamics corruption",
    )
    parser.set_defaults(const_rotate=False)

    parser.add_argument(
        "-str",
        "--stoch_translate",
        dest="stoch_translate",
        required=False,
        action="store_true",
        help="Whether to apply stochastic translation bias as the dynamics corruption",
    )
    parser.set_defaults(stoch_translate=False)

    parser.add_argument(
        "-srt",
        "--stoch_rotate",
        dest="stoch_rotate",
        required=False,
        action="store_true",
        help="Whether to apply stochastic rotation bias as the dynamics corruption",
    )
    parser.set_defaults(stoch_rotate=False)

    parser.add_argument(
        "-dr",
        "--drift",
        dest="drift",
        required=False,
        action="store_true",
        help="Whether to apply drift in translation as the dynamics corruption",
    )
    parser.set_defaults(drift=False)

    parser.add_argument(
        "-dr_deg",
        "--drift_degrees",
        default=1.15,
        type=float,
        required=False,
        help="Drift angle for the motion-drift dynamics corruption",
    )

    parser.add_argument(
        "-trd",
        "--training_dataset",
        default=None,
        type=str,
        required=False,
        help="Specify the training dataset",
    )

    parser.add_argument(
        "-vld",
        "--validation_dataset",
        default=None,
        type=str,
        required=False,
        help="Specify the validation dataset",
    )

    parser.add_argument(
        "-tsd",
        "--test_dataset",
        default=None,
        type=str,
        required=False,
        help="Specify the testing dataset",
    )

    parser.add_argument(
        "-irc",
        "--random_crop",
        dest="random_crop",
        required=False,
        action="store_true",
        help="Specify if random crop is to be applied to the egocentric observations",
    )
    parser.set_defaults(random_crop=False)

    parser.add_argument(
        "-icj",
        "--color_jitter",
        dest="color_jitter",
        required=False,
        action="store_true",
        help="Specify if random crop is to be applied to the egocentric observations",
    )
    parser.set_defaults(color_jitter=True)

    parser.add_argument(
        "-irs",
        "--random_shift",
        dest="random_shift",
        required=False,
        action="store_true",
        help="Specify if random shift is to be applied to the egocentric observations",
    )
    parser.set_defaults(random_shift=False)

    parser.add_argument(
        "-tsg",
        "--test_gpus",
        default=None,
        # type=int,
        type=str,
        required=False,
        help="Specify the GPUs to run test on",
    )

    parser.add_argument(
        "-l",
        "--log_level",
        default="info",
        type=str,
        required=False,
        help="sets the log_level. it must be one of {}.".format(
            ", ".join(HUMAN_LOG_LEVELS)
        ),
    )

    parser.add_argument(
        "-em",
        "--encoder_model",
        default=None,
        type=str,
        required=False,
        help="Name of custom state encoder model",
    )

    parser.add_argument(
        "-emb",
        "--encoder_base",
        default=None,
        type=str,
        required=False,
        help="Name of custom state encoder model",
    )

    parser.add_argument(
        "-sckpt",
        "--state_ckpt_path",
        default=None,
        type=str,
        required=False,
        help="Path to encoder model weight",
    )

    parser.add_argument(
        "-las",
        "--latent_size",
        default=None,
        type=int,
        required=False,
        help="State encoder model latent size",
    )

    parser.add_argument(
        "-i",
        "--disable_tensorboard",
        dest="disable_tensorboard",
        action="store_true",
        required=False,
        help="disable tensorboard logging",
    )
    parser.set_defaults(disable_tensorboard=False)

    parser.add_argument(
        "-a",
        "--disable_config_saving",
        dest="disable_config_saving",
        action="store_true",
        required=False,
        help="disable saving the used config in the output directory",
    )
    parser.set_defaults(disable_config_saving=False)

    parser.add_argument(
        "--version", action="version", version=f"allenact {__version__}"
    )

    return parser.parse_args()


def _config_source(config_type: Type) -> Dict[str, str]:
    if config_type is ExperimentConfig:
        return {}

    try:
        module_file_path = inspect.getfile(config_type)
        module_dot_path = config_type.__module__
        sources_dict = {module_file_path: module_dot_path}
        for super_type in config_type.__bases__:
            sources_dict.update(_config_source(super_type))

        return sources_dict
    except TypeError as _:
        return {}


def find_sub_modules(path: str, module_list: Optional[List] = None):
    if module_list is None:
        module_list = []

    path = os.path.abspath(path)
    if path[-3:] == ".py":
        module_list.append(path)
    elif os.path.isdir(path):
        contents = os.listdir(path)
        if any(key in contents for key in ["__init__.py", "setup.py"]):
            new_paths = [os.path.join(path, f) for f in os.listdir(path)]
            for new_path in new_paths:
                find_sub_modules(new_path, module_list)
    return module_list


def load_config(args) -> Tuple[ExperimentConfig, Dict[str, str]]:
    assert os.path.exists(
        args.experiment_base
    ), "The path '{}' does not seem to exist (your current working directory is '{}').".format(
        args.experiment_base, os.getcwd()
    )
    rel_base_dir = os.path.relpath(  # Normalizing string representation of path
        os.path.abspath(args.experiment_base), os.getcwd()
    )
    rel_base_dot_path = rel_base_dir.replace("/", ".")
    if rel_base_dot_path == ".":
        rel_base_dot_path = ""

    exp_dot_path = args.experiment
    if exp_dot_path[-3:] == ".py":
        exp_dot_path = exp_dot_path[:-3]
    exp_dot_path = exp_dot_path.replace("/", ".")

    module_path = (
        f"{rel_base_dot_path}.{exp_dot_path}"
        if len(rel_base_dot_path) != 0
        else exp_dot_path
    )

    try:
        importlib.invalidate_caches()
        module = importlib.import_module(module_path)
    except ModuleNotFoundError as e:
        if not any(isinstance(arg, str) and module_path in arg for arg in e.args):
            raise e
        all_sub_modules = set(find_sub_modules(os.getcwd()))
        desired_config_name = module_path.split(".")[-1]
        relevant_submodules = [
            sm for sm in all_sub_modules if desired_config_name in os.path.basename(sm)
        ]
        raise ModuleNotFoundError(
            "Could not import experiment '{}', are you sure this is the right path?"
            " Possibly relevant files include {}.".format(
                module_path, relevant_submodules
            ),
        ) from e

    experiments = [
        m[1]
        for m in inspect.getmembers(module, inspect.isclass)
        if m[1].__module__ == module.__name__ and issubclass(m[1], ExperimentConfig)
    ]
    assert (
        len(experiments) == 1
    ), "Too many or two few experiments defined in {}".format(module_path)

    config = experiments[0]()
    sources = _config_source(config_type=experiments[0])
    return config, sources


def main():
    args = get_args()

    if args.visual_corruption is not None and args.visual_severity > 0:  # This works
        VISUAL_CORRUPTION = [args.visual_corruption.replace("_", " ")]
        VISUAL_SEVERITY = [args.visual_severity]
    else:
        VISUAL_CORRUPTION = args.visual_corruption
        VISUAL_SEVERITY = args.visual_severity

    TRAINING_DATASET_DIR = args.training_dataset
    VALIDATION_DATASET_DIR = args.validation_dataset
    TEST_DATASET_DIR = args.test_dataset

    RANDOM_CROP = args.random_crop
    COLOR_JITTER = args.color_jitter
    RANDOM_SHIFT = args.random_shift

    # Dynamics Corruptions
    MOTOR_FAIL = args.motor_failure
    CONST_TRANSLATE = args.const_translate
    CONST_ROTATE = args.const_rotate
    STOCH_TRANSLATE = args.stoch_translate
    STOCH_ROTATE = args.stoch_rotate
    DRIFT = args.drift
    DRIFT_DEG = args.drift_degrees

    TEST_GPU_IDS = None
    if args.test_gpus is not None:
        TEST_GPU_IDS = [int(x) for x in args.test_gpus.split(",")]

    init_logging(args.log_level)

    get_logger().info("Running with args {}".format(args))

    ptitle("Master: {}".format("Training" if args.test_date is None else "Testing"))

    cfg, srcs = load_config(args)

    if TEST_GPU_IDS is not None:
        cfg.TEST_GPU_IDS = TEST_GPU_IDS

    print(cfg.TEST_GPU_IDS)

    cfg.monkey_patch_sensor(
        VISUAL_CORRUPTION, VISUAL_SEVERITY, RANDOM_CROP, COLOR_JITTER, RANDOM_SHIFT
    )

    cfg.monkey_patch_datasets(
        TRAINING_DATASET_DIR, VALIDATION_DATASET_DIR, TEST_DATASET_DIR
    )

    if args.experiment == 'objectnav_robothor_vanilla_rgb_custom_ddppo':
        if args.encoder_model is None:
            raise Exception('objectnav_robothor_vanilla_rgb_custom_ddppo should have a state model specified')
        else:
            cfg.create_preprocessor(args.encoder_model, args.state_ckpt_path, args.encoder_base, args.latent_size)

    if args.dyn_corr_mode:
        cfg.monkey_patch_env_args(
            MOTOR_FAIL,
            CONST_TRANSLATE,
            CONST_ROTATE,
            STOCH_TRANSLATE,
            STOCH_ROTATE,
            DRIFT,
            DRIFT_DEG,
        )

    if args.test_date is None:
        OnPolicyRunner(
            config=cfg,
            output_dir=args.output_dir,
            loaded_config_src_files=srcs,
            seed=args.seed,
            mode="train",
            deterministic_cudnn=args.deterministic_cudnn,
            deterministic_agents=args.deterministic_agents,
            extra_tag=args.extra_tag,
            disable_tensorboard=args.disable_tensorboard,
            disable_config_saving=args.disable_config_saving,
        ).start_train(
            checkpoint=args.checkpoint,
            restart_pipeline=args.restart_pipeline,
            max_sampler_processes_per_worker=args.max_sampler_processes_per_worker,
        )
    else:
        OnPolicyRunner(
            config=cfg,
            output_dir=args.output_dir,
            loaded_config_src_files=srcs,
            seed=args.seed,
            mode="test",
            deterministic_cudnn=args.deterministic_cudnn,
            deterministic_agents=args.deterministic_agents,
            extra_tag=args.extra_tag,
            disable_tensorboard=args.disable_tensorboard,
            disable_config_saving=args.disable_config_saving,
        ).start_test(
            experiment_date=args.test_date,
            checkpoint_name_fragment=args.checkpoint,
            approx_ckpt_steps_count=args.approx_ckpt_steps_count,
            skip_checkpoints=args.skip_checkpoints,
            max_sampler_processes_per_worker=args.max_sampler_processes_per_worker,
        )


if __name__ == "__main__":
    main()
