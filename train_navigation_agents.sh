# Script to train navigation agents
# for both PointNav and ObjectNav

# PointNav RGB Agents
python main.py \
    -o storage/robothor-pointnav-rgb-resnetgru-ddppo \
    -b projects/robustnav_baselines/experiments/robustnav_train pointnav_robothor_vanilla_rgb_resnet_ddppo \
    -s 12345 \
    -et rnav_pointnav_vanilla_rgb_resnet_ddppo_clean

# PointNav RGB-D Agents
python main.py \
    -o storage/robothor-pointnav-rgbd-resnetgru-ddppo \
    -b projects/robustnav_baselines/experiments/robustnav_train pointnav_robothor_vanilla_rgbd_resnet_ddppo \
    -s 12345 \
    -et rnav_pointnav_vanilla_rgbd_resnet_ddppo_clean

# PointNav RGB Agents (with Data Augmentation)
python main.py \
    -o storage/robothor-pointnav-rgb-resnetgru-daug-ddppo \
    -b projects/robustnav_baselines/experiments/robustnav_train pointnav_robothor_vanilla_rgb_resnet_ddppo \
    -s 12345 \
    -et rnav_pointnav_vanilla_rgb_resnet_daug_ddppo_clean \
    -irc True \
    -icj True \
    -irs True

# PointNav RGB Agents (with Action Prediction)
python main.py \
    -o storage/robothor-pointnav-rgb-resnetgru-act-pred-ddppo \
    -b projects/robustnav_baselines/experiments/robustnav_train pointnav_robothor_vanilla_rgb_resnet_act_pred_ddppo \
    -s 12345 \
    -et rnav_pointnav_vanilla_rgb_resnet_act_pred_ddppo_clean

# PointNav RGB Agents (with Rotation Prediction)
python main.py \
    -o storage/robothor-pointnav-rgb-resnetgru-rot-pred-ddppo \
    -b projects/robustnav_baselines/experiments/robustnav_train pointnav_robothor_vanilla_rgb_resnet_rot_pred_ddppo \
    -s 12345 \
    -et rnav_pointnav_vanilla_rgb_resnet_ddppo_rot_pred_clean

# ObjectNav RGB Agents
python main.py \
    -o storage/robothor-objectnav-rgb-resnetgru-ddppo \
    -b projects/robustnav_baselines/experiments/robustnav_train objectnav_robothor_vanilla_rgb_resnet_ddppo \
    -s 12345 \
    -et rnav_objectnav_vanilla_rgb_resnet_ddppo_clean

# ObjectNav RGB-D Agents
python main.py \
    -o storage/robothor-objectnav-rgbd-resnetgru-ddppo \
    -b projects/robustnav_baselines/experiments/robustnav_train objectnav_robothor_vanilla_rgbd_resnet_ddppo \
    -s 12345 \
    -et rnav_objectnav_vanilla_rgbd_resnet_ddppo_clean