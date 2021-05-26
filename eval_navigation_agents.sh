: '
Script to evaluate navigation agents

========================================================
[ARGUMENT_SPECIFICATIONS]
========================================================

[*] Evaluating under "clean" conditions
----------------------------------------
python main.py \
    -o storage/robothor-pointnav-rgb-resnetgru-ddppo-eval \ # Folder where evaluation JSON (trajectories + metrics will be stored)
    -b projects/robustnav_baselines/experiments/robustnav_eval pointnav_robothor_vanilla_rgb_resnet_ddppo \ # Experiment config
    -c <path-to-the-checkpoint> \ # Path to the relevant checkpoint
    -t <time-stamp-associated-with-the-checkpoint> \ # Time-stamp associated with the checkpoint to ve evaluated
    -et rnav_pointnav_vanilla_rgb_resnet_ddppo_clean \ # Additional tag to be associated with the experiments
    -s 12345 \
    -e \ # Random seed
    -tsg 0 # GPU to run evaluation on

[*] Evaluating under "visual" corruptions
----------------------------------------

[**] For Defocus Blur, Motion Blur, Spatter, Low Lighting, Speckle Noise as visual corruptions
python main.py \
    -o storage/robothor-pointnav-rgb-resnetgru-ddppo-eval \
    -b projects/robustnav_baselines/experiments/robustnav_eval pointnav_robothor_vanilla_rgb_resnet_ddppo \
    -c <path-to-the-checkpoint> \
    -t <time-stamp-associated-with-the-checkpoint> \
    -et rnav_pointnav_vanilla_rgb_resnet_ddppo_"$CORR" \
    -s 12345 \
    -e \
    -tsg 0 \
    -vc Defocus_Blur \ # Visual corruption to be applied
    -vs 5 # Severity (1-5) of the visual corruption

[**] For Lower-FOV and Camera-Crack as visual corruptions, use the corresponding experiment configs
under projects/robustnav_baselines/experiments/robustnav_eval/

[*] Evaluating under "dynamics" corruptions
----------------------------------------

[**] For Motion-Bias (Constant & Stochastic), Motion Drift & Motor Failure
python main.py \
    -o storage/robothor-pointnav-rgb-resnetgru-ddppo-eval \
    -b projects/robustnav_baselines/experiments/robustnav_eval pointnav_robothor_vanilla_rgb_resnet_ddppo_dyn \
    -c <path-to-the-checkpoint> \
    -t <time-stamp-associated-with-the-checkpoint> \
    -et rnav_pointnav_vanilla_rgb_resnet_ddppo_clean_mb_const \
    -s 12345 \
    -e \
    -tsg 0 \
    -dcr True \ # Set to true for Motion-Bias, Motion-Drift & Motor-Failure
    -ctr True \ # Set to true for Motion-Bias (Constant) translation
    -crt True \ # Set to true for Motion-Bias (Constant) rotation
    -str True \ # Set to true for Motion-Bias (Stochastic) translation
    -srt True \ # Set to true for Motion-Bias (Stochastic) rotation
    -dr True \ # Set to true for Motion-Drift
    -dr_deg 10.0 \ # Drift angle for Motion-Drift
    -mf True # Set to true for Motor-Failure

[**] For PyRobot Noise Models, use the corresponding experiment config under
projects/robustnav_baselines/experiments/robustnav_eval/

========================================================
[TASKS & SENSOR SPECIFICATIONS]
========================================================

[*] For evaluating checkpoints corresponding to different tasks (PointNav & ObjectNav), pick the
corresponding experiment config from projects/robustnav_baselines/experiments/robustnav_eval/

[*] For evaluating checkpoints corresponding to different sensor specifications (RGB & RGB-D), pick the
corresponding experiment config from projects/robustnav_baselines/experiments/robustnav_eval/

========================================================
[CHECKPOINTS & TIME-STAMPS]
========================================================

[*] Arguments to enter for the `-t` command line argument for evaluation

[**] PointNav RGB
File: pnav_rgb_agent.pt
Timestamp: 2021-03-02_05-58-58

[**] PointNav RGB-D
File: pnav_rgbd_agent.pt
Timestamp: 2021-03-02_06-16-25

[**] ObjectNav RGB
File: onav_rgb_agent.pt
Timestamp: 2021-03-02_05-12-42

[**] ObjectNav RGB-D
File: onav_rgbd_agent.pt
Timestamp: 2021-02-09_22-35-15

[**] PointNav RGB (Data Augmentation)
File: pnav_rgb_daug_agent.pt
Timestamp: 2021-03-03_05-38-47

[**] PointNav RGB (Action Prediction)
File: pnav_rgb_act_pred_agent.pt
Timestamp: 2021-03-15_05-22-27

[**] PointNav RGB (Rotation Prediction)
File: pnav_rgb_rot_pred_agent.pt
Timestamp: 2021-03-15_07-55-57

[**] PointNav RGB (Action Prediction + SS-Adapt)

[***] Clean
File: pnav_rgb_act_pred_clean_adapt_agent.pt
Timestamp: 2021-03-16_04-17-24

[***] Lower-FOV
File: pnav_rgb_act_pred_fov_adapt_agent.pt
Timestamp: 2021-03-16_04-38-43

[***] Defocus Blur
File: pnav_rgb_act_pred_defocus_blur_agent.pt
Timestamp: 2021-03-16_04-49-44

[***] Camera-Crack
File: pnav_rgb_act_pred_cam_crack_adapt_agent.pt
Timestamp: 2021-03-16_05-06-14

[***] Spatter
File: pnav_rgb_act_pred_spatter_adapt_agent.pt
Timestamp: 2021-03-16_05-16-30

[**] PointNav RGB (Rotation Prediction Prediction + SS-Adapt)

[***] Clean
File: pnav_rgb_act_pred_clean_adapt_agent.pt
Timestamp: 2021-03-16_08-24-48

[***] Lower-FOV
File: pnav_rgb_act_pred_fov_adapt_agent.pt
Timestamp: 2021-03-16_08-37-44

[***] Defocus Blur
File: pnav_rgb_act_pred_defocus_blur_agent.pt
Timestamp: 2021-03-16_08-51-22

[***] Camera-Crack
File: pnav_rgb_act_pred_cam_crack_adapt_agent.pt
Timestamp: 2021-03-16_09-12-07

[***] Spatter
File: pnav_rgb_act_pred_spatter_adapt_agent.pt
Timestamp: 2021-03-16_09-25-38


'

# ================================================================================
# PointNav RGB agents
# ================================================================================

# Clean
sudo python main.py \
    -o storage/robothor-pointnav-rgb-resnetgru-ddppo-eval \
    -b projects/robustnav_baselines/experiments/robustnav_eval pointnav_robothor_vanilla_rgb_resnet_ddppo \
    -c rnav_checkpoints/pnav_rgb_agent.pt \
    -t 2021-03-02_05-58-58 \
    -et rnav_pointnav_vanilla_rgb_resnet_ddppo_clean \
    -s 12345 \
    -e \
    -tsg 0


# Visual Corruptions
# ********************************************************************************

# (Defocus Blur, Motion Blur, Spatter, Low Lighting, Speckle Noise)
for CORR in Defocus_Blur Lighting Speckle_Noise Spatter Motion_Blur
do
    sudo python main.py \
        -o storage/robothor-pointnav-rgb-resnetgru-ddppo-eval \
        -b projects/robustnav_baselines/experiments/robustnav_eval pointnav_robothor_vanilla_rgb_resnet_ddppo \
        -c rnav_checkpoints/pnav_rgb_agent.pt \
        -t 2021-03-02_05-58-58 \
        -et rnav_pointnav_vanilla_rgb_resnet_ddppo_"$CORR"_s5 \
        -s 12345 \
        -e \
        -tsg 0 \
        -vc $CORR \
        -vs 5
done

# Lower-FOV
sudo python main.py \
    -o storage/robothor-pointnav-rgb-resnetgru-ddppo-eval \
    -b projects/robustnav_baselines/experiments/robustnav_eval pointnav_robothor_vanilla_rgb_resnet_ddppo_fov \
    -c rnav_checkpoints/pnav_rgb_agent.pt \
    -t 2021-03-02_05-58-58 \
    -et rnav_pointnav_vanilla_rgb_resnet_ddppo_fov \
    -s 12345 \
    -e \
    -tsg 1

# Camera-Crack
sudo python main.py \
    -o storage/robothor-pointnav-rgb-resnetgru-ddppo-eval \
    -b projects/robustnav_baselines/experiments/robustnav_eval pointnav_robothor_vanilla_rgb_resnet_ddppo_cam_crack \
    -c rnav_checkpoints/pnav_rgb_agent.pt \
    -t 2021-03-02_05-58-58 \
    -et rnav_pointnav_vanilla_rgb_resnet_ddppo_cam_crack \
    -s 12345 \
    -e \
    -tsg 3

# Dynamics Corruptions
# ********************************************************************************

# Motion Bias (Constant)
sudo python main.py \
    -o storage/robothor-pointnav-rgb-resnetgru-ddppo-eval \
    -b projects/robustnav_baselines/experiments/robustnav_eval pointnav_robothor_vanilla_rgb_resnet_ddppo_dyn \
    -c rnav_checkpoints/pnav_rgb_agent.pt \
    -t 2021-03-02_05-58-58 \
    -et rnav_pointnav_vanilla_rgb_resnet_ddppo_clean_mb_const \
    -s 12345 \
    -e \
    -tsg 4 \
    -dcr True \
    -ctr True \
    -crt True
    
# Motion Bias (Stochastic)
sudo python main.py \
    -o storage/robothor-pointnav-rgb-resnetgru-ddppo-eval \
    -b projects/robustnav_baselines/experiments/robustnav_eval pointnav_robothor_vanilla_rgb_resnet_ddppo_dyn \
    -c rnav_checkpoints/pnav_rgb_agent.pt \
    -t 2021-03-02_05-58-58 \
    -et rnav_pointnav_vanilla_rgb_resnet_ddppo_clean_mb_stoch \
    -s 12345 \
    -e \
    -tsg 4 \
    -dcr True \
    -str True \
    -srt True
    
# Motion Drift
sudo python main.py \
    -o storage/robothor-pointnav-rgb-resnetgru-ddppo-eval \
    -b projects/robustnav_baselines/experiments/robustnav_eval pointnav_robothor_vanilla_rgb_resnet_ddppo_dyn \
    -c rnav_checkpoints/pnav_rgb_agent.pt \
    -t 2021-03-02_05-58-58 \
    -et rnav_pointnav_vanilla_rgb_resnet_ddppo_clean_drift_deg_10 \
    -s 12345 \
    -e \
    -tsg 4 \
    -dcr True \
    -dr True \
    -dr_deg 10.0
    
# Motor Failure
sudo python main.py \
    -o storage/robothor-pointnav-rgb-resnetgru-ddppo-eval \
    -b projects/robustnav_baselines/experiments/robustnav_eval pointnav_robothor_vanilla_rgb_resnet_ddppo_dyn \
    -c rnav_checkpoints/pnav_rgb_agent.pt \
    -t 2021-03-02_05-58-58 \
    -et rnav_pointnav_vanilla_rgb_resnet_ddppo_clean_motfail \
    -s 12345 \
    -e \
    -tsg 4 \
    -dcr True \
    -mf True
    
# PyRobot Noise Models
sudo python main.py \
    -o storage/robothor-pointnav-rgb-resnetgru-ddppo-eval \
    -b projects/robustnav_baselines/experiments/robustnav_eval pointnav_robothor_vanilla_rgb_resnet_ddppo_pyrobot_dyn \
    -c rnav_checkpoints/pnav_rgb_agent.pt \
    -t 2021-03-02_05-58-58 \
    -et rnav_pointnav_vanilla_rgb_resnet_ddppo_clean_pyrobot_ilqr_1 \
    -s 12345 \
    -e \
    -tsg 4

# Visual + Dynamics Corruptions
# ********************************************************************************

# Motion Bias (Stochastic)
# (Defocus Blur, Speckle Noise, Spatter)
for CORR in Defocus_Blur Speckle_Noise Spatter
do
    sudo python main.py \
        -o storage/robothor-pointnav-rgb-resnetgru-ddppo-eval \
        -b projects/robustnav_baselines/experiments/robustnav_eval pointnav_robothor_vanilla_rgb_resnet_ddppo_dyn \
        -c rnav_checkpoints/pnav_rgb_agent.pt \
        -t 2021-03-02_05-58-58 \
        -et rnav_pointnav_vanilla_rgb_resnet_ddppo_"$CORR"_s5_mb_stoch \
        -s 12345 \
        -e \
        -tsg 1 \
        -dcr True \
        -str True \
        -srt True \
        -vc $CORR \
        -vs 5
done

# Motion Drift
# (Defocus Blur, Speckle Noise, Spatter)
for CORR in Defocus_Blur Speckle_Noise Spatter
do
    sudo python main.py \
        -o storage/robothor-pointnav-rgb-resnetgru-ddppo-eval \
        -b projects/robustnav_baselines/experiments/robustnav_eval pointnav_robothor_vanilla_rgb_resnet_ddppo_dyn \
        -c rnav_checkpoints/pnav_rgb_agent.pt \
        -t 2021-03-02_05-58-58 \
        -et rnav_pointnav_vanilla_rgb_resnet_ddppo_"$CORR"_s5_drift_deg_10 \
        -s 12345 \
        -e \
        -tsg 3 \
        -dcr True \
        -dr True \
        -dr_deg 10.0 \
        -vc $CORR \
        -vs 5
done

# PyRobot Noise
## (Defocus Blur, Speckle Noise, Spatter)
for CORR in Defocus_Blur Speckle_Noise Spatter
do
    sudo python main.py \
        -o storage/robothor-pointnav-rgb-resnetgru-ddppo-eval \
        -b projects/robustnav_baselines/experiments/robustnav_eval pointnav_robothor_vanilla_rgb_resnet_ddppo_pyrobot_dyn \
        -c rnav_checkpoints/pnav_rgb_agent.pt \
        -t 2021-03-02_05-58-58 \
        -et rnav_pointnav_vanilla_rgb_resnet_ddppo_"$CORR"_s5_pyrobot_ilqr_1 \
        -s 12345 \
        -e \
        -tsg 2 \
        -vc $CORR \
        -vs 5
done

