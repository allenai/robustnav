import os

scenes_lst = ["FloorPlan_Train10_1","FloorPlan_Train10_2","FloorPlan_Train10_3","FloorPlan_Train10_4","FloorPlan_Train10_5","FloorPlan_Train11_1","FloorPlan_Train11_2","FloorPlan_Train11_3","FloorPlan_Train11_4","FloorPlan_Train11_5","FloorPlan_Train12_1","FloorPlan_Train12_2","FloorPlan_Train12_3","FloorPlan_Train12_4","FloorPlan_Train12_5","FloorPlan_Train1_1","FloorPlan_Train1_2","FloorPlan_Train1_3","FloorPlan_Train1_4","FloorPlan_Train1_5","FloorPlan_Train2_1","FloorPlan_Train2_2","FloorPlan_Train2_3","FloorPlan_Train2_4","FloorPlan_Train2_5","FloorPlan_Train3_1","FloorPlan_Train3_2","FloorPlan_Train3_3","FloorPlan_Train3_4","FloorPlan_Train3_5","FloorPlan_Train4_1","FloorPlan_Train4_2","FloorPlan_Train4_3","FloorPlan_Train4_4","FloorPlan_Train4_5","FloorPlan_Train5_1","FloorPlan_Train5_2","FloorPlan_Train5_3","FloorPlan_Train5_4","FloorPlan_Train5_5","FloorPlan_Train6_1"]

for s in scenes_lst:
    print(f'Running scene: {s}')
    os.system(f'python sample_frames.py --scene={s}')