import os, json

from zod import ZodSequences
from zod.constants import Lidar



def get_available_scenes(zod: ZodSequences, split):

    available_scenes = []
    print('total scene num: {}'.format(len(zod.get_all_ids())))

    for scene_index, scene_name in enumerate(zod.get_all_ids()):
        if scene_name not in split:
            continue
        seq = zod[scene_name]
        lidar_frames = seq.info.get_lidar_frames(lidar=Lidar.VELODYNE)
        
        scene_not_exist = False
        for frame in lidar_frames:
            lidar_path = frame.to_dict()['filepath']
            if os.getcwd() in lidar_path:
                lidar_path = lidar_path.split(f'{os.getcwd()}/')[-1]
            if not os.path.exists(lidar_path):
                scene_not_exist = True
                break
            else:
                break
        if scene_not_exist:
            continue
        #available_scenes.append(seq)
        available_scenes.append(scene_name)
    print('exist scene num: {}'.format(len(available_scenes)))
    return available_scenes

