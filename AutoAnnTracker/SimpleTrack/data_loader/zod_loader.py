""" Example of data loader:
    The data loader has to be an iterator:
    Return a dict of frame data
    Users may create the logic of your own data loader
"""
import os, numpy as np, json
from pyquaternion import Quaternion
#from nuscenes.utils.data_classes import Box
from zod.data_classes.box import Box3D
from zod.constants import Lidar, EGO
from mot_3d.data_protos import BBox
import mot_3d.utils as utils
from mot_3d.preprocessing import nms


def transform_matrix(translation: np.ndarray = np.array([0, 0, 0]),
                     rotation: np.ndarray = np.array([1, 0, 0, 0]),
                     inverse: bool = False) -> np.ndarray:
    tm = np.eye(4)
    rotation = Quaternion(rotation)

    if inverse:
        rot_inv = rotation.rotation_matrix.T
        trans = np.transpose(-np.array(translation))
        tm[:3, :3] = rot_inv
        tm[:3, 3] = rot_inv.dot(trans)
    else:
        tm[:3, :3] = rotation.rotation_matrix
        tm[:3, 3] = np.transpose(np.array(translation))
    return tm


def zod_array2mot_bbox(b):
    '''
    nu_box = Box(b[:3], b[3:6], Quaternion(b[6:10]))
    mot_bbox = BBox(
        x=nu_box.center[0], y=nu_box.center[1], z=nu_box.center[2],
        w=nu_box.wlh[0], l=nu_box.wlh[1], h=nu_box.wlh[2],
        o=nu_box.orientation.yaw_pitch_roll[0]
    )
    '''

    zod_box = Box3D(b[:3],b[3:6], Quaternion(b[6:10]), EGO)
    mot_bbox = BBox(
    x=zod_box.center[0], y=zod_box.center[1], z=zod_box.center[2],
    w=zod_box.size[1], l=zod_box.size[0], h=zod_box.size[2],
    #w=zod_box.size[0], l=zod_box.size[1], h=zod_box.size[2],
    o=zod_box.orientation.yaw_pitch_roll[0]
    )

    if len(b) == 11:
        mot_bbox.s = b[-1]
    return mot_bbox



class ZodLoader:
    def __init__(self, configs, type_token, segment_name, data_folder, det_data_folder, start_frame):
        """ initialize with the path to data 
        Args:
            data_folder (str): root path to your data
        """
        self.configs = configs
        self.segment = segment_name
        self.data_loader = data_folder
        self.det_data_folder = det_data_folder
        self.type_token = type_token

        self.ts_info = json.load(open(os.path.join(data_folder, 'ts_info', '{:}.json'.format(segment_name)), 'r'))
        self.ego_info = np.load(os.path.join(data_folder, 'ego_info', '{:}.npz'.format(segment_name)), 
            allow_pickle=True)
        #self.calib_info = np.load(os.path.join(data_folder, 'calib_info', '{:}.npz'.format(segment_name)),
        #    allow_pickle=True)
        self.dets = np.load(os.path.join(det_data_folder, 'dets', '{:}.npz'.format(segment_name)),
            allow_pickle=True)
        self.det_type_filter = True

        self.nms = configs['data_loader']['nms']
        self.nms_thres = configs['data_loader']['nms_thres']
        
        self.use_pc = configs['data_loader']['pc']
        if self.use_pc:
            self.pcs = np.load(os.path.join(data_folder, 'pc', 'raw_pc', '{:}.npz'.format(segment_name)),
                allow_pickle=True)

        self.max_frame = len(self.dets['bboxes'])
        self.cur_frame = start_frame
    
    def __iter__(self):
        return self
    
    def __next__(self):
        if self.cur_frame >= self.max_frame:
            raise StopIteration

        result = dict()
        result['time_stamp'] = self.ts_info[self.cur_frame] #* 1e-6
        ego = self.ego_info[str(self.cur_frame)]
        #ego_matrix = transform_matrix(ego[:3], ego[3:])
        ego_matrix = ego
        result['ego'] = ego_matrix
        bboxes = self.dets['bboxes'][self.cur_frame]
        inst_types = self.dets['types'][self.cur_frame]
        frame_bboxes = [bboxes[i] for i in range(len(bboxes)) if inst_types[i] in self.type_token]
        result['det_types'] = [inst_types[i] for i in range(len(inst_types)) if inst_types[i] in self.type_token]
        result['dets'] = [zod_array2mot_bbox(b) for b in frame_bboxes]
        result['aux_info'] = dict()
        if 'velos' in list(self.dets.keys()):
            cur_velos = self.dets['velos'][self.cur_frame]
            result['aux_info']['velos'] = [cur_velos[i] for i in range(len(cur_velos)) 
                if inst_types[i] in self.type_token]
        else:
            result['aux_info']['velos'] = None
        if self.nms:
            result['dets'], result['det_types'], result['aux_info']['velos'] = \
                self.frame_nms(result['dets'], result['det_types'], result['aux_info']['velos'], self.nms_thres)
        result['dets'] = [BBox.bbox2array(d) for d in result['dets']]

        result['pc'] = None
        if self.use_pc:
            pc = self.pcs[str(self.cur_frame)][:, :3]
            calib = self.calib_info[str(self.cur_frame)]
            calib_trans, calib_rot = np.asarray(calib[:3]), Quaternion(np.asarray(calib[3:]))
            pc = np.dot(pc, calib_rot.rotation_matrix.T)
            pc += calib_trans
            result['pc'] = utils.pc2world(ego_matrix, pc)
        
        # if 'velos' in list(self.dets.keys()):
        #     cur_frame_velos = self.dets['velos'][self.cur_frame]
        #     result['aux_info']['velos'] = [cur_frame_velos[i] 
        #         for i in range(len(bboxes)) if inst_types[i] in self.type_token]
        result['aux_info']['is_key_frame'] = True
        self.cur_frame += 1
        return result
    
    def __len__(self):
        return self.max_frame
    
    def frame_nms(self, dets, det_types, velos, thres):
        frame_indexes, frame_types = nms(dets, det_types, thres)
        result_dets = [dets[i] for i in frame_indexes]
        result_velos = None
        if velos is not None:
            result_velos = [velos[i] for i in frame_indexes]
        return result_dets, frame_types, result_velos
