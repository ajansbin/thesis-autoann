# Copyright (c) OpenMMLab. All rights reserved.
from collections import defaultdict
from os import path as osp
from typing import List, Optional

import mmcv
import numpy as np
import pyquaternion
from mmcv.utils import print_log
from zod.constants import Anonymization
from zod.eval.detection import EVALUATION_CLASSES, DetectionBox, EvalBoxes
from zod.eval.detection import evaluate_nuscenes_style as zod_eval

from ..core import show_result
from ..core.bbox import Box3DMode, Coord3DMode, LiDARInstance3DBoxes
from .builder import DATASETS
from .custom_3d import Custom3DDataset
from .pipelines import Compose

BLUR = Anonymization.BLUR.value


@DATASETS.register_module()
class ZodFramesDataset(Custom3DDataset):
    r"""Zenseact Open Dataset -- Frames.

    This class serves as the API for experiments on ZOD-Frames.

    Please refer to `Zenseac Open Dataset <https://zod.zenseact.com/download>`
    for data downloading.

    Args:
        ann_file (str): Path of annotation file.
        pipeline (list[dict], optional): Pipeline used for data processing.
            Defaults to None.
        data_root (str): Path of dataset root.
        classes (tuple[str], optional): Classes used in the dataset.
            Defaults to None.
        load_interval (int, optional): Interval of loading the dataset. It is
            used to uniformly sample the dataset. Defaults to 1.
        modality (dict, optional): Modality to specify the sensor data used
            as input. Defaults to None.
        box_type_3d (str, optional): Type of 3D box of this dataset.
            Based on the `box_type_3d`, the dataset will encapsulate the box
            to its original format then converted them to `box_type_3d`.
            Defaults to 'LiDAR' in this dataset. Available options includes.
            - 'LiDAR': Box in LiDAR coordinates.
            - 'Depth': Box in depth coordinates, usually for indoor dataset.
            - 'Camera': Box in camera coordinates.
        filter_empty_gt (bool, optional): Whether to filter empty GT.
            Defaults to True.
        test_mode (bool, optional): Whether the dataset is in test mode.
            Defaults to False.
        eval_version (bool, optional): Configuration version of evaluation.
            Defaults to  'detection_cvpr_2019'.
        use_valid_flag (bool, optional): Whether to use `use_valid_flag` key
            in the info file as mask to filter gt_boxes and gt_names.
            Defaults to False.
        single_cam_mode (bool, optional): Treat zod a single-image dataset. By
            default zod operates like a multi-camera dataset that only has a
            single camera. This affects how the image loading pipeline works.
    """

    CLASSES = EVALUATION_CLASSES

    def __init__(
        self,
        ann_file,
        pipeline=None,
        data_root=None,
        classes=None,
        load_interval=1,
        modality=None,
        box_type_3d='LiDAR',
        filter_empty_gt=True,
        test_mode=False,
        eval_version='detection_cvpr_2019',  # TODO: remove?
        use_valid_flag=True,
        anonymization_mode=BLUR,
        use_png=False,
        single_cam_mode=True,
    ):
        self.load_interval = load_interval
        self.use_valid_flag = use_valid_flag
        super().__init__(
            data_root=data_root,
            ann_file=ann_file,
            pipeline=pipeline,
            classes=classes,
            modality=modality,
            box_type_3d=box_type_3d,
            filter_empty_gt=filter_empty_gt,
            test_mode=test_mode,
        )
        self.anonymization_mode = anonymization_mode
        self.use_png = use_png
        self.eval_version = eval_version
        self.single_cam_mode = single_cam_mode
        # from nuscenes.eval.detection.config import config_factory

        # self.eval_detection_configs = config_factory(self.eval_version)
        if self.modality is None:
            self.modality = dict(
                use_camera=False,
                use_lidar=True,
                use_radar=False,
                use_map=False,
                use_external=False,
            )

        # Maybe change image paths depending on settings
        if self.anonymization_mode != BLUR:
            self._rename_image_paths(
                lambda x: x.replace(BLUR, self.anonymization_mode))
        if self.use_png:
            self._rename_image_paths(lambda x: x.replace('.jpg', '.png'))

    def _rename_image_paths(self, rename_func):
        """Rename image paths.

        Args:
            rename_func (function): Function to rename image paths.
        """
        for info in self.data_infos:
            for info in info['cams'].values():
                info['data_path'] = rename_func(info['data_path'])

    def get_cat_ids(self, idx):
        """Get category distribution of single scene.

        Args:
            idx (int): Index of the data_info.

        Returns:
            dict[list]: for each category, if the current scene
                contains such boxes, store a list containing idx,
                otherwise, store empty list.
        """
        info = self.data_infos[idx]
        if self.use_valid_flag:
            mask = info['valid_flag']
            gt_names = set(info['gt_names'][mask])
        else:
            gt_names = set(info['gt_names'])

        cat_ids = []
        for name in gt_names:
            if name in self.CLASSES:
                cat_ids.append(self.cat2id[name])
        return cat_ids

    def load_annotations(self, ann_file):
        """Load annotations from ann_file.

        Args:
            ann_file (str): Path of the annotation file.

        Returns:
            list[dict]: List of annotations sorted by timestamps.
        """
        data = mmcv.load(ann_file, file_format='pkl')
        data_infos = list(sorted(data['infos'], key=lambda e: e['timestamp']))
        data_infos = data_infos[::self.load_interval]
        self.metadata = data['metadata']
        self.version = self.metadata['version']
        return data_infos

    def get_data_info(self, index):
        """Get data info according to the given index.

        Args:
            index (int): Index of the sample data to get.

        Returns:
            dict: Data information that will be passed to the data
                preprocessing pipelines. It includes the following keys:

                - sample_idx (str): Sample index.
                - pts_filename (str): Filename of point clouds.
                - sweeps (list[dict]): Infos of sweeps.
                - timestamp (float): Sample timestamp.
                - img_filename (str, optional): Image filename.
                - lidar2img (list[np.ndarray], optional): Transformations
                    from lidar to different cameras.
                - ann_info (dict): Annotation info.
        """
        info = self.data_infos[index]
        # standard protocol modified from SECOND.Pytorch
        input_dict = dict(
            sample_idx=info['frame_id'],
            pts_filename=info['lidar_path'],
            sweeps=info['sweeps'],
            timestamp=info['timestamp'] / 1e6,
            img_prefix=None,
        )

        if self.modality['use_camera']:
            cam_input = defaultdict(list)
            # NOTE: we only have a single camera in zod for now
            for _, cam_info in info['cams'].items():
                cam_input['img_filename'].append(cam_info['data_path'])
                # obtain lidar to image transformation matrix
                lidar2cam_r = np.linalg.inv(cam_info['sensor2lidar_rotation'])
                lidar2cam_t = cam_info[
                    'sensor2lidar_translation'] @ lidar2cam_r.T
                lidar2cam_rt = np.eye(4)
                lidar2cam_rt[:3, :3] = lidar2cam_r
                lidar2cam_rt[:3, 3] = -lidar2cam_t
                cam_input['lidar2cam'].append(lidar2cam_rt)
                cam_input['cam2img'].append(cam_info['cam_intrinsic'])
                cam_input['distortion'].append(cam_info['cam_distortion'])
                cam_input['undistortion'].append(cam_info['cam_undistortion'])
                cam_input['proj_model'].append(cam_info['proj_model'])
                # viewpad = np.eye(4)
                # viewpad[:intrinsic.shape[0], :intrinsic.shape[1]] = intrinsic
                # lidar2img_rt = viewpad @ lidar2cam_rt.T
                # lidar2img_rts.append(lidar2img_rt)
            if self.single_cam_mode:
                for key, value in list(cam_input.items()):
                    assert len(value) == 1
                    cam_input[key] = value[0]
                cam_input['img_info'] = dict(
                    filename=cam_input['img_filename'])
            input_dict.update(cam_input)

        if not self.test_mode:
            annos = self.get_ann_info(index)
            input_dict['ann_info'] = annos

        return input_dict

    def get_ann_info(self, index):
        """Get annotation info according to the given index.

        Args:
            index (int): Index of the annotation data to get.

        Returns:
            dict: Annotation information consists of the following keys:

                - gt_bboxes_3d (:obj:`LiDARInstance3DBoxes`):
                    3D ground truth bboxes
                - gt_labels_3d (np.ndarray): Labels of ground truths.
                - gt_names (list[str]): Class names of ground truths.
        """
        info = self.data_infos[index]
        # Always remove objects without 3d bounding boxes and ignore objects
        mask = info['has_3d'] & (~info['is_ignore'])
        if self.use_valid_flag:
            mask &= info['valid_flag']
        gt_bboxes_3d = info['gt_boxes'][mask]
        gt_names_3d = info['gt_names'][mask]
        gt_labels_3d = []
        for cat in gt_names_3d:
            if cat in self.CLASSES:
                gt_labels_3d.append(self.CLASSES.index(cat))
            else:
                gt_labels_3d.append(-1)
        gt_labels_3d = np.array(gt_labels_3d)

        # the nuscenes box center is [0.5, 0.5, 0.5], we change it to be
        # the same as KITTI (0.5, 0.5, 0)
        gt_bboxes_3d = LiDARInstance3DBoxes(
            gt_bboxes_3d,
            box_dim=gt_bboxes_3d.shape[-1],
            origin=(0.5, 0.5, 0.5)).convert_to(self.box_mode_3d)

        anns_results = dict(
            gt_bboxes_3d=gt_bboxes_3d,
            gt_labels_3d=gt_labels_3d,
            gt_names=gt_names_3d)
        return anns_results

    def _format_bbox(self, results, jsonfile_prefix=None):
        """Convert the results to the standard format.

        Args:
            results (list[dict]): Testing results of the dataset.
            jsonfile_prefix (str): The prefix of the output jsonfile.
                You can specify the output directory/filename by
                modifying the jsonfile_prefix. Default: None.

        Returns:
            str: Path of the output json file.
        """
        zod_annos = {}
        mapped_class_names = self.CLASSES

        print('Start to convert detection format...')
        for sample_id, det in enumerate(mmcv.track_iter_progress(results)):
            annos = []
            sample_token = self.data_infos[sample_id]['frame_id']
            boxes = self._det_to_zod(det, sample_token)

            for i, box in enumerate(boxes):
                name = box.detection_name
                attr = ""
                
                zod_anno = dict(
                    sample_token=sample_token,
                    translation=list(box.translation),
                    size=list(box.size),
                    rotation=list(box.rotation),
                    velocity=[0,0,0],
                    detection_name=name,
                    detection_score=box.detection_score,
                    attribute_name=attr)
                annos.append(zod_anno)
            zod_annos[sample_token] = annos
        nusc_submissions = {
            'meta': self.modality,
            'results': zod_annos,
        }

        mmcv.mkdir_or_exist(jsonfile_prefix)
        res_path = osp.join(jsonfile_prefix, 'results_zod.json')
        print('Results writes to', res_path)
        mmcv.dump(nusc_submissions, res_path)
        return res_path

    def format_results(self, results, jsonfile_prefix=None):
        """Format the results to json (standard format for COCO evaluation).

        Args:
            results (list[dict]): Testing results of the dataset.
            jsonfile_prefix (str): The prefix of json files. It includes
                the file path and the prefix of filename, e.g., "a/b/prefix".
                If not specified, a temp file will be created. Default: None.

        Returns:
            tuple: Returns (result_files, tmp_dir), where `result_files` is a
                dict containing the json filepaths, `tmp_dir` is the temporal
                directory created for saving json files when
                `jsonfile_prefix` is not specified.
        """
        assert isinstance(results, list), 'results must be a list'
        assert len(results) == len(self), (
            'The length of results is not equal to the dataset len: {} != {}'.
            format(len(results), len(self)))

        if jsonfile_prefix is None:
            tmp_dir = tempfile.TemporaryDirectory()
            jsonfile_prefix = osp.join(tmp_dir.name, 'results')
        else:
            tmp_dir = None

        # currently the output prediction results could be in two formats
        # 1. list of dict('boxes_3d': ..., 'scores_3d': ..., 'labels_3d': ...)
        # 2. list of dict('pts_bbox' or 'img_bbox':
        #     dict('boxes_3d': ..., 'scores_3d': ..., 'labels_3d': ...))
        # this is a workaround to enable evaluation of both formats on nuScenes
        # refer to https://github.com/open-mmlab/mmdetection3d/issues/449
        if not ('pts_bbox' in results[0] or 'img_bbox' in results[0]):
            result_files = self._format_bbox(results, jsonfile_prefix)
        else:
            # should take the inner dict out of 'pts_bbox' or 'img_bbox' dict
            result_files = dict()
            for name in results[0]:
                print(f'\nFormating bboxes of {name}')
                results_ = [out[name] for out in results]
                tmp_file_ = osp.join(jsonfile_prefix, name)
                result_files.update(
                    {name: self._format_bbox(results_, tmp_file_)})
        return result_files, tmp_dir


    def evaluate(
        self,
        results,
        metric='bbox',
        logger=None,
        jsonfile_prefix=None,
        result_names=['pts_bbox'],
        show=False,
        out_dir=None,
        pipeline=None,
    ):
        """Evaluation in ZOD protocol.

        Args:
            results (list[dict]): Testing results of the dataset.
            metric (str | list[str], optional): Metrics to be evaluated.
                Default: 'bbox'.
            logger (logging.Logger | str, optional): Logger used for printing
                related information during evaluation. Default: None.
            jsonfile_prefix (str, optional): The prefix of json files including
                the file path and the prefix of filename, e.g., "a/b/prefix".
                If not specified, a temp file will be created. Default: None.
            show (bool, optional): Whether to visualize.
                Default: False.
            out_dir (str, optional): Path to save the visualization results.
                Default: None.
            pipeline (list[dict], optional): raw data loading for showing.
                Default: None.

        Returns:
            dict[str, float]: Results of each evaluation metric.
        """
        assert isinstance(results, list), 'results must be a list'
        assert len(results) == len(self), (
            'The length of results is not equal to the dataset len: {} != {}'.
            format(len(results), len(self)))

        # Sometimes the results contain this additional dict layer
        if 'pts_bbox' in results[0]:
            results = [res['pts_bbox'] for res in results]
        if 'img_bbox' in results[0]:
            raise NotImplementedError('img_bbox is not supported in ZOD.')

        det_boxes, gt_boxes = EvalBoxes(), EvalBoxes()
        for idx, (det, info) in enumerate(zip(results, self.data_infos)):
            frame_id = info['frame_id']
            det_boxes.add_boxes(frame_id, self._det_to_zod(det, frame_id))
            gt_boxes.add_boxes(frame_id, self._gt_to_zod(idx, frame_id))

        results_dict = zod_eval(gt_boxes, det_boxes, verify_coordinate_system=False)
        print_log(results_dict, logger=logger)

        if show or out_dir:
            self.show(results, out_dir, show=show, pipeline=pipeline)
        return results_dict

    def _det_to_zod(self, det: dict, frame_id: str) -> List[DetectionBox]:
        dets = []
        for box3d, label, score in zip(
                det['boxes_3d'].tensor.numpy(),
                det['labels_3d'].numpy(),
                det['scores_3d'].numpy(),
        ):
            dets.append(self._obj_to_zod(frame_id, box3d, label, score))
        return dets

    def _gt_to_zod(self, idx: int, frame_id: str) -> List[DetectionBox]:
        anno: dict = self.get_ann_info(idx)
        gts = []
        for box3d, label in zip(anno['gt_bboxes_3d'], anno['gt_labels_3d']):
            gts.append(self._obj_to_zod(frame_id, box3d, label))
        return gts

    def _obj_to_zod(self,
                    frame_id: str,
                    box3d: np.ndarray,
                    label: int,
                    score: float = -1.0) -> Optional[DetectionBox]:
        # object is in lidar frame - meaning that the rotation is around
        # the z-axis
        rot = pyquaternion.Quaternion(axis=(0, 0, 1), radians=box3d[6:])

        # ego translation is same as translation since world is ego-centered
        class_name = self.CLASSES[int(label)]
        if class_name not in EVALUATION_CLASSES:
            return None
        box = DetectionBox(
            sample_token=frame_id,
            translation=tuple(box3d[:3]),
            size=tuple(box3d[3:6]),
            rotation=tuple(rot.elements),
            #ego_translation=tuple(box3d[:3]),
            detection_name=class_name,
            detection_score=float(score),
        )
        return box

    # END ZOD evaluation

    def _build_default_pipeline(self):
        """Build the default pipeline for this dataset."""
        pipeline = [
            dict(
                type='LoadPointsFromFile',
                coord_type='LIDAR',
                load_dim=5,
                use_dim=5,
                file_client_args=dict(backend='disk'),
            ),
            dict(
                type='LoadPointsFromMultiSweeps',
                sweeps_num=10,
                file_client_args=dict(backend='disk'),
            ),
            dict(
                type='DefaultFormatBundle3D',
                class_names=self.CLASSES,
                with_label=False),
            dict(type='Collect3D', keys=['points']),
        ]
        return Compose(pipeline)

    def show(self, results, out_dir, show=False, pipeline=None):
        """Results visualization.

        Args:
            results (list[dict]): List of bounding boxes results.
            out_dir (str): Output directory of visualization result.
            show (bool): Whether to visualize the results online.
                Default: False.
            pipeline (list[dict], optional): raw data loading for showing.
                Default: None.
        """
        assert out_dir is not None, 'Expect out_dir, got none.'
        pipeline = self._get_pipeline(pipeline)
        for i, result in enumerate(results):
            if 'pts_bbox' in result.keys():
                result = result['pts_bbox']
            data_info = self.data_infos[i]
            pts_path = data_info['lidar_path']
            file_name = osp.split(pts_path)[-1].split('.')[0]
            points = self._extract_data(i, pipeline, 'points').numpy()
            # for now we convert points into depth mode
            points = Coord3DMode.convert_point(points, Coord3DMode.LIDAR,
                                               Coord3DMode.DEPTH)
            inds = result['scores_3d'] > 0.1
            gt_bboxes = self.get_ann_info(i)['gt_bboxes_3d'].tensor.numpy()
            show_gt_bboxes = Box3DMode.convert(gt_bboxes, Box3DMode.LIDAR,
                                               Box3DMode.DEPTH)
            pred_bboxes = result['boxes_3d'][inds].tensor.numpy()
            show_pred_bboxes = Box3DMode.convert(pred_bboxes, Box3DMode.LIDAR,
                                                 Box3DMode.DEPTH)
            show_result(points, show_gt_bboxes, show_pred_bboxes, out_dir,
                        file_name, show)
