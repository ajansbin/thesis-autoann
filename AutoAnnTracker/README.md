This repo is for the tracking part of the auotomatic annotation pipeline.

Please go to AutoAnnTracker: <br>
``` bash
cd AutoAnnTracker/SimpleTrack
```

Existing trackers require the detections to be in world coordinate system. To convert results from lidar to world coordinates run: <br>
``` bash
python convert_results.py \
	--result-path <detections_lidar_coord> \
	--save-path <detections_world_coord> \
	--version <version> \
	--split <split> \
	--convert-to world \
	--type detections	
```

Please stand inside the specific tracker under AutoAnnTracker for running the tracking. <br>

**SimpleTrack** <br>

``` bash
cd AutoAnnTracker/SimpleTrack
```

To run preprocessing of SimpleTrack run:
``` bash
python preprocessing/zod_data/zod_preprocess.sh <datapath> <preprocessed_folder> <version> <split>
```

``` bash
python preprocessing/zod_data/detection.py \
	--raw_data_folder <datapath> \
	--data_folder <preprocessed_folder> \
	--det_name detections \
	--file_path <detections_world_coord> \
	--version <version> \
	--split <split>
```

To run tracking:
``` bash
python python tools/main_zod.py \
    --name <tracking_name> \
    --det_name detections \
    --config_path configs/zod_configs/giou.yaml \
    --result_folder <save_folder> \
    --data_folder <preprocessed_folder> \
    --process 8
```

Format results:
``` bash
python tools/zod_result_creation.py \
    --name <tracking_name> \
    --result_folder <save_folder> \
    --data_folder <preprocessed_folder>
```

Merge all classes to one result file:
``` bash
python tools/zod_type_merge.py \
    --name <tracking_name> \
    --result_folder <save_folder>
```


To format back results to lidar coordinate system run:
``` bash
cd AutoAnnTracker
```

``` bash
python tools/convert_results.py \
	--result-path <detections_lidar_coord> \
	--save-path <detections_world_coord> \
	--version <version> \
	--split <split> \
	--convert-to world \
	--type detections

python convert_results.py \
	--result-path <detections_lidar_coord> \
	--save-path <detections_world_coord> \
	--version <version> \
	--split <split> \
	--convert-to lidar \
	--type tracks	
```