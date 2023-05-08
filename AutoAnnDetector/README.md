# AutoAnnDetector

This repo is for the detection part of the Auto-ann pipeline. <br>

Please stand inside AutoAnnDetector for running the detection-code. <br>

## Setup

``` bash
cd AutoAnnDetector/mmdetection3d
```

Before training a model it can be good to install packages using the containing docker file. Please run: <br>

``` bash
docker build -f docker/Dockerfile -t mmdetection3d .
```

## ZOD Frames Train

To train a model on zod-frames please follow example below: <br>

``` bash
python tools/train.py configs/centerpoint_zod.py
```

## ZOD Sequence Inference

Before running inference you need to preprocess the annotations. Please run the following script: <br>

``` bash
python tools/create_data.py zod \
	--root-path path/to/zod \
	--out-dir out/dir/path \
	--version full \
	--extra-tag zod-seq-full \
	--sequences
```

For inference on a sequential-data with your model please run: <br>

``` bash
python tools/test.py configs/centerpoint_zod_sequence.py \
	path/to/your/trained/model.pth \
    --format-only \
    --options "jsonfile_prefix=path/to/store/json/file"
```

