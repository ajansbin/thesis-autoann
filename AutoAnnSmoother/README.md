# AutoAnnSmoother

This project is for the smoothing part of the automatic annotation pipeline. Please make sure you have gone through the previous steps in the pipeline before running the smoother. That is, you should have access to a tracking results file as a json. <br>

## Setup

The recommended way of using the code is to run it using the belonging [dockerfile](docker/Dockerfile). <br>
Please follow these steps to install and use it properly: <br>

Build docker: <br>
``` bash
docker build -f docker/Dockerfile -t smoother .
```
Run docker: <br>

``` bash
docker run -v root-dir/thesis-autoann/AutoAnnSmoother:/AutoAnnSmoother \
	-v /zod-data-path:/AutoAnnSmoother/datasets/zod \
	-v /out-dir-path:/AutoAnnSmoother/storage \
	-v /preprocessed-point-clouds-root-path:/AutoAnnSmoother/preprocessed \
	-it smoother
```

## Preprocess

Before training, you need to preprocess the point-clouds. Please run the following script inside your docker: <br>
``` bash
python smoother/preprocessing/zod_pc_preprocessing.py --data-path /path-to-zod \
	--version full \
	--split train \
	--out-dir /out-dir-path \
	--tracking-result-path /tracking-results-json-path 
```

## Training

For training, please make use of the [config-file](configs/training_config.yaml) and change it to your needs. Then run the following: <br>

``` bash
python tools/train.py --config configs/training_config.yaml \
	--data-path /path-to-zod \
	--pc-name preprocessed_full_train \
	--save-dir /out-dir-path \
	--result-path /tracking-results-json-path  \
	--name run-name
```


## Inference / Testing

For Inference, please use config file provided by the trained model. Then run the following:

``` bash
python tools/infer.py --config /configs/model_conf.json  \
    --data-path /path-to-zod \
    --version full \
    --split val \
    --save-path /out-dir-path \
    --result-path /tracking-results-json-path  \
    --model-path /model.pth
```
