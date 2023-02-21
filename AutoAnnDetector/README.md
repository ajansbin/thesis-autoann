This repo is for the detection part of the Auto-ann pipeline. <br>

Please stand inside AutoAnnDetector for running the detection-code. <br>

``` bash
cd AutoAnnDetector
```

To train a model please follow example below: <br>

``` bash
python tools/train.py config-file --out-dir path/to/outdir
```

For inference on a sequential-data with your model please run: <br>

``` bash
python tools/detect.py model-pth --out-dir path/to/outdir
```

