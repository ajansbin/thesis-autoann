This repo is for the smoothing part of the automatic annotation pipeline. <br>

Please stand inside AutoAnnSmoother for running the detection-code. <br>

``` bash
cd AutoAnnSmoother
```

To train a smoother please follow example below: <br>

``` bash
python tools/train.py config-file
```

For inference on detections with track-ids with your model please run: <br>

``` bash
python tools/infer.py model-pth --assoc-dets path/to/dets-with-trackids --out-dir path/to/outdir
```

