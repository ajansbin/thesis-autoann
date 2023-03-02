This repo is for the smoothing part of the automatic annotation pipeline. <br>

Please stand inside AutoAnnSmoother and install the project before running the smoothing-code. <br>

``` bash
cd AutoAnnSmoother
pip install -e .
```

To train a smoother please follow example below: <br>

``` bash
python tools/train.py --config path/to/config.yaml --data-path path/to/data --save-dir path/to/save/dir --result-path path/to/tracking/results.json
```

Not implemented yet: <br>
For inference on detections with track-ids with your model please run: <br>

``` bash
python tools/infer.py model-pth --assoc-dets path/to/dets-with-trackids --out-dir path/to/outdir
```

