This repo is for the tracking part of the auotomatic annotation pipeline


Please stand inside AutoAnnTracker for running the detection-code. <br>

``` bash
cd AutoAnnTracker
```

Before tracking please choose which tracker to use and prepare data accordingly: <br>
``` bash
python tools/prepare_data.py --tracker SimpleTrack --data zod --data-root path/to/zod --out-dir path/to/outdir
```

To track detections please run the following<br>

``` bash
python tools/track.py dets_json --config config-file --out-dir path/to/tracking/output
```