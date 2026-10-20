# Dataset setup and offline evaluation

The application runs without any dataset. These optional runners (`app/evaluation/`, CLI `scripts/evaluate_dataset.py`) let you evaluate parts of the system on public data **you download yourself** (check each dataset's license). Nothing is downloaded at startup or by the CLI.

**Implemented vs. planned.** Implemented: the three runners below. Planned / not done: threat-score evaluation (needs labelled scenarios), mAP-style detection metrics, official UCF-Crime frame-level AUC. **No results are bundled and none are claimed.** Because the YOLO model is COCO-pretrained and never trained on these datasets, there is no train/test leakage from the detector; the datasets' official splits are still respected (MOT17 *train* has ground truth; the public MOT17 *test* split does not, so it cannot be scored locally).

Place data under `data/datasets/` (git-ignored) or pass any path with `--root`.

## MOT17 — tracking (identity consistency, trajectory continuity)

Source: <https://motchallenge.net/data/MOT17/>

```
<root>/MOT17/train/MOT17-02-FRCNN/
    img1/000001.jpg ...
    gt/gt.txt            frame,id,x,y,w,h,flag,class,visibility
    seqinfo.ini
```

Only one detector variant per sequence (`*-FRCNN`) is used, since images and ground truth are identical across DPM/FRCNN/SDP. Scored entries: `flag != 0` and class 1 (pedestrian).

```bash
pip install motmetrics            # already in requirements.txt
python scripts/evaluate_dataset.py mot17 --root data/datasets --sequences 02 04 --max-frames 300
```

Output: MOTChallenge-format result files per sequence, `mot17_summary.txt` (MOTA, IDF1, ID switches, fragmentations, mostly tracked/lost, precision/recall via `py-motmetrics`) in `outputs/evaluation/`. The detector runs with the application's configured confidence; the result is for a generic pretrained model, not a MOT17-tuned one.

## VisDrone2019-DET — detection in varied/crowded scenes

Source: <https://github.com/VisDrone/VisDrone-Dataset>

```
<root>/VisDrone2019-DET-val/images/*.jpg
<root>/VisDrone2019-DET-val/annotations/*.txt   x,y,w,h,score,category,truncation,occlusion
```

Categories are mapped to COCO classes (pedestrian/people→person, bicycle, car/van→car, truck, bus, motor→motorcycle). Ignored regions and categories without a COCO equivalent (tricycles, "others") are excused, not counted as false positives.

```bash
python scripts/evaluate_dataset.py visdrone --root data/datasets --split val --max-images 200
```

Output: precision / recall / F1 at IoU 0.5 overall and per class, at the configured detection confidence. This is not COCO mAP. (The VisDrone-MOT tracking subset is not implemented.)

## UCF-Crime — video-level anomaly analysis

Source: <https://www.crcv.ucf.edu/projects/real-world/>

```
<root>/Anomaly_Detection_splits/Anomaly_Test.txt     one relative video path per line
<root>/Abuse/Abuse001_x264.mp4 ...
<root>/Testing_Normal_Videos_Anomaly/Normal_Videos_001_x264.mp4 ...
```

Labels are video-level only (a path containing "normal" is label 0, otherwise 1). Train and test videos come from the official split files and are never mixed.

```bash
python scripts/evaluate_dataset.py ucf-crime --root data/datasets/UCF-Crime --max-videos 30 --frame-stride 5
```

Output: `ucf_crime_videos.csv` (max score, alert counts per video) and a **descriptive** AUC of "max threat score vs. anomalous/normal". The threat score is zone-based and not trained for anomaly detection, and the default zone is a demo polygon, so treat the number as a diagnostic, not evidence of anomaly-detection performance.
