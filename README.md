# medseg3d

3D medical image segmentation with a 3D U-Net (PyTorch + MONAI).

Reads NIfTI/DICOM volumes, pre-processes them, trains a 3D U-Net, evaluates with Dice and Hausdorff distance (written from scratch), runs sliding-window inference, and saves masks plus overlay images. Training is done on Kaggle; inference is light enough to run on a normal laptop.

This is a research/learning project, not a medical device.

## Pipeline

NIfTI/DICOM -> pre-processing (MONAI) -> 3D U-Net training -> evaluation (Dice, IoU, HD, HD95) -> sliding-window inference -> mask + overlay output -> ONNX export

## Setup

```bash
python -m venv .venv
source .venv/bin/activate          #Windows: .\.venv\Scripts\Activate.ps1
pip install -r requirements-train.txt     #training
pip install -r requirements-infer.txt     #inference only (light)
```

## Usage

```bash
python scripts/train.py --config config/config.yaml
python scripts/validate.py --config config/config.yaml --ckpt outputs/checkpoints/best.pt
python scripts/predict.py --input scan.nii.gz --output mask.nii.gz --ckpt outputs/onnx/model.onnx
python scripts/export_onnx.py --ckpt outputs/checkpoints/best.pt
pytest tests/
```

## Structure

```
medseg3d/
├── config/config.yaml
├── data/           volume_io.py, transforms.py, datamodule.py
├── models/         unet3d.py, vnet.py, factory.py
├── losses/         dice_ce.py
├── evaluation/     metrics.py, report.py
├── scripts/        train, validate, infer, predict, export_onnx
├── visualization/  overlay.py
├── app/            gradio_app.py
├── notebooks/      eda, kaggle_train
├── tests/
└── docs/
```

## Training on Kaggle

Turn on the GPU, add the dataset, clone the repo, install `requirements-train.txt`, set `data_root` in the config to the `/kaggle/input/...` path, and run `scripts/train.py`. Download `best.pt` when it finishes.

## Status

Done: metrics, 3D U-Net, loss, overlay, NIfTI I/O.
To do: transforms, data module, train/validate/infer scripts, ONNX export, demo app.

## Author

Bikram Chapagain