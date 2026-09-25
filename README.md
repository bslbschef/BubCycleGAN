# BubCycleGAN

PyTorch code for bubble image-to-mask translation.

## Installation

Install PyTorch and torchvision for your system, then run:

```bash
pip install -r requirements.txt
```

## Data

- Training images: `data/images/`
- Training masks: `data/masks/`
- Test images: `test/test_images/`
- Test masks: `test/test_masks/`

## Training

```bash
python main.py --training --testing
```

## Testing

Place the trained checkpoints in `saved_models/multiple_bubble_s3_project/`, then run:

```bash
python main.py
```

Results are saved in `test/test_results/`. 