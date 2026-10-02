# Model 3: Temporal Change Detection

`train.py` follows `training/SpatialShiftAI.ipynb`, cell 102, and refuses to run without CUDA. Use a Colab Tesla T4 runtime.

```bash
!git clone -b main https://github.com/shrijit2609/GeoAI.git /content/SpatialShiftAI
%cd /content/SpatialShiftAI
!pip install torch torchvision pyarrow pandas pillow scikit-learn huggingface_hub
!python training/model3_change_detection/train.py --download --project-root /content/SpatialShiftAI
```

The notebook dataset is the Hugging Face Parquet conversion `ericyu/LEVIRCD_Cropped_256`. The script uses its train / val (or validation) / test Parquet splits, detects the same pair/mask column aliases as cell 102, and caches at most 5,000 train, 1,000 validation, and 1,000 test rows in runtime memory. LEVIR imagery is 256x256 RGB; inputs are divided by 255 then ImageNet normalized (`mean=[.485,.456,.406]`, `std=[.229,.224,.225]`). Augmentation applies synchronized horizontal flip, vertical flip, and random 0-3 quarter-turn rotation, each at probability .5.

The network is the notebook's pretrained torchvision ResNet18 shared encoder, absolute feature difference, 512→256 3x3 decoder convolution, then five stride-2 transposed-convolution blocks (256→128→64→32→16→8) and one output channel, interpolated to input dimensions. Loss is `.5 * BCEWithLogits + .5 * Dice`; optimizer AdamW (`lr=1e-4`, `weight_decay=1e-4`), with gradient norm clipped to `1.0`; ReduceLROnPlateau (`mode=max`, `factor=.5`, `patience=1`); five epochs; select the highest validation IoU; test threshold `.5`.

Outputs are saved directly to `models/model3_change_detection/`: best and final checkpoints, `change_preprocessing.json`, `metrics.json`, `training_history.json`, and `model_card.md`. The best checkpoint matches the backend Model 3 adapter architecture. Dataset/parquet files are not copied into Git.

After training, run the production adapter:

```bash
python training/model3_change_detection/inference.py before.png after.png --model-root models --mask-output /tmp/change.png
```

Benchmark scores describe LEVIR-CD only; this dataset does not establish geographic CRS or Indian deployment accuracy.
