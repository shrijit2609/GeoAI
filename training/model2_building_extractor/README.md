# Model 2: Building Footprint Extraction

`train.py` reproduces the training configuration in `training/SpatialShiftAI.ipynb`, cell 82. It intentionally refuses to train unless CUDA is available. Run it in a Colab runtime configured for a Tesla T4 GPU.

```bash
!git clone -b main https://github.com/shrijit2609/GeoAI.git /content/SpatialShiftAI
%cd /content/SpatialShiftAI
!pip install torch torchvision opencv-python-headless rasterio scikit-learn pandas huggingface_hub
!python training/model2_building_extractor/train.py --download --project-root /content/SpatialShiftAI
```

The WHU aerial dataset is downloaded from the notebook's Hugging Face mirror `giswqs/WHU-Building-Dataset`. The script pairs image/mask files by normalized stem; uses explicit `train`, `val`/`validation`, and `test` directories when every sample has a split, otherwise it applies the notebook's deterministic two-step 70/15/15 source split (`random_state=42`). It will not continue with fewer than 100 matched pairs.

Notebook contract retained: RGB 256x256; inputs scaled to `[0,1]` with no mean/std normalization; nearest-neighbor binary masks; train-only horizontal and vertical flips at 0.5 plus 90-degree rotation at 0.5; torchvision DeepLabV3-ResNet50 with `DeepLabV3_ResNet50_Weights.DEFAULT`, one-channel classifier head; `0.5 * BCEWithLogits + 0.5 * Dice`; AdamW (`lr=1e-4`, `weight_decay=1e-4`); ReduceLROnPlateau (`mode=min`, `factor=.5`, `patience=2`); 12 epochs; best checkpoint chosen by minimum validation loss; test threshold 0.5.

Outputs are written to `models/model2_building_extractor/`: `building_deeplabv3_resnet50_best.pt`, `building_preprocessing.json`, `metrics.json`, `training_history.json`, and `model_card.md`. The checkpoint is directly compatible with the backend adapter. Do not copy dataset files into Git.

After training, verify locally or in Colab with:

```bash
python training/model2_building_extractor/inference.py path/to/real/image.png --model-root models --mask-output /tmp/buildings.png
```

Metrics describe the WHU held-out split only, not deployment accuracy in India.
