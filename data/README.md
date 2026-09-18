# Dataset & Local Data Directory

This directory outlines the expected data layout and stores local caching artifacts for the AI Document Verification System.

## Dataset Location & Setup

The system is configured by default to read the **Find it Again! (findit2)** receipt forgery benchmark from `archive/findit2/` or any custom path defined by the `DATASET_ROOT` environment variable in your `.env` file.

### Expected Directory Layout

```
archive/findit2/  (or custom $DATASET_ROOT)
├── train/
│   ├── <filename>.png       (Raw document image)
│   ├── <filename>.txt       (Ground-truth text / bounding boxes - FOR ANALYSIS ONLY)
│   └── ...
├── val/
│   ├── <filename>.png
│   ├── <filename>.txt
│   └── ...
└── test/
    ├── <filename>.png
    ├── <filename>.txt
    └── ...
```

### Configuring a Custom Path

To point the pipeline to a different directory or drive location:
1. Copy `.env.example` to `.env`.
2. Update `DATASET_ROOT`:
   ```bash
   DATASET_ROOT=/path/to/your/findit2
   ```

### Dataset Terms & Licensing Notice

Due to dataset distribution licensing, raw receipt images and ground-truth text annotations are not bundled inside the Git repository. Users must obtain the **Find it Again!** benchmark dataset independently from its official source and place it in the configured `DATASET_ROOT` directory.

### Local Cache

- `data/cache/`: Runtime feature matrices (`features_train.npz`, `features_val.npz`, `features_test.npz`) and serialized OCR cache (`ocr_cache.json`) are stored here for rapid development cycles. They are automatically ignored by Git.
