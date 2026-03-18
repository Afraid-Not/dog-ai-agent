# Dog AI Agent

Dog breed identification and health analysis system using deep learning.

## Project Structure

```
├── ai-service/          # ML model training and embedding pipeline
│   ├── train_embedding.py         # Training script (Triplet Loss, 100 epochs)
│   ├── train_embedding_model.py   # Model definition (MobileNetV2 backbone)
│   └── trained_models/            # Saved model weights (.h5, .pth)
├── backend/             # Backend service (stub)
├── frontend/            # Frontend (stub)
├── data/                # Data pipeline and datasets
│   ├── Annotation/      # Stanford Dogs annotations (120 breeds)
│   ├── Images/          # Stanford Dogs images
│   ├── crawl_cidd*.py   # CIDD website crawlers
│   ├── extract_disorders.py    # Genetic disorder extraction
│   ├── match_breeds.py         # Stanford Dogs ↔ CIDD breed matching
│   ├── breed_match_fixing.py   # Improved matching with similarity scoring
│   ├── cidd_breeds.json        # 160 CIDD breeds
│   ├── cidd_breed_disorders.json  # Genetic disorders per breed
│   └── breed_match_fixed.csv     # Final breed mapping
└── deprecated/          # Deprecated code (empty)
```

## Tech Stack

- **Language:** Python
- **ML Framework:** PyTorch (torchvision, MobileNetV2)
- **Training:** Batch Hard Triplet Loss, Online Hard Mining
- **Model:** MobileNetV2 → 120 breed classifier → 128-dim embedding (L2 normalized)
- **Data:** Stanford Dogs (120 breeds), CIDD (160 breeds with genetic disorders)
- **Scraping:** requests, BeautifulSoup4
- **Other:** TensorFlow Datasets, PIL, NumPy, H5py

## Pipeline

1. **Data Collection:** Crawl CIDD for breed health data → `cidd_breeds.json`
2. **Disorder Extraction:** Parse genetic disorders → `cidd_breed_disorders.json`
3. **Breed Matching:** Map Stanford Dogs breeds to CIDD breeds → `breed_match_fixed.csv`
4. **Model Training:** Train embedding model with Triplet Loss on Stanford Dogs images
5. **Inference:** Image → Breed ID + Health insights from CIDD data

## Commands

```bash
# Install data dependencies
pip install -r data/requirements.txt

# Data pipeline
python data/crawl_cidd_breeds.py
python data/extract_disorders.py
python data/match_breeds.py
python data/breed_match_fixing.py

# Train embedding model
python ai-service/train_embedding.py
```

## Model Architecture

- **Backbone:** MobileNetV2 (ImageNet pre-trained, frozen)
- **Classifier Head:** 1280 → 120 classes (loaded from Keras model_1.h5, frozen)
- **Embedding Layers:** 120 → 256 → 128 (trainable, L2 normalized)
- **Training:** P=8 classes × K=4 samples per batch, Recall@1/Recall@5 evaluation

## Git

- **Main branch:** `main`
- Large model files tracked via Git LFS (`.gitattributes`)
