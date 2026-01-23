<p align="center">
  <a href="https://twelvelabs.io">
    <img src="assets/tl_logo_black.png" alt="TwelveLabs" height="60">
  </a>
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
  <a href="https://voxel51.com">
    <img src="assets/logo_voxel51.png" alt="Voxel51" height="60">
  </a>
</p>

# Visual AI Hackathon 2026 Worker Safety Challenge Enablement Kit

> **Semantic Dataset Curator & Visualizer using TwelveLabs + FiftyOne**

This project demonstrates an end-to-end workflow for building high-quality training sets from raw surveillance footage **without manual video scrubbing**. It serves as the primary enablement asset for the **2026 Visual AI Hackathon @ Northeastern** Worker Safety Challenge.

<p align="center">
  <img src="assets/preview.png" alt="FiftyOne Preview" width="800">
</p>

## 🎯 Strategic Goal

**"Small Data ≠ Manual Data"** — Modern semantic search can replace 40+ hours of manual video annotation.

## 🏗️ Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                    Video Dataset (Raw Footage)                    │
└────────────────────────────┬─────────────────────────────────────┘
                             │
                             ▼
┌──────────────────────────────────────────────────────────────────┐
│              TwelveLabs Video Understanding Platform              │
│  ┌─────────────────────┐    ┌─────────────────────────────────┐  │
│  │   Marengo 3.0       │    │   Pegasus 1.2                   │  │
│  │   (Embeddings)      │    │   (Zero-shot Auto-labeling)     │  │
│  └─────────────────────┘    └─────────────────────────────────┘  │
└────────────────────────────┬─────────────────────────────────────┘
                             │
                             ▼
┌──────────────────────────────────────────────────────────────────┐
│                     Semantic Clustering                           │
│                  (KMeans on Embeddings)                           │
└────────────────────────────┬─────────────────────────────────────┘
                             │
                             ▼
┌──────────────────────────────────────────────────────────────────┐
│                   FiftyOne Visualization                          │
│         (Interactive UMAP + Dataset Exploration)                  │
└────────────────────────────┬─────────────────────────────────────┘
                             │
                             ▼
┌──────────────────────────────────────────────────────────────────┐
│               PyTorch Dataset Export (.pt)                        │
│          (Ready for classifier fine-tuning)                       │
└──────────────────────────────────────────────────────────────────┘
```

## 🚀 Quick Start

### Prerequisites

- Python 3.9+
- [TwelveLabs API Key](https://api.twelvelabs.io/)
- Video dataset organized by labels

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd voxel51

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install fiftyone twelvelabs python-dotenv torch torchvision opencv-python scikit-learn
```

### Configuration

Create a `.env` file in the project root:

```env
# Required
DATASET_PATH=/path/to/your/dataset
TL_API_KEY=your_twelvelabs_api_key

# Optional (with defaults)
DATASET_NAME=workplace_surveillance_videos
DATASET_SPLIT=train
DATASET_VIDEOS_PER_LABEL=3
TL_INDEX_NAME=fiftyone-twelvelabs-index
```

### Dataset Structure

```
dataset/
├── train/
│   ├── 0_safe_walkway_violation/
│   │   ├── video1.mp4
│   │   └── video2.mp4
│   ├── 1_unauthorized_intervention/
│   │   └── ...
│   └── ...
└── test/
    └── ...
```

### Running the Script

```bash
# Full workflow: ingest, cluster, label, visualize
python main.py

# Skip video ingestion (use already indexed videos)
python main.py --skip-ingestion

# Export dataset only (no visualization)
python main.py --skip-ingestion --export-only

# Custom number of clusters
python main.py --num-clusters 10

# Custom output path
python main.py --output my_dataset.pt
```

## 📋 Workflow Steps

1. **Video Ingestion** — Upload videos to TwelveLabs (filters out videos < 4s)
2. **Embedding Generation** — Extract visual embeddings via Marengo 3.0
3. **Semantic Clustering** — Group similar videos using KMeans
4. **Auto-labeling** — Generate descriptive labels via Pegasus 1.2
5. **Visualization** — Explore clusters in FiftyOne's interactive UI
6. **Export** — Save as PyTorch dataset for downstream training

## 📦 Output

The script exports a `worker_safety_dataset.pt` file containing:

| Field        | Type          | Description                          |
|--------------|---------------|--------------------------------------|
| `embedding`  | Tensor        | 1024-dim visual embedding            |
| `label_idx`  | Tensor        | Integer cluster ID                   |
| `label_str`  | str           | Semantic label (e.g., "forklift_operation_safety") |
| `video_id`   | str           | TwelveLabs video reference           |

### Loading the Dataset

```python
import torch

dataset = torch.load("worker_safety_dataset.pt")
print(f"Dataset size: {len(dataset)}")

# Access a sample
sample = dataset[0]
print(f"Embedding shape: {sample['embedding'].shape}")
print(f"Label: {sample['label_str']}")
```

## 🔧 CLI Options

| Option              | Description                                      |
|---------------------|--------------------------------------------------|
| `--skip-ingestion`  | Skip video upload (use existing TwelveLabs index)|
| `--export-only`     | Skip FiftyOne visualization                      |
| `--num-clusters N`  | Number of KMeans clusters (default: 8)           |
| `--output PATH`     | Output path for .pt file                         |

## 🏆 Challenge Context

- **Event**: Visual AI Hackathon @ Northeastern
- **Track**: Worker Safety Challenge
- **Objective**: Build efficient video classifiers from small, curated datasets

## 🔌 FiftyOne + Twelve Labs Plugin

For enhanced semantic video search capabilities directly within FiftyOne, check out the official **FiftyOne + Twelve Labs Plugin**:

```bash
fiftyone plugins download https://github.com/danielgural/semantic_video_search
```

**Key Features:**
- 🧠 Generate multimodal embeddings (visual, audio, OCR) from full videos
- 🔄 Automatically split videos into meaningful clips
- 🔍 Run semantic search over indexed videos using natural language prompts
- 📦 Store results in clip-level FiftyOne datasets

[View Plugin Repository →](https://github.com/danielgural/semantic_video_search)

## 📚 Resources

- [TwelveLabs Documentation](https://docs.twelvelabs.io/)
- [FiftyOne Documentation](https://docs.voxel51.com/)

## 📄 License

MIT License — See [LICENSE](LICENSE) for details.
