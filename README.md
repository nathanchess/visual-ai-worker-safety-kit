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

This project demonstrates an end-to-end workflow for building high-quality training sets from raw surveillance footage **without manual video scrubbing**. It serves as the primary enablement asset for the **Visual AI Hackathon** Worker Safety Challenge.

<p align="center">
  <img src="assets/preview.png" alt="FiftyOne Preview" width="800">
</p>

## Strategic Goal

**"Small Data ≠ Manual Data"** — Modern semantic search can replace 40+ hours of manual video annotation.

By combining TwelveLabs' video understanding with FiftyOne's data management, you can:
- Auto-generate embeddings that capture semantic content
- Cluster similar videos without manual labeling
- Generate zero-shot labels using video-to-text models
- Export directly to PyTorch for training

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│              FiftyOne Dataset (from Hugging Face)                │
│                  Safe & Unsafe Behaviours Dataset                │
└────────────────────────────┬─────────────────────────────────────┘
                             │
                             ▼
┌──────────────────────────────────────────────────────────────────┐
│              TwelveLabs Video Understanding Platform             │
│  ┌─────────────────────┐    ┌─────────────────────────────────┐  │
│  │   Marengo 3.0       │    │   Pegasus 1.2                   │  │
│  │   512-dim Embeddings│    │   Zero-shot Auto-labeling       │  │
│  └─────────────────────┘    └─────────────────────────────────┘  │
└────────────────────────────┬─────────────────────────────────────┘
                             │
                             ▼
┌──────────────────────────────────────────────────────────────────┐
│                     Semantic Clustering                          │
│                  (KMeans on Embeddings)                          │
└────────────────────────────┬─────────────────────────────────────┘
                             │
                             ▼
┌──────────────────────────────────────────────────────────────────┐
│                   FiftyOne Visualization                         │
│         (Interactive UMAP + Dataset Exploration)                 │
└────────────────────────────┬─────────────────────────────────────┘
                             │
                             ▼
┌──────────────────────────────────────────────────────────────────┐
│              PyTorch DataLoader via to_torch()                   │
│          (Ready for classifier fine-tuning)                      │
└──────────────────────────────────────────────────────────────────┘
```

## Quick Start

### Prerequisites

- Python 3.9+
- [TwelveLabs API Key](https://playground.twelvelabs.io/dashboard/api-keys)

### Installation

```bash
# Clone the repository
git clone https://github.com/nathanchess/visual-ai-worker-safety-kit
cd visual-ai-worker-safety-kit

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install fiftyone twelvelabs python-dotenv torch torchvision scikit-learn
```

### Configuration

Create a `.env` file in the project root:

```env
# Required
TL_API_KEY=your_twelvelabs_api_key

# Optional (with defaults)
DATASET_NAME=safe_unsafe_behaviours
DATASET_SPLIT=train
VIDEOS_PER_LABEL=3
MIN_DURATION=4.0
NUM_CLUSTERS=8
TL_INDEX_NAME=fiftyone-twelvelabs-index
```

### Running

```bash
# Full workflow: ingest, embed, cluster, label, visualize
python main.py

# Skip video ingestion (use already indexed videos)
python main.py --skip-ingestion

# Skip clustering (just fetch embeddings)
python main.py --skip-ingestion --skip-clustering

# Export DataLoader only (no visualization)
python main.py --skip-ingestion --export-only
```

Or use the Jupyter notebook for an interactive walkthrough:

```bash
jupyter notebook main.ipynb
```

## Dataset

The [Safe and Unsafe Behaviours Dataset](https://huggingface.co/datasets/Voxel51/Safe_and_Unsafe_Behaviours) is automatically downloaded from Hugging Face. It contains 8 classes:

| Label | Description |
|-------|-------------|
| Safe Walkway Violation | Worker NOT using designated safe walkway |
| Unauthorized Intervention | Accessing equipment without authorization |
| Opened Panel Cover | Electrical/machinery panel left open |
| Carrying Overload with Forklift | Forklift with excessive/unstable load |
| Safe Walkway | Worker correctly using safe walkway |
| Authorized Intervention | Proper procedures for equipment access |
| Closed Panel Cover | Panels properly secured |
| Safe Carrying | Forklift with appropriate load |

## Key FiftyOne Patterns

This project demonstrates idiomatic FiftyOne patterns for efficient data workflows:

### View Operations for Filtering

```python
from fiftyone import ViewField as F

# Chain filters declaratively
view = (
    dataset
    .match_tags("train")
    .match(F("metadata.duration") >= 4.0)
    .match(~F("tl_video_id").exists())
)
```

### Efficient Iteration with Autosave

```python
# select_fields loads only what you need
# autosave=True batches database writes
for sample in view.select_fields("tl_video_id").iter_samples(
    progress=True, autosave=True
):
    sample["tl_embedding"] = fetch_embedding(sample.tl_video_id)
```

### Batch Operations with values/set_values

```python
# Extract field values as a list
video_ids = indexed_view.values("tl_video_id")

# Batch update all samples at once
indexed_view.set_values("cluster", classifications)
```

### PyTorch Export with GetItem

```python
from fiftyone.utils.torch import GetItem

class WorkerSafetyGetItem(GetItem):
    @property
    def required_keys(self):
        return ["tl_embedding", "ground_truth"]
    
    def __call__(self, d):
        return {
            "embedding": torch.tensor(d.get("tl_embedding")),
            "label_idx": self.label_to_idx[d.get("ground_truth").label],
        }

# Create PyTorch dataset from any FiftyOne view
torch_dataset = indexed_view.to_torch(WorkerSafetyGetItem(LABEL_TO_IDX))
```

## Workflow Steps

| Step | Tool | Description |
|------|------|-------------|
| **1. Ingest** | FiftyOne | Load dataset from Hugging Face with metadata |
| **2. Index** | TwelveLabs | Upload videos, generate Marengo 3.0 embeddings |
| **3. Retrieve** | TwelveLabs | Fetch 512-dim embeddings to FiftyOne samples |
| **4. Cluster** | scikit-learn | Group similar videos with KMeans |
| **5. Auto-label** | TwelveLabs Pegasus | Generate semantic labels per cluster |
| **6. Visualize** | FiftyOne | Explore with UMAP + interactive app |
| **7. Export** | FiftyOne `to_torch()` | Create PyTorch DataLoader |

## Output

The workflow creates a PyTorch `DataLoader` ready for training:

```python
for batch in train_loader:
    embeddings = batch["embedding"]  # [batch_size, 512]
    labels = batch["label_idx"]      # [batch_size]
    
    # Your training code here
    outputs = model(embeddings)
    loss = criterion(outputs, labels)
```

## CLI Options

| Option | Description |
|--------|-------------|
| `--skip-ingestion` | Skip video upload (use existing TwelveLabs index) |
| `--skip-clustering` | Skip clustering and auto-labeling |
| `--export-only` | Skip FiftyOne visualization |
| `--batch-size N` | Batch size for DataLoader (default: 4) |

## Challenge Context

- **Event**: Visual AI Hackathon
- **Track**: Worker Safety Challenge
- **Objective**: Build efficient video classifiers from small, curated datasets

## Resources

### Documentation
- [TwelveLabs Docs](https://docs.twelvelabs.io/)
- [FiftyOne Docs](https://docs.voxel51.com/)
- [FiftyOne Views Cheat Sheet](https://docs.voxel51.com/cheat_sheets/views_cheat_sheet.html)
- [FiftyOne Filtering Cheat Sheet](https://docs.voxel51.com/cheat_sheets/filtering_cheat_sheet.html)

### FiftyOne + TwelveLabs Plugin

For enhanced semantic video search directly within FiftyOne:

```bash
fiftyone plugins download https://github.com/danielgural/semantic_video_search
```

[View Plugin Repository →](https://github.com/danielgural/semantic_video_search)

## License

MIT License — See [LICENSE](LICENSE) for details.
