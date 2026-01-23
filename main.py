#!/usr/bin/env python3
"""
Visual AI Hackathon Worker Safety Challenge Enablement Kit
=================================================

This script demonstrates an end-to-end workflow between TwelveLabs and FiftyOne
for semantic dataset curation and visualization in the workplace safety domain.

Key Features:
- Video ingestion and indexing via TwelveLabs Marengo 3.0
- Semantic clustering using video embeddings
- Auto-labeling with TwelveLabs Pegasus 1.2
- Interactive visualization with FiftyOne

Usage:
    1. Set environment variables in .env file
    2. Run: python main.py
"""

import os
import json
import argparse

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.cluster import KMeans
from dotenv import load_dotenv

import fiftyone as fo
import fiftyone.brain as fob
from fiftyone.core.labels import Classification

from twelvelabs import TwelveLabs
from twelvelabs.indexes import IndexesCreateRequestModelsItem


def load_config():
    """Load configuration from environment variables."""
    load_dotenv()

    config = {
        "dataset_path": os.getenv("DATASET_PATH"),
        "dataset_name": os.getenv("DATASET_NAME", "workplace_surveillance_videos"),
        "dataset_split": os.getenv("DATASET_SPLIT", "train"),
        "videos_per_label": int(os.getenv("DATASET_VIDEOS_PER_LABEL", "3")),
        "tl_index_name": os.getenv("TL_INDEX_NAME", "fiftyone-twelvelabs-index"),
        "tl_api_key": os.getenv("TL_API_KEY"),
    }

    if not config["dataset_path"]:
        raise ValueError("DATASET_PATH must be set in the .env file")
    if not config["tl_api_key"]:
        raise ValueError("TL_API_KEY must be set in the .env file")

    return config


def get_or_create_twelvelabs_index(client: TwelveLabs, index_name: str) -> str:
    """
    Get existing TwelveLabs index or create a new one.

    Args:
        client: TwelveLabs client instance
        index_name: Name of the index to retrieve or create

    Returns:
        Index ID
    """
    indexes = client.indexes.list()
    for index in indexes:
        if index.index_name == index_name:
            print(f"Found index '{index_name}' with ID {index.id}")
            return index.id

    # Create new index with Marengo 3.0 and Pegasus 1.2
    index = client.indexes.create(
        index_name=index_name,
        models=[
            IndexesCreateRequestModelsItem(
                model_name="marengo3.0", model_options=["visual", "audio"]
            ),
            IndexesCreateRequestModelsItem(
                model_name="pegasus1.2", model_options=["visual", "audio"]
            ),
        ],
    )
    print(f"Created index '{index_name}' with ID {index.id}")
    return index.id


def get_video_duration(video_path: str) -> float:
    """Get video duration in seconds using OpenCV."""
    try:
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        cap.release()
        return frame_count / fps if fps > 0 else 0
    except Exception:
        return 0


def ingest_videos(
    client: TwelveLabs,
    index_id: str,
    dataset_path: str,
    dataset_split: str,
    videos_per_label: int,
) -> dict:
    """
    Ingest videos from the dataset into TwelveLabs index.

    Args:
        client: TwelveLabs client instance
        index_id: TwelveLabs index ID
        dataset_path: Path to the dataset directory
        dataset_split: Dataset split to process (e.g., "train")
        videos_per_label: Maximum number of videos to process per label

    Returns:
        Dictionary mapping video filenames to TwelveLabs video IDs
    """
    print(f"Loading videos from {dataset_path} with '{dataset_split}' split")
    print(f"Processing up to {videos_per_label} videos per label")

    video_ids = {}

    for split in os.listdir(dataset_path):
        if split not in dataset_split:
            continue

        split_dir = os.path.join(dataset_path, split)
        if not os.path.isdir(split_dir):
            continue

        for label_folder in os.listdir(split_dir):
            folder_path = os.path.join(split_dir, label_folder)
            if not os.path.isdir(folder_path):
                continue

            print(f"Reading '{label_folder}' from {folder_path}")
            video_count = 0

            for video_filename in os.listdir(folder_path):
                if video_count >= videos_per_label:
                    break

                video_path = os.path.join(folder_path, video_filename)

                # Skip videos shorter than 4 seconds
                duration = get_video_duration(video_path)
                if duration < 4.0:
                    print(f"Skipping {video_filename}: Duration {duration:.2f}s < 4s")
                    continue

                try:
                    with open(video_path, "rb") as f:
                        video_bytes = f.read()

                    task = client.tasks.create(
                        index_id=index_id,
                        video_file=video_bytes,
                        user_metadata=json.dumps(
                            {"local_video_file_path": video_path}
                        ),
                    )
                    print(f"Created task for {video_path} with ID {task.id}")

                    # Wait for indexing to complete
                    wait_task = client.tasks.wait_for_done(task_id=task.id)
                    if wait_task.status != "ready":
                        raise Exception(
                            f"Task {task.id} failed with status {wait_task.status}"
                        )

                    retrieve_task = client.tasks.retrieve(task_id=task.id)
                    video_ids[video_filename] = retrieve_task.video_id
                    print(f"Video indexed successfully with ID {retrieve_task.video_id}")
                    video_count += 1

                except Exception as e:
                    print(f"Failed to index {video_filename}: {e}")

    return video_ids


def fetch_videos_from_index(client: TwelveLabs, index_id: str):
    """
    Fetch videos and their embeddings from TwelveLabs index.

    Yields:
        Tuples of (video_file_path, video_id, video_embedding)
    """
    response = client.indexes.videos.list(index_id=index_id)

    for video in response:
        video_info = client.indexes.videos.retrieve(
            index_id=index_id,
            video_id=video.id,
            embedding_option=["visual"],
        )
        video_file_path = video_info.user_metadata.get("local_video_file_path")
        video_embedding = video_info.embedding.video_embedding.segments[0].float_

        yield video_file_path, video.id, video_embedding


def populate_fiftyone_dataset(
    client: TwelveLabs, dataset: fo.Dataset, index_id: str
) -> list:
    """
    Populate FiftyOne dataset with videos and embeddings from TwelveLabs.

    Args:
        client: TwelveLabs client instance
        dataset: FiftyOne dataset instance
        index_id: TwelveLabs index ID

    Returns:
        List of video embeddings
    """
    # Clear existing samples
    dataset.delete_samples(dataset)

    embeddings = []

    for video_file_path, video_id, video_embedding in fetch_videos_from_index(
        client, index_id
    ):
        embeddings.append(video_embedding)

        sample = fo.Sample(
            filepath=video_file_path,
            video_id=video_id,
        )
        dataset.add_sample(sample)

        print(f"{video_file_path} -> {video_id}")
        print(f"  Embedding preview: {video_embedding[:5]}")

    print(f"Added {len(embeddings)} samples to dataset")
    return embeddings


def generate_label_with_pegasus(client: TwelveLabs, video_id: str) -> str:
    """
    Generate a semantic label for a video using TwelveLabs Pegasus 1.2.

    Args:
        client: TwelveLabs client instance
        video_id: TwelveLabs video ID

    Returns:
        Generated label string
    """
    result = client.analyze(
        video_id=video_id,
        prompt=(
            "Generate a single label either as a single word or phrase "
            "(with _ separating spaces) to represent the video and its "
            "respective cluster of similar videos. This dataset relates to "
            "workplace safety violations and good practices, so please "
            "identify exact violation or good practice in video"
        ),
        temperature=0.2,
    )
    return result.data


def cluster_and_label_videos(
    client: TwelveLabs,
    dataset: fo.Dataset,
    embeddings: list,
    num_clusters: int = 8,
) -> tuple:
    """
    Cluster videos using KMeans and generate semantic labels.

    Args:
        client: TwelveLabs client instance
        dataset: FiftyOne dataset instance
        embeddings: List of video embeddings
        num_clusters: Number of clusters for KMeans

    Returns:
        Tuple of (cluster_labels, label_map)
    """
    print(f"Clustering {len(embeddings)} videos into {num_clusters} clusters...")

    kmeans = KMeans(n_clusters=num_clusters, random_state=0)
    cluster_labels = kmeans.fit_predict(embeddings)

    # Reset cluster field if it exists
    if "cluster" in dataset.get_field_schema():
        dataset.delete_sample_field("cluster")

    labels = {}

    for sample, label in zip(dataset, cluster_labels):
        if label not in labels:
            print(f"Generating label for cluster {label} using Pegasus 1.2...")
            labels[label] = generate_label_with_pegasus(client, sample.video_id)
            print(f"  Label: {labels[label]}")

        sample["cluster"] = labels[label]
        sample.save()

    return cluster_labels, labels


def compute_visualization(dataset: fo.Dataset, embeddings: list):
    """Compute 2D UMAP visualization of embeddings."""
    print("Computing 2D visualization with UMAP...")

    results = fob.compute_visualization(
        dataset,
        embeddings=embeddings,
        num_dims=2,
        brain_key="image_embeddings",
        verbose=True,
        seed=51,
    )
    return results


class WorkerSafetyDataset(Dataset):
    """
    PyTorch Dataset for the Worker Safety Challenge.

    Items returned:
        - embedding (torch.Tensor): Visual embedding of the video
        - label_idx (torch.Tensor): Integer label index (cluster ID)
        - label_str (str): Semantic string description of the label
        - video_id (str): TwelveLabs Video ID
    """

    def __init__(self, embeddings, labels, label_map, video_ids):
        self.embeddings = torch.tensor(embeddings, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.label_map = label_map
        self.video_ids = video_ids

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        label_idx = self.labels[idx].item()
        label_str = self.label_map.get(
            label_idx, self.label_map.get(np.int32(label_idx), "Unknown")
        )

        return {
            "embedding": self.embeddings[idx],
            "label_idx": self.labels[idx],
            "label_str": label_str,
            "video_id": self.video_ids[idx],
        }


def export_dataset(
    embeddings: list,
    cluster_labels,
    label_map: dict,
    dataset: fo.Dataset,
    output_path: str = "worker_safety_dataset.pt",
):
    """
    Export the dataset as a PyTorch .pt file.

    Args:
        embeddings: List of video embeddings
        cluster_labels: Array of cluster labels
        label_map: Dictionary mapping cluster IDs to label strings
        dataset: FiftyOne dataset instance
        output_path: Path to save the .pt file
    """
    video_ids_ordered = [s.video_id for s in dataset]

    train_dataset = WorkerSafetyDataset(
        embeddings=embeddings,
        labels=cluster_labels,
        label_map=label_map,
        video_ids=video_ids_ordered,
    )

    print(f"Saving dataset with {len(train_dataset)} samples to {output_path}...")
    torch.save(train_dataset, output_path)
    print("Dataset exported successfully!")


def main():
    """Main entry point for the enablement kit."""
    parser = argparse.ArgumentParser(
        description="CVPR 2026 Worker Safety Challenge Enablement Kit"
    )
    parser.add_argument(
        "--skip-ingestion",
        action="store_true",
        help="Skip video ingestion (use existing indexed videos)",
    )
    parser.add_argument(
        "--export-only",
        action="store_true",
        help="Only export dataset (skip visualization)",
    )
    parser.add_argument(
        "--num-clusters",
        type=int,
        default=8,
        help="Number of clusters for KMeans (default: 8)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="worker_safety_dataset.pt",
        help="Output path for exported dataset",
    )
    args = parser.parse_args()

    # Load configuration
    config = load_config()

    # Initialize TwelveLabs client
    print("Initializing TwelveLabs client...")
    tl_client = TwelveLabs(api_key=config["tl_api_key"])

    # Get or create TwelveLabs index
    index_id = get_or_create_twelvelabs_index(tl_client, config["tl_index_name"])

    # Initialize FiftyOne dataset
    print(f"Initializing FiftyOne dataset '{config['dataset_name']}'...")
    if fo.dataset_exists(config["dataset_name"]):
        fo.delete_dataset(config["dataset_name"])
    dataset = fo.Dataset(config["dataset_name"])

    # Video ingestion (optional)
    if not args.skip_ingestion:
        ingest_videos(
            client=tl_client,
            index_id=index_id,
            dataset_path=config["dataset_path"],
            dataset_split=config["dataset_split"],
            videos_per_label=config["videos_per_label"],
        )

    # Fetch embeddings and populate dataset
    print("Fetching video embeddings from TwelveLabs...")
    embeddings = populate_fiftyone_dataset(tl_client, dataset, index_id)

    if not embeddings:
        print("No videos found in index. Please run without --skip-ingestion first.")
        return

    # Cluster and auto-label videos
    cluster_labels, label_map = cluster_and_label_videos(
        client=tl_client,
        dataset=dataset,
        embeddings=embeddings,
        num_clusters=args.num_clusters,
    )

    # Export dataset
    export_dataset(
        embeddings=embeddings,
        cluster_labels=cluster_labels,
        label_map=label_map,
        dataset=dataset,
        output_path=args.output,
    )

    # Launch visualization (optional)
    if not args.export_only:
        compute_visualization(dataset, embeddings)

        print("\nLaunching FiftyOne App...")
        print("Navigate to http://localhost:5151 in your browser")
        session = fo.launch_app(dataset, auto=False, port=5151)
        session.wait()


if __name__ == "__main__":
    main()
