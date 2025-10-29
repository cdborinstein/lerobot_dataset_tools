from huggingface_hub import HfApi, login
from datasets import load_dataset, concatenate_datasets, DatasetDict
import sys

# Login first (or use huggingface-cli login in terminal)
# login()

REPO_ID = "argus-systems/pickup-carrot-openpi"


def create_tag(tag_name, message):
    """Create a version tag for the dataset

    Args:
        tag_name (str): Name of the tag to create (e.g., "v1.0")
        message (str): Description message for the tag

    Usage:
        python manage_dataset.py create-tag v1.0 "Initial version"
    """
    api = HfApi()
    api.create_tag(
        repo_id=REPO_ID,
        repo_type="dataset",
        tag=tag_name,
        tag_message=message,
        revision="main"
    )
    print(f"✓ Tag '{tag_name}' created successfully!")

def list_versions():
    """List all branches and tags in the dataset repository

    Args:
        None

    Usage:
        python manage_dataset.py list-versions
    """
    api = HfApi()
    refs = api.list_repo_refs(repo_id=REPO_ID, repo_type="dataset")

    print("Branches:")
    for branch in refs.branches:
        print(f"  - {branch.name}")

    print("\nTags:")
    for tag in refs.tags:
        print(f"  - {tag.name}")

def delete_tag(tag_name):
    """Delete a version tag from the dataset

    Args:
        tag_name (str): Name of the tag to delete (e.g., "v1.0")

    Usage:
        python manage_dataset.py delete-tag v1.0
    """
    api = HfApi()
    confirm = input(f"Delete tag '{tag_name}'? (yes/no): ")
    if confirm.lower() == 'yes':
        api.delete_tag(repo_id=REPO_ID, repo_type="dataset", tag=tag_name)
        print(f"✓ Deleted tag: {tag_name}")

def list_episodes():
    """List all episodes in the dataset with file counts

    Args:
        None

    Returns:
        list: Sorted list of episode numbers found in the dataset

    Usage:
        python manage_dataset.py list-episodes
    """
    api = HfApi()
    files = api.list_repo_files(repo_id=REPO_ID, repo_type="dataset")

    # Get unique episode numbers from parquet files
    parquet_files = [f for f in files if f.endswith('.parquet') and 'episode_' in f]
    episode_numbers = sorted(set([int(f.split('episode_')[1].split('.')[0]) for f in parquet_files]))

    print(f"\nFound {len(episode_numbers)} episodes:")
    for ep_num in episode_numbers:
        # Count associated files for this episode
        related_files = [f for f in files if f'episode_{ep_num:06d}' in f]
        print(f"  Episode {ep_num}: {len(related_files)} files (1 parquet + {len(related_files)-1} videos)")

    return episode_numbers

def delete_episodes(episode_indices):
    """Delete specific episodes from the dataset

    Deletes all associated files for each episode including:
    - Parquet data file (1 per episode)
    - Video files from all cameras (4 per episode)

    Args:
        episode_indices (list of int): List of episode indices to delete (e.g., [0, 1, 5])

    Usage:
        python manage_dataset.py delete-episodes 0,1,5
        python manage_dataset.py delete-episodes 10
    """
    api = HfApi()

    # Get all files in the dataset
    all_files = api.list_repo_files(repo_id=REPO_ID, repo_type="dataset")

    # Find files to delete based on episode indices
    # Each episode has: 1 parquet + 4 mp4s (one per camera)
    files_to_delete = []
    for idx in episode_indices:
        episode_pattern = f'episode_{idx:06d}'
        matching_files = [f for f in all_files if episode_pattern in f]
        files_to_delete.extend(matching_files)

    if not files_to_delete:
        print(f"⚠ No files found for episodes: {episode_indices}")
        return

    # Group files by episode for display
    print(f"\nFiles to be deleted for {len(episode_indices)} episode(s):")
    for idx in episode_indices:
        episode_pattern = f'episode_{idx:06d}'
        episode_files = [f for f in files_to_delete if episode_pattern in f]
        print(f"\n  Episode {idx} ({len(episode_files)} files):")
        for f in episode_files:
            print(f"    - {f}")

    confirm = input(f"\nDelete {len(files_to_delete)} total files? (yes/no): ")
    if confirm.lower() == 'yes':
        print("\nDeleting files...")
        for file_path in files_to_delete:
            try:
                api.delete_file(
                    path_in_repo=file_path,
                    repo_id=REPO_ID,
                    repo_type="dataset",
                    commit_message=f"Delete episodes {episode_indices}"
                )
                print(f"✓ Deleted: {file_path}")
            except Exception as e:
                print(f"✗ Failed to delete {file_path}: {e}")

        print(f"\n✓ Successfully deleted episodes {episode_indices}")
    else:
        print("Cancelled.")

def merge_datasets(source_repo_id, target_repo_id=None, output_repo_id=None):
    """Safely merge two datasets using HuggingFace's native concatenate function

    This function uses datasets.concatenate_datasets() which properly handles:
    - Internal metadata and indices
    - Schema validation
    - Episode renumbering
    - File associations (parquet + videos)

    Args:
        source_repo_id (str): Source dataset to merge from (e.g., "username/source-dataset")
        target_repo_id (str): Target dataset to merge into (default: uses REPO_ID)
        output_repo_id (str): Where to push merged result (default: uses target_repo_id)

    Usage:
        # Merge source into REPO_ID and update REPO_ID
        python manage_dataset.py merge-datasets username/source-dataset

        # Merge into a new dataset (keeps both originals intact)
        python manage_dataset.py merge-datasets username/source-dataset --output username/merged-dataset

        # Merge two specific datasets
        python manage_dataset.py merge-datasets username/source --target username/target --output username/result
    """
    # Set defaults
    if target_repo_id is None:
        target_repo_id = REPO_ID
    if output_repo_id is None:
        output_repo_id = target_repo_id

    print(f"\n=== Merging Datasets ===")
    print(f"Source: {source_repo_id}")
    print(f"Target: {target_repo_id}")
    print(f"Output: {output_repo_id}")
    print()

    # Load target dataset
    print(f"Loading target dataset: {target_repo_id}...")
    try:
        target_dataset = load_dataset(target_repo_id, split="train")
        print(f"✓ Loaded target: {len(target_dataset)} examples")
    except Exception as e:
        print(f"✗ Error loading target dataset: {e}")
        return

    # Load source dataset
    print(f"\nLoading source dataset: {source_repo_id}...")
    try:
        source_dataset = load_dataset(source_repo_id, split="train")
        print(f"✓ Loaded source: {len(source_dataset)} examples")
    except Exception as e:
        print(f"✗ Error loading source dataset: {e}")
        return

    # Validate schemas match
    print("\nValidating dataset schemas...")
    if target_dataset.features != source_dataset.features:
        print("✗ Error: Dataset schemas don't match!")
        print(f"\nTarget features: {target_dataset.features}")
        print(f"\nSource features: {source_dataset.features}")
        print("\nDatasets must have identical columns and types to merge.")
        return
    print("✓ Schemas match")

    # Show merge plan
    print(f"\nMerge plan:")
    print(f"  Target episodes: {len(target_dataset)} examples")
    print(f"  Source episodes: {len(source_dataset)} examples")
    print(f"  Total after merge: {len(target_dataset) + len(source_dataset)} examples")
    print(f"\nMerged dataset will be pushed to: {output_repo_id}")

    if output_repo_id == target_repo_id:
        print("⚠ Warning: This will UPDATE the target dataset")
    else:
        print("✓ Original datasets will remain unchanged")

    confirm = input(f"\nProceed with merge? (yes/no): ")
    if confirm.lower() != 'yes':
        print("Cancelled.")
        return

    # Concatenate datasets
    print("\nMerging datasets...")
    try:
        merged_dataset = concatenate_datasets([target_dataset, source_dataset])
        print(f"✓ Successfully merged: {len(merged_dataset)} total examples")
    except Exception as e:
        print(f"✗ Error merging datasets: {e}")
        return

    # Push to hub
    print(f"\nPushing merged dataset to {output_repo_id}...")
    try:
        merged_dataset.push_to_hub(
            output_repo_id,
            commit_message=f"Merge {source_repo_id} into {target_repo_id}"
        )
        print(f"✓ Successfully pushed to {output_repo_id}")
    except Exception as e:
        print(f"✗ Error pushing to hub: {e}")
        return

    print(f"\n=== Merge Complete ===")
    print(f"✓ Merged {len(source_dataset)} examples from {source_repo_id}")
    print(f"✓ Combined with {len(target_dataset)} examples from {target_repo_id}")
    print(f"✓ Total: {len(merged_dataset)} examples in {output_repo_id}")
    print(f"\nView your merged dataset at: https://huggingface.co/datasets/{output_repo_id}")

"""
Dataset Management Script for HuggingFace LeRobot Datasets

This script provides command-line tools to manage versions and episodes of LeRobot datasets
on HuggingFace Hub. It supports creating version tags, listing episodes, deleting
specific episodes, and safely merging datasets.

Prerequisites:
    - Login to HuggingFace: huggingface-cli login
    - Set REPO_ID to your dataset repository
    - Install dependencies: pip install huggingface-hub datasets

Running from Terminal:
    # Navigate to the script directory
    cd /path/to/lerobot_dataset_tools

    # Run the script with Python
    python manage_dataset.py <command> [arguments]

    # Or make it executable (Unix/Mac only)
    chmod +x manage_dataset.py
    ./manage_dataset.py <command> [arguments]

Usage Examples:

    # List all versions (branches and tags)
    python manage_dataset.py list-versions

    # List all episodes in the dataset
    python manage_dataset.py list-episodes

    # Create a version tag before making changes
    python manage_dataset.py create-tag v1.0 "Original dataset with all episodes"

    # Delete specific episodes (by episode number, comma-separated)
    python manage_dataset.py delete-episodes 5,10,15

    # Delete a single episode
    python manage_dataset.py delete-episodes 42

    # Delete a version tag
    python manage_dataset.py delete-tag v1.0

    # Merge datasets (SAFE - uses HuggingFace's native concatenate_datasets)
    # Basic merge: merge source into REPO_ID
    python manage_dataset.py merge-datasets username/source-dataset

    # Merge into a NEW dataset (keeps originals intact - RECOMMENDED)
    python manage_dataset.py merge-datasets username/source-dataset --output username/merged-dataset

    # Merge two specific datasets into a new one
    python manage_dataset.py merge-datasets username/source --target username/target --output username/result

Workflow Examples:

    1. Episode Cleanup Workflow:
       a. Create a tag to preserve current state:
          python manage_dataset.py create-tag v1.0 "Before cleanup"

       b. List episodes to see what you have:
          python manage_dataset.py list-episodes

       c. Delete unwanted episodes:
          python manage_dataset.py delete-episodes 5,10,15,20

       d. Create a new tag for the cleaned version:
          python manage_dataset.py create-tag v1.1 "Removed bad episodes"

    2. Safe Dataset Merge Workflow:
       a. Create tags to preserve both datasets:
          python manage_dataset.py create-tag v1.0 "Before merge"
          (repeat for source dataset if you own it)

       b. Merge into a NEW dataset (safest option):
          python manage_dataset.py merge-datasets username/source-dataset --output username/merged-dataset

       c. Verify the merged dataset looks correct on HuggingFace Hub

       d. If satisfied, you can now use the merged dataset

Note:
    - Deleting episodes modifies the main branch
    - Tags preserve snapshots of the dataset at specific points
    - Deleted episodes will leave gaps in numbering (e.g., 0-4, 6-9, etc.)
    - Merging uses HuggingFace's datasets.concatenate_datasets() which safely handles:
      * Internal metadata and indices
      * Schema validation
      * Automatic episode renumbering
      * File associations (parquet + videos)
    - Always merge to a NEW dataset (--output) first to keep originals intact!
"""

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python manage_dataset.py list-versions")
        print("  python manage_dataset.py list-episodes")
        print("  python manage_dataset.py create-tag <tag_name> <message>")
        print("  python manage_dataset.py delete-tag <tag_name>")
        print("  python manage_dataset.py delete-episodes <ep1,ep2,ep3>")
        print("  python manage_dataset.py merge-datasets <source_repo> [--target <target_repo>] [--output <output_repo>]")
        sys.exit(1)

    command = sys.argv[1]

    if command == "list-versions":
        list_versions()
    elif command == "list-episodes":
        list_episodes()
    elif command == "create-tag":
        if len(sys.argv) < 4:
            print("Usage: python manage_dataset.py create-tag <tag_name> <message>")
            sys.exit(1)
        create_tag(sys.argv[2], sys.argv[3])
    elif command == "delete-tag":
        if len(sys.argv) < 3:
            print("Usage: python manage_dataset.py delete-tag <tag_name>")
            sys.exit(1)
        delete_tag(sys.argv[2])
    elif command == "delete-episodes":
        if len(sys.argv) < 3:
            print("Usage: python manage_dataset.py delete-episodes <ep1,ep2,ep3>")
            sys.exit(1)
        episode_indices = [int(x.strip()) for x in sys.argv[2].split(',')]
        delete_episodes(episode_indices)
    elif command == "merge-datasets":
        if len(sys.argv) < 3:
            print("Usage: python manage_dataset.py merge-datasets <source_repo> [--target <target_repo>] [--output <output_repo>]")
            sys.exit(1)

        source_repo = sys.argv[2]
        target_repo = None
        output_repo = None

        # Parse optional arguments
        i = 3
        while i < len(sys.argv):
            if sys.argv[i] == "--target" and i + 1 < len(sys.argv):
                target_repo = sys.argv[i + 1]
                i += 2
            elif sys.argv[i] == "--output" and i + 1 < len(sys.argv):
                output_repo = sys.argv[i + 1]
                i += 2
            else:
                print(f"Unknown argument: {sys.argv[i]}")
                sys.exit(1)

        merge_datasets(source_repo, target_repo, output_repo)
    else:
        print(f"Unknown command: {command}")
        sys.exit(1)
