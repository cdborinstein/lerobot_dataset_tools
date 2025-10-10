from huggingface_hub import HfApi, login
import sys

# Login first (or use huggingface-cli login in terminal)
# login()

REPO_ID = "INSERT-REPO-ID"


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

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python manage_dataset.py list-versions")
        print("  python manage_dataset.py list-episodes")
        print("  python manage_dataset.py create-tag <tag_name> <message>")
        print("  python manage_dataset.py delete-tag <tag_name>")
        print("  python manage_dataset.py delete-episodes <ep1,ep2,ep3>")
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
    else:
        print(f"Unknown command: {command}")
        sys.exit(1)
