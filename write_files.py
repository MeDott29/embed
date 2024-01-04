import os
import shutil
import git
import json

def save_github_files(owner, repo, files, save_dir):
    # Clone the repository
    git.Repo.clone_from(f"https://github.com/{owner}/{repo}.git", save_dir)

    # Get a set of all file paths in the repository
    all_files = set(os.path.join(root, file) 
                    for root, dirs, files in os.walk(save_dir) 
                    for file in files)

    # Get a set of the file paths we want to keep
    keep_files = set(os.path.join(save_dir, file) for file in files)

    # Calculate the set of files to remove
    remove_files = all_files - keep_files

    # Remove the unwanted files
    for file_path in remove_files:
        os.remove(file_path)

    # Flatten the directory structure
    for file_path in keep_files:
        flat_name = file_path.replace('/', '_')
        shutil.move(file_path, os.path.join(save_dir, flat_name))

    # Remove empty directories
    for root, dirs, files in os.walk(save_dir, topdown=False):
        for name in dirs:
            try:
                os.rmdir(os.path.join(root, name))
            except OSError:
                pass  # Directory not empty

# usage
owner = "jxnl"
repo = "instructor"
save_dir = "instructor_filtered"

# Load file paths from github_files_cache.json
with open('github_files_cache.json', 'r') as f:
    files = json.load(f)

save_github_files(owner, repo, files, save_dir)