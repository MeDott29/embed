import requests
import json

def get_github_repository_files(owner, repo, cache_file='github_files_cache.json'):
    try:
        # Try to load the file list from the cache file
        with open(cache_file, 'r') as f:
            files = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        # If the cache file doesn't exist or is invalid, fetch the file list from the API
        url = f"https://api.github.com/repos/{owner}/{repo}/git/trees/main?recursive=1"
        response = requests.get(url)
        tree = json.loads(response.text)
        ignored_paths = ["Dockerfile",".diff",".jsonl",".json",".txt",".html",".png",".yml","tests/",".vscode/",".github/", ".grit/", ".vsccode/", "wandb/", ".coveragerc", ".gitignore", ".mypy.ini", ".pre-commit-config.yaml",".ruff.toml","LICENSE","ellipsis.yaml","mkdocs.yml","poetry.lock","pyproject.toml","requirements-doc.txt","requirements.txt"]
        files = [file['path'] for file in tree['tree'] if file['type'] == 'blob' and not any(ignored_path in file['path'] for ignored_path in ignored_paths)]
        # Save the file list to the cache file
        with open(cache_file, 'w') as f:
            json.dump(files, f)
    return files

owner = "jxnl"
repo = "instructor"
files = get_github_repository_files(owner, repo)

print(files)
print(f"Found {len(files)} files of interest in {repo}")
for file in files:
    print(file)

import os
from collections import defaultdict

# Create a dictionary where the keys are file extensions
# and the values are lists of file paths with that extension
files_by_extension = defaultdict(list)
for file in files:
    extension = os.path.splitext(file)[1]
    files_by_extension[extension].append(file)

# Print the count for each file type
# If there's only one file with a given extension, print the file path as well
for ext, file_list in files_by_extension.items():
    print(f"{ext}: {len(file_list)}")
    if len(file_list) == 1:
        print(f"File path: {file_list[0]}")