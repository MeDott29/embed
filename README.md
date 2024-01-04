repo_data.py grabs the tree from your repo of interest, filters out many different files and writes the remaining files to github_files_cache.json
write_files.py clones the repo, filters it based on the filepaths from github_files_cache.json, and flattens the directory structure
