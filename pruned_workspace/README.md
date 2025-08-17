Pruned workspace
=================

This directory contains a conservative, minimal snapshot of files needed to run the core agents and tests.

Policy for missing files
- If a file is missing from this pruned snapshot you should retrieve it from the branch `platform/stable` (the baseline used to create this pruned branch).
- Example: to fetch a single file from that branch into your working tree, run:

  git checkout platform/stable -- path/to/missing/file

- After fetching any additional files, add and commit them on the `platform/pruned` branch only after verifying they are required.

Notes
- This pruned snapshot intentionally excludes build artifacts and Python bytecode (e.g. `__pycache__` and `*.pyc`) and is conservative by design — if something is missing, fetch it from `platform/stable` using the command above.
- If you need to update this README to point at a different source branch, update the branch name in this file to reflect the desired source.

Contact
- If you're unsure whether to pull a particular file from `platform/stable`, open an issue or contact the maintainer before merging large changes.
