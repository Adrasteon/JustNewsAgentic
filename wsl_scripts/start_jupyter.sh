#!/bin/bash
# JupyterLab Launcher for RAPIDS Environment
# Starts JupyterLab with development-friendly settings
echo "🧪 Starting JupyterLab for RAPIDS Development..."
source /home/nvidia/.venvs/rapids25.06_python3.12/bin/activate
jupyter-lab --allow-root --ip=0.0.0.0 --no-browser --NotebookApp.token='' --NotebookApp.password='' 2>&1
