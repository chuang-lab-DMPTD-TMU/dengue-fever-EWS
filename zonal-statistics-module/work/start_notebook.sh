#!/bin/bash

# Check if conda environment is activated
echo "Checking environment:"
which python || { echo "Python not found! Is your Conda environment activated?"; exit 1; }
which jupyter || { echo "Jupyter not found! Is Jupyter installed in your environment?"; exit 1; }

# List available notebooks
notebooks=(*.ipynb)

# Check if any notebooks exist
if [[ ${#notebooks[@]} -eq 0 ]]; then
  echo "No notebooks found!"
  exit 1
fi

# Prompt user to select a notebook
echo "Select a notebook to run:"
for i in "${!notebooks[@]}"; do
  echo "$((i+1)). ${notebooks[$i]}"
done

# Read user input
read -p "Enter the number of the notebook to run: " choice

# Validate user input
if [[ $choice -ge 1 && $choice -le ${#notebooks[@]} ]]; then
  notebook="${notebooks[$((choice-1))]}"
  echo "Launching $notebook..."
  jupyter notebook "$notebook" --ip=0.0.0.0 --port=8888 --no-browser --allow-root
else
  echo "Invalid selection. Exiting."
  exit 1
fi
