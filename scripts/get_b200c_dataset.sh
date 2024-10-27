#!/bin/bash
SCRIPT_PATH=$(realpath "$0")
SCRIPT_DIR=$(dirname "$SCRIPT_PATH")
B200C_DIR=${SCRIPT_DIR}/../B200C

# The directory exists, right?
if ! test -d ${B200C_DIR}; then
  echo "Failed to find the expected B200C path at ${B200C_DIR}!"
  exit 1
fi

# If you got here, the directory exists, so get the dataset. This will download the ZIP file to this directory
echo "Downloading dataset; expected file size is 1.1 GB so this may take a while..."
#curl -L -o ./b200c.zip https://www.kaggle.com/api/v1/datasets/download/ronanpickell/b200c-lego-classification-dataset

# Unzip the files to the B200C directory
echo "Unzipping to ${B200C_DIR}..."
unzip ${SCRIPT_DIR}/b200c.zip -d ${B200C_DIR}

# Remove the ZIP file
echo "Cleaning up..."
rm ${SCRIPT_DIR}/b200c.zip