#!/bin/bash

# Exit on failures
set -e

DOCKER_USER="team16"
DOCKER_HOME="/home/${DOCKER_USER}"

# Fix for an issue where .gitconfig doesn't exist before creating devcontainer
GITCONFIG=${HOME}/.gitconfig

if [ -d "$GITCONFIG" ]; then
    # Directory exists, remove directory
    rm -rf $GITCONFIG
fi

# Now that we know it doesn't exist, create it
touch $GITCONFIG

# Get the host machine paths without expanding symlinks
HOST_PROJECT_PATH=$(pwd)
PROJECT_NAME=$(basename $HOST_PROJECT_PATH)
DOCKER_PROJECT_PATH="${DOCKER_HOME}/${PROJECT_NAME}"

DEVCONTAINER_PATH=${DEVCONTAINER_PATH:=${HOST_PROJECT_PATH}/.devcontainer}

# Create the .env file
ENV_FILE=${DEVCONTAINER_PATH}/.env

# Wipe the old one, then start building a new one
rm -f $ENV_FILE

# Add additional variables needed in the target environment
echo "HOST_PROJECT_PATH=$HOST_PROJECT_PATH" >> $ENV_FILE
echo "DOCKER_PROJECT_PATH=$DOCKER_PROJECT_PATH" >> $ENV_FILE
echo "PROJECT_NAME=${PROJECT_NAME}" >> $ENV_FILE
echo "HOST_HOME_MOUNT_PATH=/home/${USER}/host_home" >> $ENV_FILE
echo "DEVCONTAINER_PATH=$DEVCONTAINER_PATH" >> $ENV_FILE
echo "DOCKER_HOME=${DOCKER_HOME}" >> $ENV_FILE
echo "DOCKER_USER=${DOCKER_USER}" >> $ENV_FILE

# Set env variable for netrc file
HOST_NETRC_FILE="${HOME}/.netrc"
DOCKER_NETRC_FILE="${DOCKER_HOME}/.netrc"
if [ ! -f "${HOST_NETRC_FILE}" ]; then
    HOST_NETRC_FILE=/dev/null
    DOCKER_NETRC_FILE="${DOCKER_HOME}/.netrc-null"
fi

echo "HOST_NETRC_FILE=$HOST_NETRC_FILE" >> $ENV_FILE
echo "DOCKER_NETRC_FILE=$DOCKER_NETRC_FILE" >> $ENV_FILE

# Set env variable for ssh file
HOST_SSH_FILE=/dev/null
DOCKER_SSH_FILE="${DOCKER_HOME}/.ssh/ssh.null"

for key_name in "id_rsa" "id_ed25519"
do
  TEMP_SSH_FILE="${HOME}/.ssh/${key_name}"

  if [ -f "$TEMP_SSH_FILE" ]; then
    export SSH_FILE_NAME="${key_name}"
    export HOST_SSH_FILE="${TEMP_SSH_FILE}"
    export DOCKER_SSH_FILE="${DOCKER_HOME}/.ssh/${SSH_FILE_NAME}"
  fi
done

echo "HOST_SSH_FILE=$HOST_SSH_FILE" >> $ENV_FILE
echo "DOCKER_SSH_FILE=$DOCKER_SSH_FILE" >> $ENV_FILE

# Need your uid for the docker-compose.devcontainer.yaml
echo "HOST_UID=$(id -u):$(id -g)" >> $ENV_FILE

# Set env variable for docker config dir
HOST_DOCKER_DIR="${HOME}/.docker"
DOCKER_DOCKER_DIR="${DOCKER_HOME}/.docker"
if [ ! -d "${HOST_DOCKER_DIR}" ]; then
    HOST_DOCKER_DIR=/dev/null
    DOCKER_DOCKER_DIR="${DOCKER_HOME}/.docker-null"
fi

echo "HOST_DOCKER_DIR=$HOST_DOCKER_DIR" >> $ENV_FILE
echo "DOCKER_DOCKER_DIR=$DOCKER_DOCKER_DIR" >> $ENV_FILE

# If we want to use ROS in the future (system integration)
echo "ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST" >> $ENV_FILE

# Ensure the .vscode-server folder exists
VSCODE_SERVER_PATH=${DEVCONTAINER_PATH}/.vscode-server
if [ ! -d $VSCODE_SERVER_PATH ]; then
    mkdir $VSCODE_SERVER_PATH
else
    # Wipe the cached settings to force the devcontainer.json settings/extensions to be used
    rm -f $VSCODE_SERVER_PATH/data/Machine/settings.json
    rm -f $VSCODE_SERVER_PATH/data/Machine/.writeMachineSettingsMarker
    rm -f $VSCODE_SERVER_PATH/data/Machine/.installExtensionsMarker
fi

SSH_PORT=${SSH_PORT:-8888}
echo "SSH_PORT=${SSH_PORT}" >> $ENV_FILE

# For troubleshooting xcb/matplotlib issues, uncomment the next line:
#echo "QT_DEBUG_PLUGINS=1" >> $ENV_FILE

# If you want to use nVidia drivers then you'll need to uncomment this:
# Define which nVidia drivers/libs to load into the container
# See https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/1.10.0/user-guide.html#driver-capabilities
# echo "NVIDIA_DRIVER_CAPABILITIES=compute,graphics,utility,video,display" >> $ENV_FILE