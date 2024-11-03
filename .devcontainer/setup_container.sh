#!/bin/bash
set -e

sudo service ssh start

# Enable any ssh key that can ssh into machine to be able to ssh into dev container
if [ -f /home/"$USER_HOST"/.ssh/authorized_keys ]; then
  cp /home/"$USER_HOST"/.ssh/authorized_keys /home/"$USER_DOCKER"/.ssh/authorized_keys
fi

# Allow local machine to ssh into dev container with existing key
if [ -f /home/"$USER_HOST"/.ssh/id_rsa.pub ]; then
  cat /home/"$USER_HOST"/.ssh/id_rsa.pub >> /home/"$USER_DOCKER"/.ssh/authorized_keys
fi

# Pip install the project with the -e (editable) flag so Python recognizes brick_id as a package. 
pushd $DOCKER_PROJECT_PATH > /dev/null
  echo "Installing brick_id"
  python3 -m pip install -e .
  echo "export PYTHONPATH=$PYTHONPATH:/home/team16/ece5554-lego-id/" >> /home/team16/.bashrc
popd > /dev/null