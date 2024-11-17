# Devcontainer Install

A devcontainer is essentially a Docker container that allows you to update the code inside the container without also needing to rebuild the container. It gives you all the packages you need installed, with the correct versions, and executed as a script in a way that will never conflict with whatever other versions you've got on your system.

These instructions will walk through working with devcontainers on an Ubuntu Linux distribution. 

The easiest way to get started with devcontainers is to install VSCode, Microsoft's free code editor. If you don't have it already, VSCode can be installed with:

```
sudo apt install code
```

Once installed, open VSCode and then go to `File` > `Open Folder` and browse to the root folder of this project. On the left side, click the Extensions tab:

![Extensions tab](./imgs/devcontainer_setup-extensions_tab.png)

Then search for and install `Dev Containers` and `Docker`:

![Get extensions](./imgs/devcontainer_setup-get_extensions.png)

Once installed, use `Ctrl` + `Shift` + `p` or go to `Help` > `Show All Commands` and then search for `devcontainer cli`:

![Install devcontainer cli](./imgs/devcontainer_setup-install_devcontainer_cli.png)

Once this is installed, you're ready to use the devcontainer. Click the blue `><` icon in the bottom-left corner:

![Open devcontainer options](./imgs/devcontainer_setup-devcontainer_options.png)

And then choose `Reopen in container`: 

![Reopen in container](./imgs/devcontainer_setup-reopen_in_container.png)

If prompted for a container configuration, choose `Add configuration to workspace` and then choose to create the container configuration `From 'docker-compose.devcontainer.yaml'`. If prompted, no features or optional files/directories need to be installed, just click `OK`. 

You should now see a popup stating `Connecting to Dev Container (show log)` and you can click that message to see the log showing you the various steps that are executing to setup the container.



That's it! You're in the devcontainer now. At this point VSCode should look and feel like it's running on your host machine, but without any dependencies to manage or package conflicts. 



## Using with PyCharm

Post script on the install notes here: If you prefer using PyCharm, follow the steps above to get the docker-cli installed. This lets you run commands like `devcontainer up` to start the devcontainer and `devcontainer exec` to run commands inside the container. One of the volumes mounted by the devcontainer is `~/.local/share/JetBrains`, which should be where the files are setup on your machine. If they're not there then I *highly recommend* moving them there or you'll need to update paths in the devcontainer files to get the volumes to mount correctly. 

I've installed PyCharm using the JetBrains Toolbox, and the bash script that starts PyCharm is located (on my machine) at `~/.local/share/JetBrains/Toolbox/apps/pycharm-professional/bin/pycharm.sh`. I have this project cloned on my machine to `~/Documents/personal/projects/ece5554-lego-id`. If you find the paths to `pycharm.sh` and `ece5554-lego-id` on your machine, you can create an alias to start PyCharm inside the devcontainer:

```
alias brick_id='cd ~/Documents/personal/projects/ece5554-lego-id && devcontainer up && devcontainer exec sh ~/.local/share/JetBrains/Toolbox/apps/pycharm-professional/bin/pycharm.sh'
```

Add this to your `~/.bash_aliases` file, remember to `source ~/.bashrc` after doing so, then run `brick_id` and then PyCharm starts up inside the devcontainer.

NOTE: You may get a notice that your PyCharm is unlicensed; this is a quirk to the startup method provided and it's not clear where the licensing information is kept. You can accept the trial terms or you can go to your account and "Download a file for offline activation" and use that key to manually activate the copy running in the devcontainer. If you rebuild the devcontainer you'll have to repeat this step.