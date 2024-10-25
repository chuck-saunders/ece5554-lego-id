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



