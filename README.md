# ece5554-lego-id

## Project Proposal

Please find our project proposal [here on Google Sites](https://sites.google.com/view/toy-brick-identification). 

## Getting Started

Before cloning, **please [install git-lfs](https://docs.github.com/en/repositories/working-with-files/managing-large-files/installing-git-large-file-storage)**. From [this document on GitLab](https://docs.gitlab.com/ee/topics/git/lfs/): 

> **Git can’t track changes to** binary files (like audio, video, or **image files**) the same way it tracks changes to text files. While text-based files can generate plaintext diffs, any change to a binary file requires Git to completely replace the file in the repository. Repeated changes to large files increase your repository’s size. Over time, this increase in size can slow down regular Git operations like `clone`, `fetch`, or `pull`.

(emphasis added)

You can install git-lfs on Ubuntu with the following command:

```
sudo apt install git-lfs
```

and then you can get it ready for use with:

```
git lfs install --skip-repo
```

### Using the Devcontainer

I highly recommend using devcontainers, but it's not required. You can check out install and usage instructions [here](./doc/DEVCONTAINER.MD). 

### Not Using the Devcontainer

If you are not using the devcontainer, then you *should* be able to `pip install` the package and all required dependencies (specified in `./pyproject.toml`) should be automatically installed. **NOTE** This may cause conflicts with your existing system packages. Environment management (pyenv, poetry, etc.) is up to you. 

Be sure to install the package with the `--editable` or `-e` flag, so changes you make to the project are automatically reflected in the installed package:

```
cd /path/to/project
python3 -m pip install -e .
```

Be sure also to add the project root to your PYTHONPATH:

```
echo "export PYTHONPATH=$PYTHONPATH:/path/to/project/" >> /home/your_username/.bashrc
```

## Getting the B200C Dataset

Included in the project is a script at `./scripts/get_b200c_data.sh`. This will download the dataset ZIP file to the `./scripts/` directory, unzip the files to the `./B200C/` directory, then remove the ZIP file. The ZIP file is approximately 1.1 GB and the unzipped files are approximately 1.7 GB, so this will consume about 2.8 GB of disk space before freeing the 1.1 GB consumed by the ZIP file. 

The `./B200C/` directory exists in the Git repo but the contents are ignored. Adding the images to a project subfolder by script allows the training dataset to be referenced by conventional and docker-based workflows. 

## Creating Dataset Reference Material

Assuming you followed the steps above for getting started with the project and you are currently at the project root, you can create the dataset reference material with the following command:

```
python3 ./brick_id/dataset/part_selection.py
```

Running this script will do several things:

1. It will seed a random number generator with the class number `5554`, then draw samples from the B200C dataset,
2. It will scrape BrickLink for images associated with each part. These images will be downloaded to the `./brick_id/dataset/imgs` folder, which is configured to be ignored by Git. 
3. It will combine the datasets and images into a PDF file that gives the image, a text description, and the quantities required for each part. For the quantities, there is a `Min` value, which is the fewest number of parts required. The Min value means the same physical part may appear in multiple dataset images. The `Ideal` value is the quantity required to ensure a unique instance of the part number is in each dataset. Finally, the quantities required for each dataset are given. 

The output PDF file is located at `./brick_id/dataset/scenario_pull_sheet.pdf`, and is also setup to be ignored by Git. 