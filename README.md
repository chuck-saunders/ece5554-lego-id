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

I highly recommend using devcontainers, but it's not required. You can check out install instructions [here](./doc/DEVCONTAINER.MD). 

