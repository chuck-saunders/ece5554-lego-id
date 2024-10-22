# ece5554-lego-id



## Getting Started

Before cloning, **please [install git-lfs](https://docs.github.com/en/repositories/working-with-files/managing-large-files/installing-git-large-file-storage)**. From [this document on GitLab](https://docs.gitlab.com/ee/topics/git/lfs/): 

> Git can’t track changes to binary files (like audio, video, **or image files**) the same way it tracks changes to text files. While text-based files can generate plaintext diffs, any change to a binary file requires Git to completely replace the file in the repository. Repeated changes to large files increase your repository’s size. Over time, this increase in size can slow down regular Git operations like `clone`, `fetch`, or `pull`.

(emphasis added)

### Using the Devcontainer

I highly recommend using devcontainers - it's essentially all the packages you need installed, with the correct versions, done as a script and executed in a way that will never conflict with whatever other versions you've got on your system. Granted, for *Python* we could just use Python version managers, like anaconda, or conda, or poetry, etc. Even there there is still something to 

