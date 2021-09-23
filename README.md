# JADOC (Joint Approximate Diagonalization under Orthogonality Constraints) `beta v0.1`

`jadoc` is a Python 3.x package for joint approximate diagonalization of multiple positive (semi)-definite matrices under orthogonality constraints.

## Installation

:warning: Before downloading `jadoc`, please make sure [Git](https://git-scm.com/downloads) and [Anaconda](https://www.anaconda.com/) with **Python 3.x** are installed.

In order to download `jadoc`, open a command-line interface by starting [Anaconda Prompt](https://docs.anaconda.com/anaconda/user-guide/getting-started/), navigate to your working directory, and clone the `mgreml` repository using the following command:

```  
git clone https://github.com/devlaming/jadoc.git
```

Now, enter the newly created `jadoc` directory using:

```
cd jadoc
```

Then run the following commands to create a custom Python environment which has all of `mgreml`'s dependencies (i.e. an environment that has packages such as `numpy` and `pandas` pre-installed):

```
conda env create --file jadoc.yml
conda activate jadoc
```

(or `activate mgreml` instead of `conda activate mgreml` on some machines).

In case you cannot create a customised conda environment (e.g. because of insufficient user rights) or simply prefer to use Anaconda Navigator or `pip` to install packages e.g. in your base environment rather than a custom environment, please note that `jadoc` only requires Python 3.x with the packages `numpy`, `pandas`, and `numba` installed.

Once the above has completed, you can now run the following commands sequently, to test if `jadoc` is functioning properly:

```
python
import jadoc
jadoc.Test()
```

## Tutorial

tba

## Updating `jadoc`

You can update to the newest version of `jadoc` using `git`. First, navigate to your `jadoc` directory (e.g. `cd jadoc`), then run
```
git pull
```
If `jadoc` is up to date, you will see 
```
Already up to date.
```
otherwise, you will see `git` output similar to 
```
remote: Enumerating objects: 4, done.
remote: Counting objects: 100% (4/4), done.
remote: Compressing objects: 100% (3/3), done.
remote: Total 3 (delta 0), reused 3 (delta 0), pack-reused 0
Unpacking objects: 100% (3/3), 1.96 KiB | 111.00 KiB/s, done.
From https://github.com/devlaming/jadoc
   9c7474e..2b07455  main       -> origin/main
Updating 9c7474e..2b07455
Fast-forward
 README.md | 107 ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
 1 file changed, 107 insertions(+)
 create mode 100644 README.md
 ```
which tells you which files were changed.

If you have modified the `jadoc` source code yourself, `git pull` may fail with an error such as `error: Your local changes [...] would be overwritten by merge`. 

In case the Python dependencies have changed, you can update the `jadoc` environment with

```
conda env update --file jadoc.yml
```

## Support

Before contacting us, please try the following:

1. Go over the tutorial in this `README.md` file
2. Go over the method, described in the paper (citation below)

### Contact

In case you have a question that is not resolved by going over the preceding two steps, or in case you have encountered a bug, please send an e-mail to r\[dot\]devlaming\[at\]vu\[dot\]nl.

## Citation

If you use the software, please cite

[R. de Vlaming and E.A.W. Slob (2021). Joint Approximate Diagonalization under Orthogonality Constraints. *tba*: **tba**.](tba)

## Derivations

For full details on the derivation underpunning the `jadoc` tool, see the paper, available on **tba**.

## License

This project is licensed under GNU GPL v3.

## Authors

Ronald de Vlaming (Vrije Universiteit Amsterdam)

Eric Slob (University of Cambridge)
