# JADOC (Joint Approximate Diagonalization under Orthogonality Constraints) `beta v0.1`

`jadoc` is a Python 3.x package for joint approximate diagonalization of multiple positive (semi)-definite matrices under orthogonality constraints.

## Installation

:warning: Before downloading `jadoc`, please make sure [Git](https://git-scm.com/downloads) and [Anaconda](https://www.anaconda.com/) with **Python 3.x** are installed.

In order to download `jadoc`, open a command-line interface by starting [Anaconda Prompt](https://docs.anaconda.com/anaconda/user-guide/getting-started/), navigate to your working directory, and clone the `jadoc` repository using the following command:

```  
git clone https://github.com/devlaming/jadoc.git
```

Now, enter the newly created `jadoc` directory using:

```
cd jadoc
```

Then run the following commands to create a custom Python environment which has all of `jadoc`'s dependencies (i.e. an environment that has packages such as `numpy` and `scipy` pre-installed):

```
conda env create --file jadoc.yml
conda activate jadoc
```

(or `activate jadoc` instead of `conda activate jadoc` on some machines).

In case you cannot create a customised conda environment (e.g. because of insufficient user rights) or simply prefer to use Anaconda Navigator or `pip` to install packages e.g. in your base environment rather than a custom environment, please note that `jadoc` only requires Python 3.x with the packages `numpy`, `scipy`, `pandas`, and `numba` installed.

Once the above has been completed, you can now run the following commands, to test if `jadoc` is functioning properly:

```
python -c "import jadoc; jadoc.Test()"
```

This command should yield output along the following lines:
```
Simulating 10 distinct 100-by-100 P(S)D matrices with alpha=0.9, for run 1
Starting JADOC
Computing low-dimensional decomposition of input matrices
Initial regularization coefficient = 1
Final regularization coefficient = 1.580194019438573
Starting quasi-Newton algorithm with line search (golden section)
ITER 0: L=34.74, RMSD(g)=0.003092, step=0.618
ITER 1: L=34.243, RMSD(g)=0.006551, step=0.629
ITER 2: L=33.008, RMSD(g)=0.008855, step=0.659
ITER 3: L=31.86, RMSD(g)=0.007825, step=0.71
ITER 4: L=31.057, RMSD(g)=0.005775, step=0.758
ITER 5: L=30.577, RMSD(g)=0.003688, step=0.772
ITER 6: L=30.345, RMSD(g)=0.0023, step=0.822
ITER 7: L=30.259, RMSD(g)=0.001238, step=0.872
ITER 8: L=30.234, RMSD(g)=0.000667, step=0.804
ITER 9: L=30.227, RMSD(g)=0.000398, step=0.73
ITER 10: L=30.224, RMSD(g)=0.000261, step=0.71
ITER 11: L=30.222, RMSD(g)=0.000181, step=0.713
ITER 12: L=30.221, RMSD(g)=0.000132, step=0.716
ITER 13: L=30.221, RMSD(g)=0.000101, step=0.715
Returning transformation matrix B
Runtime: 0.846 seconds
Root-mean-square deviation off-diagonals before transformation: 0.13898
Root-mean-square deviation off-diagonals after transformation: 0.07789
```

This output shows 10 positive (semi)-definite 100-by-100 matrices were generated, denoted by **C**<sub>1</sub>, ..., **C**<sub>10</sub>, after which JADOC calculated a matrix **B** such that **BC**<sub>*k*</sub>**B**<sup>T</sup> is as diagonal as possible for *k* = 1, ..., 10. Runtime is printed together with the root-mean-square deviation of the off-diagonal elements of **C**<sub>*k*</sub> and **BC**<sub>*k*</sub>**B**<sup>T</sup>.

## Tutorial

Once `jadoc` is up-and-running, you can simply incorporate it in your Python code, as illustrated in the following bit of Python code:

```
import jadoc
import numpy as np

N=100
K=10
C=np.empty((K,N,N))

for k in range(K):
    X=np.random.normal(size=(N,N))
    C[k]=(X@X.T)/N

B=jadoc.PerformJADOC(C)

print(B@B.T)
```

The print statement at the end shows that the obtained transformation matrix is orthonormal within numerical precision.

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
2. Go over the method, described in the preprint (citation below)

### Contact

In case you have a question that is not resolved by going over the preceding two steps, or in case you have encountered a bug, please send an e-mail to r\[dot\]devlaming\[at\]vu\[dot\]nl.

## Citation

If you use the software, please cite the preprint of our manuscript:

[R. de Vlaming and E.A.W. Slob (2021). Joint Approximate Diagonalization under Orthogonality Constraints. *arXiv*:**2110.03235**.](https://arxiv.org/abs/2110.03235)

## Derivations

For full details on the derivation underpunning the `jadoc` tool, see the prepint of our manuscript, available on [arXiv](https://arxiv.org/abs/2110.03235).

## License

This project is licensed under GNU GPL v3.

## Authors

Ronald de Vlaming (Vrije Universiteit Amsterdam)

Eric Slob (University of Cambridge)
