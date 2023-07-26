# JADOC (Joint Approximate Diagonalization under Orthogonality Constraints) `beta v0.1`

`jadoc` is a Python 3.x package for joint approximate diagonalization of multiple Hermitian matrices under orthogonality constraints.

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
Simulating 5 distinct 500-by-500 real symmetric positive (semi)-definite matrices with alpha=0.9, for run 1
Starting JADOC
Computing low-dimensional approximation of input matrices
Regularization strength = 0.95
Starting quasi-Newton algorithm with line search (golden section)
ITER 0: L=-108.785, RMSD(g)=2.3e-05, step=0.694
ITER 1: L=-108.791, RMSD(g)=3.2e-05, step=0.695
ITER 2: L=-108.802, RMSD(g)=4.6e-05, step=0.696
ITER 3: L=-108.824, RMSD(g)=6.9e-05, step=0.693
ITER 4: L=-108.877, RMSD(g)=0.000106, step=0.678
ITER 5: L=-108.976, RMSD(g)=0.00013, step=0.678
ITER 6: L=-109.094, RMSD(g)=0.000126, step=0.682
ITER 7: L=-109.206, RMSD(g)=0.000119, step=0.686
ITER 8: L=-109.303, RMSD(g)=0.000108, step=0.691
ITER 9: L=-109.382, RMSD(g)=9.6e-05, step=0.695
ITER 10: L=-109.446, RMSD(g)=8.5e-05, step=0.698
ITER 11: L=-109.495, RMSD(g)=7.5e-05, step=0.702
ITER 12: L=-109.534, RMSD(g)=6.6e-05, step=0.706
ITER 13: L=-109.566, RMSD(g)=5.9e-05, step=0.708
ITER 14: L=-109.592, RMSD(g)=5.3e-05, step=0.711
ITER 15: L=-109.613, RMSD(g)=4.8e-05, step=0.713
ITER 16: L=-109.631, RMSD(g)=4.4e-05, step=0.714
ITER 17: L=-109.645, RMSD(g)=4e-05, step=0.714
ITER 18: L=-109.658, RMSD(g)=3.7e-05, step=0.714
ITER 19: L=-109.669, RMSD(g)=3.5e-05, step=0.713
ITER 20: L=-109.678, RMSD(g)=3.3e-05, step=0.713
ITER 21: L=-109.687, RMSD(g)=3.1e-05, step=0.712
ITER 22: L=-109.694, RMSD(g)=2.9e-05, step=0.712
ITER 23: L=-109.701, RMSD(g)=2.8e-05, step=0.712
ITER 24: L=-109.707, RMSD(g)=2.6e-05, step=0.712
ITER 25: L=-109.712, RMSD(g)=2.5e-05, step=0.711
ITER 26: L=-109.717, RMSD(g)=2.4e-05, step=0.711
ITER 27: L=-109.722, RMSD(g)=2.3e-05, step=0.712
ITER 28: L=-109.726, RMSD(g)=2.2e-05, step=0.712
ITER 29: L=-109.73, RMSD(g)=2.1e-05, step=0.713
ITER 30: L=-109.734, RMSD(g)=2e-05, step=0.714
ITER 31: L=-109.737, RMSD(g)=1.9e-05, step=0.715
ITER 32: L=-109.74, RMSD(g)=1.8e-05, step=0.715
ITER 33: L=-109.743, RMSD(g)=1.8e-05, step=0.715
ITER 34: L=-109.745, RMSD(g)=1.7e-05, step=0.715
ITER 35: L=-109.748, RMSD(g)=1.7e-05, step=0.714
ITER 36: L=-109.75, RMSD(g)=1.6e-05, step=0.714
ITER 37: L=-109.752, RMSD(g)=1.6e-05, step=0.714
ITER 38: L=-109.754, RMSD(g)=1.5e-05, step=0.715
ITER 39: L=-109.756, RMSD(g)=1.5e-05, step=0.715
ITER 40: L=-109.758, RMSD(g)=1.4e-05, step=0.716
ITER 41: L=-109.76, RMSD(g)=1.4e-05, step=0.717
ITER 42: L=-109.761, RMSD(g)=1.4e-05, step=0.717
ITER 43: L=-109.763, RMSD(g)=1.3e-05, step=0.717
ITER 44: L=-109.764, RMSD(g)=1.3e-05, step=0.717
ITER 45: L=-109.766, RMSD(g)=1.3e-05, step=0.717
ITER 46: L=-109.767, RMSD(g)=1.2e-05, step=0.718
ITER 47: L=-109.768, RMSD(g)=1.2e-05, step=0.718
ITER 48: L=-109.769, RMSD(g)=1.2e-05, step=0.719
ITER 49: L=-109.77, RMSD(g)=1.1e-05, step=0.719
ITER 50: L=-109.772, RMSD(g)=1.1e-05, step=0.72
ITER 51: L=-109.773, RMSD(g)=1.1e-05, step=0.72
ITER 52: L=-109.774, RMSD(g)=1.1e-05, step=0.721
ITER 53: L=-109.775, RMSD(g)=1.1e-05, step=0.721
ITER 54: L=-109.776, RMSD(g)=1e-05, step=0.721
ITER 55: L=-109.776, RMSD(g)=1e-05, step=0.721
ITER 56: L=-109.777, RMSD(g)=1e-05, step=0.721
Returning transformation matrix B
Runtime: 9.929 seconds
Root-mean-square deviation off-diagonals before transformation: 0.061598
Root-mean-square deviation off-diagonals after transformation: 0.033501
```

This output shows 5 positive (semi)-definite 500-by-500 matrices were generated, denoted by **C**<sub>1</sub>, ..., **C**<sub>10</sub>, after which JADOC calculated a matrix **B** such that **BC**<sub>*k*</sub>**B**<sup>\*</sup> is as diagonal as possible for *k* = 1, ..., 10, where **B**<sup>\*</sup> denotes conjugate transpose of **B**, which simply equals the transpose of **B** in this case, because **B** is a real matrix, as **C**<sub>*k*</sub> are real matrices.

Runtime is printed together with the root-mean-square deviation of the off-diagonal elements of **C**<sub>*k*</sub> and **BC**<sub>*k*</sub>**B**<sup>\*</sup>.

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

print((((B@B.T)-np.eye(N))**2).sum())
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

**Update July 26, 2023**: Since the initial pre-print has been posted on arXiv, `jadoc` (*i*) has been generalised to handle Hermitian input matrices (rather than just symmetric matrices) and (*ii*) has been tweaked in terms of how the input matrices are regularised after obtaining their low-dimensional approximation. An updated version of the manuscript will be shared asap.

## License

This project is licensed under GNU GPL v3.

## Authors

Ronald de Vlaming (Vrije Universiteit Amsterdam)

Eric Slob (University of Cambridge)
