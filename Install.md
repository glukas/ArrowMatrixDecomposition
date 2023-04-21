# Install on Piz Daint

The following instructions have been tested on Piz Daint in April 2023. You may need to adapt the instructions based on your needs.

* Swap the compiler:

```commandline
module swap PrgEnv-cray PrgEnv-gnu
```

- Install Python version >= 3.8. The following is tested for version 3.11:

```
wget https://www.python.org/ftp/python/3.11.2/Python-3.11.2.tgz
tar -zxvf Python-3.11.2.tgz
mkdir ~/bin/python3.11
cd Python-3.11.2/
./configure --prefix=/users/<username>/bin/python3.11
make
make install
```

- paste into your ~./profile:

```
alias python311=/users/glukas/bin/python3.11/bin/python3.11
```
Then, 
```commandline
source ~/.profile
```

- Create a fresh venv with the new python version and activate it

```commandline
python311 -m venv py3.11spmm
source py3.11spmm/bin/activate
python --version
```
This should print version 3.11.

Now, install the dependencies:

```commandline
pip install igraph
pip install 'scipy>=1.8'
env MPICC=cc python -m pip install --no-cache-dir mpi4py
```
For GPU:
```commandline
pip install cupy
```
For Logging:
```commandline
pip install wandb
```
For Analysis:
```commandline
pip install matplotlib
pip install tqdm
```
