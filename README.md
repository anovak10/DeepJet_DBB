DeepJet: Repository for training and evaluation of deep neural networks for HEP
===============================================================================
 ----  Deep Double C tagger modifications ----
 ----  gpu_env modified to run on Maxwell Cluster at DESY  ----
 ----  Modification to the class plotter and to CNN output layer ----

Setup (DESY-Maxwell or CERN)
==========
If you proceed to the installation on a CERN machine, then it is essential to perform all these steps on lxplus7. Simple ssh to 'lxplus7' instead of 'lxplus'.

Pre-Installtion: Anaconda setup (only once)
Download miniconda3
```
cd <afs work directory: you need some disk space for this!>
wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
```
Please follow the installation process. If you don't know what an option does, please answer 'yes'.
After installation, you have to log out and log in again for changes to take effect.
If you don't use bash, you might have to add the conda path to your .rc file
```
export PATH="<your miniconda directory>/miniconda3/bin:$PATH"
```
This has to be only done once.


Installation:
```
mkdir <your working dir>
cd <your working dir>
#git clone https://github.com/mstoye/DeepJet #Original repo
#git clone https://github.com/anovak10/DeepJet_DBB.git is what I forked on March 4th, 2018
git clone git@github.com:mastrolorenzo/DeepJet_DBB.git
cd DeepJet/environment

## Normal environment
./setupEnv.sh deepjetLinux3.conda

# GPU enabled environment
./setupEnv.sh deepjetLinux3.conda gpu
source gpu_env.sh
pip uninstall tensorflow
pip install tensorflow-gpu
```
Some tools to plot the NN architecture:
```
pip install graphviz
pip install pydot-ng
```
This will take a while. Please log out and in again once the installation is finised.

When the installation was successful, the DeepJet tools need to be compiled.
```
cd <your working dir>
cd DeepJet/environment
bash
source lxplus_env.sh / gpu_env.sh
cd ../modules
make -j 4
```
After successfully compiling the tools, log out and in again.
The environment is set up.

DESY Maxwell setup:
Request to be added to group of users at maxwell.service@desy.de
```
ssh naf-cms.desy.de
ssh max-wgs # Login node
# Install your environment as above
# Install to $HOME as /scratch/ is machine dependant
salloc -N 1 --partition=all --constraint=GPU --time=300 # To ask for an interactive GPU worker node session. Please note that time is in minutes!
ssh <worker node name>
nvidia-smi # to check gpu availability
```


Usage
==============

After logging in, please source the right environment (please cd to the directory first!):
```
cd <your working dir>/DeepJet/environment
bash
source lxplus_env.sh / gpu_env.sh
```

The training/test input preprocessing
====

- define the data structure for the training (example in modules/TrainData_template.py)
  for simplicity, copy the file to TrainData_template.py and adjust it. 
  Define a new class name (e.g. TrainData_template), leave the inheritance untouched
  
- register this class in DeepJet/convertFromRoot/convertFromRoot.py by 
  a) importing it (the line in the code is indiacted by a comment)
  b) adding it to the class list below'

- Make a list of training and testing samples
  ```
  python list_writer.py --train <path/to/train/files> --test <path/to/test/files>
  ```
- So far, the data structure used are:
  ```
  TrainData_deepDoubleC_db_cpf_sv_reduced   for Hcc vs QCD discrimination
  TrainData_deepDoubleCvB_db_cpf_sv_reduced   for Hcc vs Hbb discrimination 	
  ```  

- convert the root file to the data strucure for training:
  ```
  # Prepare train data
  cd DeepJet/convertFromRoot
  ./convertFromRoot.py -i /path/to/the/root/ntuple/list_of_root_files.txt -o /output/path/that/needs/some/disk/space -c TrainData_myclass
  #example
  python convertFromRoot.py -i train_list.txt -o Jan23_train_full_BB -c TrainData_deepDoubleB_db_pf_cpf_sv

  # Prepare test data
  python convertFromRoot.py --testdatafor Jan23_train_full_BB/trainsamples.dc -i test_files.txt -o Jan23_test_full_BB
  ```
  
  This step can take a while.

GPU Setup
====
In case of wanting to run on gpu's, check availability with 
```
nvidia-smi
#After loading cuda library also:
#nvcc -V
```



Training
====
In trainEval.py verify settings or go with the included defaults.

```
cd DeepJet/Train
python trainEval.py --train [--gpu] -i <path/to/train/dataCollection.dc> -n <name/suffix/of/the/training/directory>
```


Evaluation
====
In trainEval.py verify settings or go with the included defaults. By default the test dataCollection is searched for in an equivalent directory to that of train dataCollection e.g. xxxxx_train_xxxx/dataCllection.dc -> xxxxx_test_xxxx/dataCllection.dc. Specify if otherwise.

```
cd DeepJet/Train
python trainEval.py --eval [--gpu ] -i <path/to/test/dataCollection.dc> -n <name/suffix/of/the/training/directory>
```


