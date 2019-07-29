
This subdirectory contains the code to reproduce the experiments from the conference paper **Yet Another ADNI Machine Learning Paper? Paving The Way Towards Fully-reproducible Research on Classification of Alzheimer's Disease** using data obtained from [ADNI](http://adni.loni.usc.edu/) dataset. It is developed by the [ARAMIS Lab](http://www.aramislab.fr).

A more recent journal paper, **Reproducible evaluation of classification methods in Alzheimer's disease: Framework and application to MRI and PET data** using data obtained from [ADNI](http://adni.loni.usc.edu/), [AIBL](https://aibl.csiro.au/research/neuroimaging/) and [OASIS](http://www.oasis-brains.org/) datasets, includes more extensive and revised results.

# We strongly advise to use the code of the more recent paper
You can find it by switching to branch `2018_NeuroImage`


In case you still want to use the code in the current branch, we assume that you have installed all the dependencies of [Clinica software platform](http://www.clinica.run) and downloaded the ADNI original data.

### Ensure that you have installed the corresponding version of Clinica
Install Clinica in developer's mode using the commit that corresponds to the version of Clinica used in this work. For that, we suggest to use an isolated Conda environment to avoid conflicts with previous versions of libraries already installed in your computer.

```bash
git clone https://github.com/aramis-lab/clinica.git clinica_for_2017_MLMI
cd clinica_for_2017_MLMI/
git checkout 0bb46b3b8badfacd1632cf0270b096eb270629d1
# Former commit-id : a27de358cd4546ad59595d3c565afd3e9b8524a0
conda env create -f environment.yml --name 2017_MLMI_env
conda activate 2017_MLMI_env
pip install .
conda deactivate
```

Then you can execute the instructions in files inside the Code folder, in the following order
```
1-ADNI_conversion.txt
2-create_subjects_list.py
3-ADNI_preprocessing.txt
4-ADNI_classification.py
```
