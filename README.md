# LocalMapper

Implementation of LocalMapper developed by Prof. Yousung Jung group at Seoul National University (contact: yousung@gmail.com).

## Contents

- [Developer](#developer)
- [OS Requirements](#os-requirements)
- [Python Dependencies](#python-dependencies)
- [Installation Guide](#installation-guide)
- [Data](#data)
- [Reproduce the results](#reproduce-the-results)
- [Demo](#demo)
- [Publication](#publication)
- [License](#license)

## Developer
Shuan Chen (shuankaist@kaist.ac.kr)<br>

## OS Requirements
This repository has been tested on both **Linux** and **Windows** operating systems.

## Python Dependencies
* Python (version >= 3.6)
* Numpy (version >= 1.16.4)
* PyTorch (version >= 1.0.0)
* RDKit (version >= 2019)
* DGL (version >= 0.5.2)
* DGLLife (version >= 0.2.6)

## Installation Guide
Create a virtual environment to run the code of LocalMapper.<br>
Make sure to install pytorch with the cuda version that fits your device.<br>
This process usually takes around 5-10 minutes to complete.<br>
```
git clone https://github.com/kaist-amsg/LocalMapper.git
cd LocalMapper
conda create -c conda-forge -n rdenv python -y
conda activate rdenv
conda install pytorch cudatoolkit=11.3 -c pytorch -y
conda install -c conda-forge rdkit -y
conda install -c dglteam dgl-cuda11.3
pip install dgllife
```

## Data
#### USPTO dataset
The raw reactions of USPTO 50K and USPTO FULL are downloaded from the github repo of [RXNMapper](https://github.com/rxn4chemistry/rxnmapper).

The mapped reactions of USPTO 50K and USPTO FULL are available at [Dropbox](https://www.dropbox.com/scl/fo/y2ltxgem80uiwbhky9w35/h?rlkey=30mkkxh2llhxze34zpql24jaf&dl=0).

#### Reference dataset
AAM predictions on reactions sampled from [USPTO 50K](https://pubs.acs.org/doi/10.1021/acs.jcim.6b00564), [Golden dataset](https://onlinelibrary.wiley.com/doi/10.1002/minf.202100138), and [Jaworski et al.](https://www.nature.com/articles/s41467-019-09440-2) generated  by LocalMapper, [RXNMapper](https://www.science.org/doi/10.1126/sciadv.abe4166), and [GraphormerMapper](https://pubs.acs.org/doi/10.1021/acs.jcim.2c00344) are provided [here](https://github.com/kaist-amsg/LocalMapper/tree/main/comparison).

## Reproduce the results
### [0] Change the chemist name
Go to `LocalMapper/manual/` folder and change name of file `User.user` to `[your-name].user`.

### [1] Sample the reaction from raw_data
Downlaod raw data of USPTO_50K from 
Go to `LocalMapper/scripts/` folder and run `Sample.py` with -i (iteration) = 1
```
python Sample.py -i 1
```

### [2] Manual map the sampled reaction
Back to `LocalMapper/manual/` folder and use `Check_atom_mapping.ipynb` to correct the sampled reactions (0: reject and remap, 1: accept, 2: reject and skip). 
**Make sure the templates you generate are chemically correct. The model is very sensitive to these templates.**


### [3] Train LocalMapper model
Go to the `LocalMapper/scripts/` folder, and run following training code 
```
python Train.py -i 1
```

This training process usually takes 3~6 hours to complete using cuda-supporting GPU depending on the number of training reactions.

### [4] Predict the atom-mapping for raw data 
To use the model to predict the atom-mapping on raw reactions, simply run
```
python Test.py -i 1
```

### [5] Repeat step [1]~[4] 
To sample more data for training, sample the data again and train-test the LocalMapper model by changing the arguement `-i`
To start, you should run
```
python Sample.py -i 2
```

## Demo
See `Demo.ipynb` for running instructions and expected output.

```
import torch
from AtomMapping import AtomMapper

device = torch.device('cpu')  # 'cpu' or 'cuda:0'
trained_dataset = 'USPTO_FULL' # USPTO_50K or USPTO_FULL
mapper = AtomMapper(device, dataset=trained_dataset)

rxn = 'CC(C)S.CN(C)C=O.Fc1cccnc1F.O=C([O-])[O-].[K+].[K+]>>CC(C)Sc1ncccc1F'
prediction = mapper.get_atom_map(rxn, return_confidence=return_confidence)
mapped_rxn = prediction['mapped_rxn']
print (mapped_rxn) # '[CH3:1][CH:2]([CH3:3])[SH:4].CN(C)C=O.[F:11][c:10]1[cH:9][cH:8][cH:7][n:6][c:5]1F.O=C([O-])[O-].[K+].[K+]>>[CH3:1][CH:2]([CH3:3])[S:4][c:5]1[n:6][cH:7][cH:8][cH:9][c:10]1[F:11]'
```

## Publication
Under review

## License
This project is covered under the **The GNU General Public License v3.0**.
