# LocalMapper

Implementation of LocalMapper developed by Prof. Yousung Jung group at Seoul National University (contact: yousung@gmail.com).

## Contents

- [Developer](#developer)
- [OS Requirements](#os-requirements)
- [Python Dependencies](#python-dependencies)
- [Installation Guide](#installation-guide)
- [Usage](#usage)
- [Data](#data)
- [Reproduce the results](#reproduce-the-results)
- [Publication](#publication)
- [License](#license)

## Developer
Shuan Chen (shuan.micc@gmail.com)<br>

## OS Requirements
This repository has been tested on both **Linux** and **Windows** operating systems.

## Python Dependencies
* Python (version >= 3.6)
* Numpy (version >= 1.16.4)
* Matplotlib (version >=3.3.4)
* PyTorch (version >= 1.0.0)
* RDKit (version >= 2019)
* DGL (version >= 0.5.2)
* DGLLife (version >= 0.2.6)

## Installation Guide
### From pip
```
conda create -n localmapper python=3.6 -y
conda activate localmapper
pip install localmapper
```

### From Github
```
git clone https://github.com/snu-micc/LocalMapper.git
cd LocalMapper
conda create -n localmapper python=3.6 -y
conda activate localmapper
pip install -e .
```

## Usage
### Single rxn input
```
from localmapper import localmapper
mapper = localmapper()
rxn = 'CC(C)S.CN(C)C=O.Fc1cccnc1F.O=C([O-])[O-].[K+].[K+]>>CC(C)Sc1ncccc1F'
result = mapper.get_atom_map(rxn)
```
The expected output of `result` should be
```
'[CH3:1][CH:2]([CH3:3])[SH:4].CN(C)C=O.[F:11][c:10]1[cH:9][cH:8][cH:7][n:6][c:5]1F.O=C([O-])[O-].[K+].[K+]>>[CH3:1][CH:2]([CH3:3])[S:4][c:5]1[n:6][cH:7][cH:8][cH:9][c:10]1[F:11]'
```

### Multiple rxns input
```
rxns = ['CC(C)S.CN(C)C=O.Fc1cccnc1F.O=C([O-])[O-].[K+].[K+]>>CC(C)Sc1ncccc1F', CCOCC.C[Mg+].O=Cc1ccc(F)cc1Cl.[Br-]>>CC(O)c1ccc(F)cc1Cl]
results = mapper.get_atom_map(rxns)
```
The expected output of `results` should be
```
['[CH3:1][CH:2]([CH3:3])[SH:4].CN(C)C=O.[F:11][c:10]1[cH:9][cH:8][cH:7][n:6][c:5]1F.O=C([O-])[O-].[K+].[K+]>>[CH3:1][CH:2]([CH3:3])[S:4][c:5]1[n:6][cH:7][cH:8][cH:9][c:10]1[F:11]',
 'CCOCC.[CH3:1][Mg+].[O:3]=[CH:2][c:4]1[cH:5][cH:6][c:7]([F:8])[cH:9][c:10]1[Cl:11].[Br-]>>[CH3:1][CH:2]([OH:3])[c:4]1[cH:5][cH:6][c:7]([F:8])[cH:9][c:10]1[Cl:11]']
```

### Return results as dictionary
```
rxns = ['CC(C)S.CN(C)C=O.Fc1cccnc1F.O=C([O-])[O-].[K+].[K+]>>CC(C)Sc1ncccc1F', CCOCC.C[Mg+].O=Cc1ccc(F)cc1Cl.[Br-]>>CC(O)c1ccc(F)cc1Cl]
results = mapper.get_atom_map(rxns, return_dict=True)
```
The expected output of `results` should be
```
[{'rxn': 'CC(C)S.CN(C)C=O.Fc1cccnc1F.O=C([O-])[O-].[K+].[K+]>>CC(C)Sc1ncccc1F',
  'mapped_rxn': '[CH3:1][CH:2]([CH3:3])[SH:4].CN(C)C=O.[F:11][c:10]1[cH:9][cH:8][cH:7][n:6][c:5]1F.O=C([O-])[O-].[K+].[K+]>>[CH3:1][CH:2]([CH3:3])[S:4][c:5]1[n:6][cH:7][cH:8][cH:9][c:10]1[F:11]',
  'template': '[S:1].F-[c:2]>>[S:1]-[c:2]',
  'confident': True},
 {'rxn': 'CCOCC.C[Mg+].O=Cc1ccc(F)cc1Cl.[Br-]>>CC(O)c1ccc(F)cc1Cl',
  'mapped_rxn': 'CCOCC.[CH3:1][Mg+].[O:3]=[CH:2][c:4]1[cH:5][cH:6][c:7]([F:8])[cH:9][c:10]1[Cl:11].[Br-]>>[CH3:1][CH:2]([OH:3])[c:4]1[cH:5][cH:6][c:7]([F:8])[cH:9][c:10]1[Cl:11]',
  'template': '[C:1]-[Mg+].[C:2]=[O:3]>>[C:1]-[C:2]-[O:3]',
  'confident': True}]
```
See `Demo.ipynb` for more running instructions and plotting the results.

## Data
#### USPTO dataset
The raw reactions of USPTO 50K and USPTO FULL are downloaded from the github repo of [RXNMapper](https://github.com/rxn4chemistry/rxnmapper).

The mapped reactions of USPTO 50K and USPTO FULL are available at [Figshare](https://doi.org/10.6084/m9.figshare.25046471.v1).

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

## Publication
```bibtex
@article{chen2024precise,
  title={Precise atom-to-atom mapping for organic reactions via human-in-the-loop machine learning},
  author={Chen, Shuan and An, Sunggi and Babazade, Ramil and Jung, Yousung},
  journal={Nature Communications},
  volume={15},
  number={1},
  pages={2250},
  year={2024},
  publisher={Nature Publishing Group UK London}
}
```

## License
This project is covered under the **The GNU General Public License v3.0**.
