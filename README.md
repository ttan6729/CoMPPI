# CoMPPI ( deep learning model for multi-label PPI prediction)
This repository contains an official implementation of CoMPPI and datasets used for evaluating multi-label PPI prediction model.
----
## Environemnts
- python3.11
- torch2.0.1
- keras2.13.1
- numpy1.24.3
- pandas2.0.3
----
### Usage
```
usage: PPIM [-h] [-m M] [-o O] [-sf SF] [-i1 I1] [-i2 I2] [-i3 I3] [-e E] [-b B] [-s S] [-PSSM PSSM]
            [-blastdb BLASTDB] [-d D] [-sv SV] [-itr ITR] [-co CO] [-dp DP] [-lr LR] [-beta1 BETA1] [-dm DM]
            [-mp MP] [-tp TP] [-rp RP] [-ab AB]

options:
  -h, --help        show this help message and exit
  -m M              mode, s1 for random scheme, s3 for bfs scheme, s4 for dfs schem, s2 for using existing paration
  -o O              output path, the suffix will be used as path for saveing model and data
  -sf SF            optional input, contains path for sequence and relation file
  -i1 I1            sequence file
  -i2 I2            relation file
  -i3 I3            file path of test set indices (for mode s2)
  -e E              epochs
  -b B              batch size
  -s S              save the best mode
  -PSSM PSSM        path of PSSM
  -blastdb BLASTDB  path of blast db
  -d D              depth
  -sv SV            model save path
  -itr ITR          iteration number for validation
  -co CO            comment
  -dp DP            if add defaultPath path
  -lr LR            learning rate
  -beta1 BETA1      beta1 for adad optimizer
  -dm DM            dataset mode
  -mp MP            the path of the model weight to be loaded
  -tp TP            the path of the test set that to be saved
  -rp RP            the result path for file
  -ab AB            ablation study
```
### Sample command for training and testing
```

```

