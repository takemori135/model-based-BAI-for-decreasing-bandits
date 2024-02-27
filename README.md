# Source Code for "Model-Based Best Arm Identification for Decreasing Bandits"

## Dependencies
We use the Python programming language for implementation.
Dependencies can be install as follows:

```bash
   pip install -r requirements.txt
```

## Reproduction of the Experiments 
### Experiments in the Synthetic Environments
Experimental results are stored in the `data` directory.
However, one can reproduce the experiments in Section 7 as follows:

```bash
    python -m unittest tests.test_ns_bai.NSBAI.test_dump_rhos -v
    python -m unittest tests.test_ns_bai.NSBAI.test_run_onegroupof_opt_arm_exp_dim4 -v
    python -m unittest tests.test_ns_bai.NSBAI.test_run_onegroupof_opt_arm_exp_dim50 -v
    python -m unittest tests.test_ns_bai.NSBAI.test_generate_plots -v
```
Here, by executing the first line, `rhos.json` will be stored in the data directory.
Then, by executing the second (resp. third) line, experimental results of the lower dimension (resp. higher dimension) will be saved in the data directory.
The 4th line generates some figures (Fig. 2, Fig. 3, 4, 5, 6) in the image directory.

### Table in Section C
One can generate the table in Section C by the following command and executing the jupyter notebook `notebooks/table-appendix.ipynb`.
```bash
    python -m unittest tests.test_cc.TestCC -v
```

## Directory Structure 
We briefly explain the structure of directories.
Images are stored in the `image` directory
and experimental results are stored in the `data` directory.
The proposed methods and baselines for the decreasing bandits BAI problem are implemented in the 
`nonstationary_bai` directory.
Scripts for reproducing experiments are in the `test` directory.

## LICENSE
BSD 3-Clause Clear License (see LICENSE.txt)
