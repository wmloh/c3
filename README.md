# Conditionally Coupled Contextual ($C_3$) Thompson Sampling

*Official GitHub repository for the implementation of **A Practical Algorithm for Feature-Rich, Non-Stationary Bandit Problems**.*

📖 You can read our paper [here](https://openreview.net/pdf?id=tRbwfej9uY).

## Setup

**MIND dataset** Download the MIND dataset [here](https://msnews.github.io/). In our paper, we use only the full datasets, i.e. `MINDlarge_train` and `MINDlarge_dev`. Place them in `$ROOT/data`. For example, the `behaviors.tsv` should be located at `$ROOT/data/MINDlarge_train/behaviors.tsv`

**Python** We use Python 3.10.12. As for dependencies, we use PyTorch (follow instructions from the official website) and the following packages: `numpy pandas matplotlib scipy scikit-learn striatum coba gpytorch transformers`.

## Guides on $C_3$

**Implementation file** The main implementation of $C_3$ can be found in `modules/model/c3.py`. The `C3` class is a PyTorch `Module`. The standard `forward` function is simply an IWKR prediction (i.e. mean-based). To sample an arm based on the Beta posterior distribution, use `infer_batch`. The training functions are encapsulated in `fit` and `fit_time_partitioned`.

**Training and inference** A dummy usage example involving the training of the embedding model $\phi$ and online evaluation can be found in `examples/basic_usage.py` in the section labeled "C3 TRAINING & EVALUATION". 

**Embedding and visualization** To visualize the embedding of the trained $\phi$ (embedding model) and to gain some understanding of how $C_3$ makes predictions, you can use `model.project` to get the embeddeding. A complete visualization example can be found in `examples/basic_usage.py` in the section labeled "EMBEDDING VISUALIZATION".

## Evaluation

By default, the results of experiments will be stored in `results/$DATASET/$ALGORITHM/$SEED`. Note that there has been upgrades to the implementation of the $C_3$ algorithm after the time of paper submission. The evaluation scripts are based on the legacy implementation in `modules/model/c3_paper.py` to ensure consistency with the results shown in the paper.

**OpenML** To run evaluation on the four OpenML datasets, configure the variables in `scripts/launch_eval_openml.sh` then run it with 

```bash
bash scripts/launch_eval_openml.sh
```

$C_3$ hyperparameters and experiment configurations can be adjusted in `configs/$DATASET.json`. To run on a different OpenML dataset, simply duplicate one of the config files and change the `"DATASET"` value to another name. 

**MIND** To run evaluation on the MIND dataset, configure the variables in `scripts/launch_eval_mind.sh` then run it with 

```bash
bash scripts/launch_eval_mind.sh
```

Hyperparameters for algorithms can be adjusted in `configs/mindargs/$ALGORITHM.json`.

**Collating results** After running the experiments with multiple seeds, collate the results and produce the figures by either running the following Python scripts:

```bash
python3 scripts/collate_openml_results.py
python3 scripts/collate_mind_results.py
```

If you have run the launch scripts with the default settings, you should be able to run the collate scripts directly. Otherwise, you may need to change the variables to match the new paths or names.

**Differences between the legacy and current $C_3$ implementation** There are three changes.

1.  The variance of IWKR estimate computation (in the `forward` function) has been corrected but the change causes a slight deviation in the results. Hence, we leave it unchanged in the legacy version. The Thompson sampling computation (in the `infer_batch` function) is correct and unchanged. 
2.  The `pick_best_val` implementation in `C3.fit` is incorrect in the version that was used at the submission of the paper. The error has been marked with a `TODO` comment. The current version rectifies this part of the implementation.
3.  The current version includes a minimum alpha and beta value threshold (specified in the parameter `min_alpha_beta`). The default value is 0.1. This helps enforce desirable properties. For example, when a query is far from all reference points, the minimum threshold ensures the standard deviation of the Beta posterior remains high and the density function being more uniform. To disable this, simply set the value to 0.

## Citation

If you used $C_3$, please include the following citation:

```
@article{
    loh2026a,
    title={A Practical Algorithm for Feature-Rich, Non-Stationary Bandit Problems},
    author={Wei Min Loh and Sajib Kumer Sinha and Ankur Agarwal and Pascal 
    Poupart},
    journal={Transactions on Machine Learning Research},
    issn={2835-8856},
    year={2026},
    url={https://openreview.net/forum?id=tRbwfej9uY}
}
```