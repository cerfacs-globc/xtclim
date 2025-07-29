# xtclim

## ML-based extreme events detection and characterization (CERFACS)

The code is adapted from CERFACS' [repository](https://github.com/cerfacs-globc/xtclim/tree/master).
The implementation of a pipeline with itwinai framework is shown below.

## Method

Convolutional Variational AutoEncoder.

## Input

"3D daily images", daily screenshots of Europe for three climate
variables (maximum temperature, precipitation, wind).

## Output

Error between original and reconstructed image: postprocessed for analysis in
the `presentation_notebook.ipynb` file.

## Idea

The more unusual an image (anomaly), the higher error.

## Information on files

In the preprocessing folder, the `preprocess_functions_2d_ssp.py` class loads
NetCDF files from a `data` folder, which has to be specified in `dataset_root`
in the config file `config.yaml` (please change the location).
The given class normalizes,and adjusts the data for the network.
The function `preprocess_2d_seasons.py` splits the data into
seasonal files. Preprocessed data is stored in the `input` folder.

The file `train.py` trains the network. Caution: It will overwrite the
weights of the network already saved in outputs (unless you change the
path name `outputs/cvae_model_3d.pth` in the script). This file also contains
the inference script that evaluates the network on the available
datasets - train, test, and projection.

## How to launch training workflow

The config file `config.yaml` contains all the steps to execute the workflow.
You can launch it from the root of the repository with:

```bash
itwinai exec-pipeline --config-name config.yaml
```

> [!NOTE]
> To help debugging errors, prepend `HYDRA_FULL_ERROR=1` to your command, or
> set it as an evironment variable with `export HYDRA_FULL_ERROR=1`.
> Example:
>
> ```bash
> HYDRA_FULL_ERROR=1 itwinai exec-pipeline --config-name config.yaml
> ```

To dynamically override some (nested) fields from terminal you can do:

```bash
itwinai exec-pipeline --config-name config.yaml \
    GENERAL.dataset_root=/path/to/data \
    GENERAL.input_path=input \
    GENERAL.output_path=output
```

To run only some steps, e.g., only training step after the training
dataset has been generated, use:

```bash
itwinai exec-pipeline --config-name config.yaml +pipe_steps=[training-step]
```

## TODOs

Integration of post-processing step + distributed strategies
