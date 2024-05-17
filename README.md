# xtclim
## ML-based extreme events detection and characterization

**Method**: Convolutional Variational AutoEncoder.

**Input**: "Daily images", daily screenshots of Europe for maximum temperature (CMCC-ESM2 model, 1° spatial resolution).

**Output**: Error between original and reconstructed image: postprocessed for analysis in the "scenario_season_comparison.ipynb" file.

**Idea**: The more unusual an image (anomaly), the higher error.

**Folder Architecture**:
<img width="283" alt="image" src="https://github.com/cerfacs-globc/xtclim/assets/44207112/97813269-1632-4ae1-b30d-c0e17940734c">


**How to Run**:

_In the following part, "network" refers to the deep learning model, the Convolutional Variational Auto-Encoder (CVAE)._

- In the preprocessing folder, first run "preprocess_functions_2d_ssp.py" to load NetCDF files from the data folder. It will normalize and adjust the data for the network. Then run "preprocess_2d_seasons.py" to split the data into seasonal files. Preprocessed data is stored in the "input" folder.

- Run "python train.py" to train the network. Caution: It will overwrite the weights of the network already saved in outputs (unless you change the path name '../outputs/cvae_model_3d.pth' in the script).

- Run "python anomaly.py" to evaluate the network on the available datasets - train, test, and projection.

- The "data" folder is too heavy to be stored on the repository. It should contain tasmax NetCDF files for both historical (1950-1999) and projection (2015-2100) time periods (e.g. tasmax_day_CMCC-ESM2_historical_r1i1p1f1_gn_19500101-19991231.nc).

**More Details**:
- "model.py" contains the details of the architecture of the network, and "engine.py" defines the train and evaluate functions.
- The carbon footprint of the training is estimated with the codecarbon package (see first and last lines of "src/train.py").
