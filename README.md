# xtclim
## ML-based extreme events detection and characterization

**Method**: Convolutional Variational AutoEncoder.

**Input**: "3D daily images", daily screenshots of Europe for three climate variables (maximum temperature, precipitation, wind).

**Output**: Error between original and reconstructed image.

**Idea**: The more unusual an image (anomaly), the higher error.

**Use**:
- Run "python train.py" once in the right working directory to train the network. Caution: It will overwrite the model saved in outputs unless you change the path name '../outputs/cvae_model_3d.pth' in the script.

- Run "python anomaly.py" once in the right working directory to evaluate the model on the three available datasets - train, test, and projection.

- The "data" folder could not be uploaded because of its weight. It contains the NetCDF files pr_day_CMCC-ESM2_historical_r1i1p1f1_gn_19750101-19991231.nc for three climate variables (pr precipitations, tasmax maximum temperature at surface, and sfcWind wind), for both historical (1975-1999) and projection (2065-2089) time periods.

- The "input" folder is incomplete for the same reason. Only preprocessed test data could be uploaded. The three missing files can be obtained by running "preprocessing/preprocess_functions_3d_data.ipynb" or "preprocessing/preprocess_3d_data.ipynb" (original data visualization in this file) with the NetCDF files mentioned above.

Template for the network architecture: https://debuggercafe.com/convolutional-variational-autoencoder-in-pytorch-on-mnist-dataset/.

Estimation of the carbon footprint of the training with the codecarbon package (see first and last lines of "src/train.py").
