# xtclim
## ML-based extreme events detection and characterization

**Method**: Convolutional Variational AutoEncoder.

**Input**: "3D daily images", daily screenshots of Europe for three climate variables (maximum temperature, precipitation, wind).

**Output**: Error between original and reconstructed image.

**Idea**: The more unusual an image (anomaly), the higher error.

**Use**:
- Run "python train.py" once in the right working directory to train the network.
/!\ It will overwrite the model saved in outputs if you don't change the name of the path '../outputs/cvae_model_3d.pth' in the code - e.g. '../outputs/cvae_model_3d_v2.pth'.

- Run "python anomaly.py" once in the right working directory to evaluate the model on the three datasets available - train, test, and projection.

- The "data" folder could not be uploaded because of its weight. It contains the NetCDF files pr_day_CMCC-ESM2_historical_r1i1p1f1_gn_19750101-19991231.nc for three climate variables (pr precipitations, tasmax maximum temperature at surface, and sfcWind wind), for both historical (1975-1999) and projection (2065-2089) time periods.

- The "input" folder is incomplete for the same reason. The preprocessed test data only could be uploaded. The three missing files can be obtained by running preprocess_3d_data.ipynb with the previously mentioned NetCDF files.

Template for the network architecture: https://debuggercafe.com/convolutional-variational-autoencoder-in-pytorch-on-mnist-dataset/.
