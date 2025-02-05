import torch
import torch.nn as nn

import configparser as cp

#### Configuration file
config = cp.ConfigParser()
config.read('xtclim.json')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Mean-Squared Error as the average difference between the pixels
# in the original image vs. the reconstructed one
criterion = nn.MSELoss()
# pixel-wise MSE loss
pixel_wise_criterion = nn.MSELoss(reduction='none')

# KL divergence handles dispersion of information in latent space
# a balance is to be found with the prevailing reconstruction error
beta = config.get('MODEL', 'beta')
#beta = 0.1

# number of evaluations for each dataset
n_avg = config.get('MODEL', 'n_avg')
#n_avg = 20
