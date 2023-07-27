import torch
import torch.nn as nn


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Mean-Squared Error as the average difference between the pixels
# in the original image vs. the reconstructed one
criterion = nn.MSELoss()

# KL divergence handles dispersion of information in latent space
# a balance is to be found with the prevailing reconstruction error
beta = 0.1
