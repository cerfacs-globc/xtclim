import configparser as cp

import torch
import torch.nn as nn


def initialization(config_path: str = "./xtclim.json"):
    # Configuration file
    config = cp.ConfigParser()
    config.read(config_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Mean-Squared Error as the average difference between the pixels
    # in the original image vs. the reconstructed one
    criterion = nn.MSELoss()
    # pixel-wise MSE loss
    pixel_wise_criterion = nn.MSELoss(reduction="none")

    return device, criterion, pixel_wise_criterion
