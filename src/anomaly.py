import torch
import numpy as np
import pandas as pd

import model
from torch.utils.data import DataLoader
from engine import evaluate
from initialization import device, beta, criterion


# projection set and data loader
proj_time = pd.read_csv("../input/dates_proj_data.csv")
proj_data = np.load("../input/preprocessed_3d_proj_data.npy")
n_proj = len(proj_data)
projset = [ ( torch.from_numpy(np.reshape(proj_data[i], (3, 32, 32))), 
                proj_time['0'][i] ) for i in range(n_proj) ]
projloader = DataLoader(
    projset, batch_size=1, shuffle=False
)

# train set and data loader
train_time = pd.read_csv("../input/dates_train_data.csv")
train_data = np.load("../input/preprocessed_3d_train_data.npy")
n_train = len(train_data)
trainset = [ ( torch.from_numpy(np.reshape(train_data[i], (3, 32, 32))), 
                train_time['0'][i] ) for i in range(n_train) ]
trainloader = DataLoader(
    trainset, batch_size=1, shuffle=False
)

# test set and data loader
test_time = pd.read_csv("../input/dates_test_data.csv")
test_data = np.load("../input/preprocessed_3d_test_data.npy")
n_test = len(test_data)
testset = [ ( torch.from_numpy(np.reshape(test_data[i], (3, 32, 32))), 
                test_time['0'][i] ) for i in range(n_test) ]
testloader = DataLoader(
    testset, batch_size=1, shuffle=False
)


# load previously trained model
cvae_model = model.ConvVAE().to(device)
cvae_model.load_state_dict(torch.load('../outputs/cvae_model_3d.pth'))

# get the losses for each data set
proj_avg_loss, proj_recon_images, proj_losses = evaluate(cvae_model, projloader, projset, 
                                          device, criterion, beta)
train_avg_loss, train_recon_images, train_losses = evaluate(cvae_model, trainloader, trainset, 
                                          device, criterion, beta)
test_avg_loss, test_recon_images, test_losses = evaluate(cvae_model, testloader, testset, 
                                          device, criterion, beta)

# save the losses time series
pd.DataFrame(proj_losses).to_csv("../outputs/proj_losses_3d_.csv")
pd.DataFrame(proj_losses).to_csv("../outputs/train_losses_3d_.csv")
pd.DataFrame(proj_losses).to_csv("../outputs/test_losses_3d_.csv")

print('Projection average loss:', proj_avg_loss)
print('Train average loss:', train_avg_loss)
print('Test average loss:', test_avg_loss)