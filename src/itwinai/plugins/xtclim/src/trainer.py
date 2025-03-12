# This script trains the Convolutional Variational Auto-Encoder (CVAE)
# network on preprocessed CMIP6 Data


"""
from codecarbon import EmissionsTracker

# Instantiate the tracker object
tracker = EmissionsTracker(
    output_dir="../code_carbon/",  # define the directory where to write the emissions results
    output_file="emissions.csv",  # define the name of the file containing the emissions
    results
    # log_level='error' # comment out this line to see regular output
)
tracker.start()
"""

from typing import List

import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.utils import make_grid

from itwinai.components import Trainer, monitor_exec
from itwinai.plugins.xtclim.src import model
from itwinai.plugins.xtclim.src.engine import train, validate
from itwinai.plugins.xtclim.src.initialization import initialization
from itwinai.plugins.xtclim.src.utils import save_loss_plot


class TorchTrainer(Trainer):
    def __init__(
        self,
        input_path: str,
        output_path: str,
        seasons: List[str],
        epochs: int = 100,
        lr: float = 1e-3,
        batch_size: int = 64,
        n_memb: int = 1,
        beta: float = 0.1,
        n_avg: int = 20,
        stop_delta: float = 1e2,
        patience: int = 15,
        early_count: int = 0,
        old_valid_loss: float = 0.0,
        min_valid_epoch_loss: float = 100.0,
        kernel_size: int = 4,
        init_channels: int = 8,
        image_channels: int = 2,
        latent_dim: int = 128,
    ):
        super().__init__()
        self.input_path = input_path
        self.output_path = output_path
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.stop_delta = stop_delta
        self.patience = patience
        self.early_count = early_count
        self.old_valid_loss = old_valid_loss
        self.beta = beta
        self.n_avg = n_avg
        self.seasons = seasons
        self.n_memb = n_memb
        self.min_valid_epoch_loss = min_valid_epoch_loss
        # Model parameters
        self.kernel_size = kernel_size
        self.init_channels = init_channels
        self.image_channels = image_channels
        self.latent_dim = latent_dim

    @monitor_exec
    def execute(self):
        device, criterion, _ = initialization()

        for season in self.seasons:
            print(f"Training season: {season}")
            
            # initialize the model
            cvae_model = model.ConvVAE(
                kernel_size=self.kernel_size,
                init_channels=self.init_channels,
                image_channels=self.image_channels,
                latent_dim=self.latent_dim,
            ).to(device)
            optimizer = optim.Adam(cvae_model.parameters(), lr=self.lr)

            # load training set and train data
            train_time = pd.read_csv(
                self.input_path + f"/dates_train_{season}_data_{self.n_memb}memb.csv"
            )
            train_data = np.load(
                self.input_path + f"/preprocessed_1d_train_{season}_data_{self.n_memb}memb.npy"
            )
            n_train = len(train_data)
            trainset = [
                (torch.from_numpy(np.reshape(train_data[i], (2, 32, 32))), train_time["0"][i])
                for i in range(n_train)
            ]
            # load train set, shuffle it, and create batches
            trainloader = DataLoader(trainset, batch_size=self.batch_size, shuffle=True)

            # load validation set and validation data
            test_time = pd.read_csv(
                self.input_path + f"/dates_test_{season}_data_{self.n_memb}memb.csv"
            )
            test_data = np.load(
                self.input_path + f"/preprocessed_1d_test_{season}_data_{self.n_memb}memb.npy"
            )
            n_test = len(test_data)
            testset = [
                (torch.from_numpy(np.reshape(test_data[i], (2, 32, 32))), test_time["0"][i])
                for i in range(n_test)
            ]
            testloader = DataLoader(testset, batch_size=self.batch_size, shuffle=False)

            # a list to save all the reconstructed images in PyTorch grid format
            grid_images = []
            # a list to save the loss evolutions
            train_loss = []
            valid_loss = []

            for epoch in range(self.epochs):
                print(f"Epoch {epoch + 1} of {self.epochs}")

                # train the model
                train_epoch_loss = train(
                    cvae_model, trainloader, trainset, device, optimizer, criterion, self.beta
                )

                # evaluate the model on the test set
                valid_epoch_loss, recon_images = validate(
                    cvae_model, testloader, testset, device, criterion, self.beta
                )

                # keep track of the losses
                train_loss.append(train_epoch_loss)
                valid_loss.append(valid_epoch_loss)

        # save the reconstructed images from the validation loop
        # save_reconstructed_images(recon_images, epoch+1, season, self.output_path)

        # convert the reconstructed images to PyTorch image grid format
        image_grid = make_grid(recon_images.detach().cpu())
        grid_images.append(image_grid)
        # save one example of reconstructed image before and after training

        # if epoch == 0 or epoch == self.epochs-1:
        #    save_ex(recon_images[0], epoch, season, self.output_path)

        # decreasing learning rate
        if (epoch + 1) % 20 == 0:
            self.lr /= 5

        # -------

        # early stopping to avoid overfitting
        #        if (
        #            epoch > 1
        #            and (old_valid_loss - valid_epoch_loss) / old_valid_loss < stop_delta
        #        ):
        # if the marginal improvement in validation loss is too small
        #            early_count += 1

        # if early_count > patience:
        # if too small improvement for a few epochs in a row, stop learning
        #        save_ex(recon_images[0], epoch, season, self.output_path)
        # break

        #        else:
        # if the condition is not verified anymore, reset the count
        #            early_count = 0
        #        old_valid_loss = valid_epoch_loss

        # ---------------

        # save best model
        if valid_epoch_loss < self.min_valid_epoch_loss:
            self.min_valid_epoch_loss = valid_epoch_loss
            torch.save(
                cvae_model.state_dict(),
                self.output_path + f"/cvae_model_{season}_1d_{self.n_memb}memb.pth",
            )

            print(f"Train Loss: {train_epoch_loss:.4f}")
            print(f"Val Loss: {valid_epoch_loss:.4f}")

            save_loss_plot(train_loss, valid_loss, season, self.output_path)
            # save the loss evolutions
            pd.DataFrame(train_loss).to_csv(
                self.output_path + f"/train_loss_indiv_{season}_1d_{self.n_memb}memb.csv"
            )
            pd.DataFrame(valid_loss).to_csv(
                self.output_path + f"/test_loss_indiv_{season}_1d_{self.n_memb}memb.csv"
            )

        # emissions = tracker.stop()
        # print(f"Emissions from this training run: {emissions:.5f} kg CO2eq")
