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
from itwinai.plugins.xtclim.src.engine import evaluate, train, validate
from itwinai.plugins.xtclim.src.initialization import initialization
from itwinai.plugins.xtclim.src.utils import save_image, save_loss_plot


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
        stop_delta: float = 1e-2,
        patience: int = 15,
        kernel_size: int = 4,
        init_channels: int = 8,
        image_channels: int = 2,
        latent_dim: int = 128,
    ):
        super().__init__()
        self.input_path = input_path
        self.output_path = output_path
        self.seasons = seasons
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.n_memb = n_memb
        self.beta = beta
        self.n_avg = n_avg
        self.stop_delta = stop_delta
        self.patience = patience
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

            # Initialize model and optimizer
            cvae_model = model.ConvVAE(
                kernel_size=self.kernel_size,
                init_channels=self.init_channels,
                image_channels=self.image_channels,
                latent_dim=self.latent_dim,
            ).to(device)
            optimizer = optim.Adam(cvae_model.parameters(), lr=self.lr)

            # Load training data
            trainset = self.load_dataset("train", season)
            trainloader = DataLoader(trainset, batch_size=self.batch_size, shuffle=True)

            # Load validation data
            testset = self.load_dataset("test", season)
            testloader = DataLoader(testset, batch_size=self.batch_size, shuffle=False)

            train_loss, valid_loss = [], []
            best_val_loss = float("inf")
            early_count = 0

            for epoch in range(self.epochs):
                print(f"Epoch {epoch + 1}/{self.epochs}")

                # Train for one epoch
                train_epoch_loss = train(cvae_model, trainloader, trainset, device, optimizer, criterion, self.beta)
                train_loss.append(train_epoch_loss)

                # Validate after training
                valid_epoch_loss, recon_images = validate(cvae_model, testloader, testset, device, criterion, self.beta)
                valid_loss.append(valid_epoch_loss)

                # Save best model and losses
                if valid_epoch_loss < best_val_loss:
                    best_val_loss = valid_epoch_loss
                    torch.save(cvae_model.state_dict(), f"{self.output_path}/cvae_model_{season}_1d_{self.n_memb}memb.pth")
                    save_loss_plot(train_loss, valid_loss, season, self.output_path)
                    pd.DataFrame(train_loss).to_csv(
                        f"{self.output_path}/train_loss_per_epoch_indiv_{season}_1d_{self.n_memb}memb.csv", index=False
                    )
                    pd.DataFrame(valid_loss).to_csv(
                        f"{self.output_path}/test_loss_per_epoch_indiv_{season}_1d_{self.n_memb}memb.csv", index=False
                    )

                # Learning rate decay
                if (epoch + 1) % 20 == 0:
                    for g in optimizer.param_groups:
                        g["lr"] /= 5

                # Early stopping
                if epoch > 0:
                    improvement = (valid_loss[-2] - valid_epoch_loss) / valid_loss[-2]
                    if improvement < self.stop_delta:
                        early_count += 1
                        if early_count > self.patience:
                            print("Early stopping triggered.")
                            break
                    else:
                        early_count = 0
            # Save final reconstructed image grid
            image_grid = make_grid(recon_images.detach().cpu())
            # Optionally save image grid if needed
            save_image(image_grid, f"{self.output_path}/recon_grid_{season}.png")
        # emissions = tracker.stop()
        # print(f"Emissions from this training run: {emissions:.5f} kg CO2eq")

    def load_dataset(self, mode, season):
        """Load dataset with given mode (train/test/proj) and season"""
        data = np.load(f"{self.input_path}/preprocessed_1d_{mode}_{season}_data_{self.n_memb}memb.npy")
        time = pd.read_csv(f"{self.input_path}/dates_{mode}_{season}_data_{self.n_memb}memb.csv")
        dataset = [(torch.from_numpy(np.reshape(data[i], (2, 32, 32))), time.iloc[i, 0]) for i in range(len(data))]
        return dataset


class TorchInference(Trainer):
    def __init__(
        self,
        input_path: str,
        output_path: str,
        scenarios: List[str],
        seasons: List[str],
        on_train_test: bool = False,
        n_memb: int = 1,
        kernel_size: int = 4,
        init_channels: int = 8,
        image_channels: int = 2,
        latent_dim: int = 128,
    ):
        super().__init__()
        self.input_path = input_path
        self.output_path = output_path
        self.scenarios = scenarios
        self.seasons = seasons
        self.n_memb = n_memb
        self.on_train_test = on_train_test
        # Model parameters
        self.kernel_size = kernel_size
        self.init_channels = init_channels
        self.image_channels = image_channels
        self.latent_dim = latent_dim

    @monitor_exec
    def execute(self):
        device, criterion, pixel_wise_criterion = initialization()
        for season in self.seasons:
            print(f"Running inference for season: {season}")
            # Load pre-trained model
            inference_model = model.ConvVAE(
                kernel_size=self.kernel_size,
                init_channels=self.init_channels,
                image_channels=self.image_channels,
                latent_dim=self.latent_dim,
            ).to(device)
            model_path = f"{self.output_path}/cvae_model_{season}_1d_{self.n_memb}memb.pth"
            inference_model.load_state_dict(torch.load(model_path, weights_only=True))
            inference_model.eval()

            if self.on_train_test:
                # Load training data
                trainset = self.load_dataset("train", season)
                dataloader = DataLoader(trainset, batch_size=1, shuffle=False)
                # Run evaluation
                val_loss, recon_images, losses, pixel_wise_losses = evaluate(
                    inference_model, dataloader, trainset, device, criterion, pixel_wise_criterion
                )
                # Save anomaly score (loss) per timestep
                pd.DataFrame(losses).to_csv(f"{self.output_path}/train_loss_indiv_{season}_1d_{self.n_memb}memb.csv")
                testset = self.load_dataset("test", season)
                dataloader = DataLoader(testset, batch_size=1, shuffle=False)
                # Run evaluation
                val_loss, recon_images, losses, pixel_wise_losses = evaluate(
                    inference_model, dataloader, testset, device, criterion, pixel_wise_criterion
                )
                # Save anomaly score (loss) per timestep
                pd.DataFrame(losses).to_csv(f"{self.output_path}/test_loss_indiv_{season}_1d_{self.n_memb}memb.csv")

            else:
                for scenario in self.scenarios:
                    # Load projection data (future climate data)
                    projset = self.load_dataset("proj", season, scenario)
                    dataloader = DataLoader(projset, batch_size=1, shuffle=False)
                    # Run evaluation
                    val_loss, recon_images, losses, pixel_wise_losses = evaluate(
                        inference_model, dataloader, projset, device, criterion, pixel_wise_criterion
                    )
                    # Save anomaly score (loss) per timestep
                    pd.DataFrame(losses).to_csv(f"{self.output_path}/proj{scenario}_loss_indiv_{season}_1d_{self.n_memb}memb.csv")
                    # Optionally, save reconstructed images
                    # image_grid = make_grid(recon_images.detach().cpu())
                    # torch.save(
                    # image_grid, f"{self.output_path}/reconstructed_grid_{season}.pt"
                    # )
                    # Ou bien, enregistrer en image avec matplotlib ou PIL

                    # Optional: Save loss plot for visual reference
                    save_loss_plot([], losses, season, self.output_path)
                    print("val_loss = ", val_loss)
            print(f"Saved inference results for {season}")

    def load_dataset(self, mode, season, scenario=""):
        """Load dataset with given mode (train/test/proj) and season"""
        data = np.load(f"{self.input_path}/preprocessed_1d_{mode}{scenario}_{season}_data_{self.n_memb}memb.npy")
        time = pd.read_csv(f"{self.input_path}/dates_{mode}_{season}_data_{self.n_memb}memb.csv")
        dataset = [(torch.from_numpy(np.reshape(data[i], (2, 32, 32))), time.iloc[i, 0]) for i in range(len(data))]
        return dataset
