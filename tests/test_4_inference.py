import os
import numpy as np
import pandas as pd
import torch

from itwinai.plugins.xtclim.src.model import ConvVAE
from itwinai.plugins.xtclim.src.trainer import TorchTrainer, TorchInference

def test_trainer():
    input_path = "mock_inputs"
    output_path = "mock_outputs"
    seasons = ["winter", "spring"]

    def generate_mock_data(path: str, seasons, modes=["train", "test"], n_memb=1, num_samples=20):
        os.makedirs(path, exist_ok=True)
        for season in seasons:
            for mode in modes:
                # Données aléatoires (2, 32, 32) -> aplaties
                data = np.random.rand(num_samples, 2 * 32 * 32).astype(np.float32)
                np.save(f"{path}/preprocessed_1d_{mode}_{season}_data_{n_memb}memb.npy", data)

                # Dates fictives
                dates = pd.date_range("2000-01-01", periods=num_samples, freq="D")
                pd.DataFrame({"date": dates.strftime("%Y-%m-%d")}).to_csv(
                    f"{path}/dates_{mode}_{season}_data_{n_memb}memb.csv", index=False
                )

    generate_mock_data(input_path, seasons)
    os.makedirs(output_path, exist_ok=True)

    trainer = TorchTrainer(
        input_path=input_path,
        output_path=output_path,
        seasons=seasons,
        epochs=5,
        lr=1e-3,
        batch_size=4,
        n_memb=1,
        beta=0.1,
        n_avg=2,
        stop_delta=1e-2,
        patience=3,
        kernel_size=4,
        init_channels=8,
        image_channels=2,
        latent_dim=16
    )
    trainer.execute()


def test_inference():
    input_path = "mock_inputs"
    output_path = "mock_outputs"
    seasons = ["winter", "spring"]
    scenarios = ["ssp245"]
    n_memb = 1
    latent_dim = 16
    init_channels = 8
    kernel_size = 4
    image_channels = 2

    def generate_mock_data(path: str, seasons, scenarios, n_memb=1, num_samples=20):
        os.makedirs(path, exist_ok=True)
        for season in seasons:
            for scenario in scenarios:
                # Données aléatoires (2, 32, 32) -> aplaties
                data = np.random.rand(num_samples, 2 * 32 * 32).astype(np.float32)
                np.save(f"{path}/preprocessed_1d_proj{scenario}_{season}_data_{n_memb}memb.npy", data)

                # Dates fictives
                dates = pd.date_range("2000-01-01", periods=num_samples, freq="D")
                pd.DataFrame({"date": dates.strftime("%Y-%m-%d")}).to_csv(
                    f"{path}/dates_proj_{season}_data_{n_memb}memb.csv", index=False
                )

    generate_mock_data(input_path, seasons, scenarios, n_memb=n_memb)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for season in seasons:
        dummy_model = ConvVAE(
            kernel_size=kernel_size,
            init_channels=init_channels,
            image_channels=image_channels,
            latent_dim=latent_dim
        ).to(device)
        torch.save(dummy_model.state_dict(), f"{output_path}/cvae_model_{season}_1d_{n_memb}memb.pth")

    inference = TorchInference(
        input_path=input_path,
        output_path=output_path,
        scenarios=scenarios,
        seasons=seasons,
        on_train_test=False,
        n_memb=n_memb,
        kernel_size=kernel_size,
        init_channels=init_channels,
        image_channels=image_channels,
        latent_dim=latent_dim
    )
    inference.execute()

    for season in seasons:
        for scenario in scenarios:
            expected_path = f"{output_path}/proj{scenario}_loss_indiv_{season}_1d_{n_memb}memb.csv"
            assert os.path.exists(expected_path), f"{expected_path} not found!"
