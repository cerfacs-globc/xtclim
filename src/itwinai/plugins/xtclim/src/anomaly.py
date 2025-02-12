import configparser as cp
import json
from operator import add

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from itwinai.plugins.xtclim.src import model
from itwinai.plugins.xtclim.src.engine import evaluate
from itwinai.plugins.xtclim.src.initialization import initialization


def anomaly(config_path="./xtclim.json", input_path="./input", output_path="./outputs"):
    # Configuration file
    config = cp.ConfigParser()
    config.read(config_path)

    # pick the season to study among:
    # '' (none, i.e. full dataset), 'winter_', 'spring_', 'summer_', 'autumn_'
    seasons = json.loads(config.get("GENERAL", "seasons"))

    # choose wether to evaluate train and test data, and/or projections
    past_evaluation = config.get("MODEL", "past_evaluation")
    future_evaluation = config.get("MODEL", "future_evaluation")

    # number of evaluations for each dataset
    n_avg = config.getint("MODEL", "n_avg")
    # n_avg = 20

    device, criterion, pixel_wise_criterion = initialization(self.config_path)

    if past_evaluation:
        for season in seasons:
            # load previously trained model
            cvae_model = model.ConvVAE().to(device)
            cvae_model.load_state_dict(
                torch.load(output_path + f"/cvae_model_{season}_1d.pth")
            )

            # train set and data loader
            train_time = pd.read_csv(input_path + f"/dates_train_{season}_data.csv")
            train_data = np.load(
                input_path + f"/preprocessed_1d_train_{season}_data_allssp.npy"
            )
            n_train = len(train_data)
            trainset = [
                (
                    torch.from_numpy(np.reshape(train_data[i], (3, 32, 32))),
                    train_time["0"][i],
                )
                for i in range(n_train)
            ]
            trainloader = DataLoader(trainset, batch_size=1, shuffle=False)

            # test set and data loader
            test_time = pd.read_csv(input_path + f"/dates_test_{season}_data.csv")
            test_data = np.load(input_path + f"/preprocessed_1d_test_{season}_data_allssp.npy")
            n_test = len(test_data)
            testset = [
                (torch.from_numpy(np.reshape(test_data[i], (3, 32, 32))), test_time["0"][i])
                for i in range(n_test)
            ]
            testloader = DataLoader(testset, batch_size=1, shuffle=False)

            # average over a few iterations
            # for a better reconstruction estimate
            train_avg_losses, _, tot_train_losses, _ = evaluate(
                cvae_model, trainloader, trainset, device, criterion, pixel_wise_criterion
            )
            test_avg_losses, _, tot_test_losses, _ = evaluate(
                cvae_model, testloader, testset, device, criterion, pixel_wise_criterion
            )
            for i in range(1, n_avg):
                train_avg_loss, _, train_losses, _ = evaluate(
                    cvae_model,
                    trainloader,
                    trainset,
                    device,
                    criterion,
                    pixel_wise_criterion,
                )
                tot_train_losses = list(map(add, tot_train_losses, train_losses))
                train_avg_losses += train_avg_loss
                test_avg_loss, _, test_losses, _ = evaluate(
                    cvae_model, testloader, testset, device, criterion, pixel_wise_criterion
                )
                tot_test_losses = list(map(add, tot_test_losses, test_losses))
                test_avg_losses += test_avg_loss
            tot_train_losses = np.array(tot_train_losses) / n_avg
            tot_test_losses = np.array(tot_test_losses) / n_avg
            train_avg_losses = train_avg_losses / n_avg
            test_avg_losses = test_avg_losses / n_avg

            pd.DataFrame(tot_train_losses).to_csv(
                output_path + f"/train_losses_{season}_1d_allssp.csv"
            )
            pd.DataFrame(tot_test_losses).to_csv(
                output_path + f"/test_losses_{season}_1d_allssp.csv"
            )
            print("Train average loss:", train_avg_losses)
            print("Test average loss:", test_avg_losses)

    if future_evaluation:
        scenarios = json.loads(config.get("GENERAL", "scenarios"))
        for season in seasons:
            # load previously trained model
            cvae_model = model.ConvVAE().to(device)
            cvae_model.load_state_dict(
                torch.load(output_path + f"/cvae_model_{season}_1d.pth")
            )

            for scenario in scenarios:
                # projection set and data loader
                proj_time = pd.read_csv(input_path + f"/dates_proj_{season}_data.csv")
                proj_data = np.load(
                    input_path + f"/preprocessed_1d_proj{scenario}_{season}_data_allssp.npy"
                )
                n_proj = len(proj_data)
                projset = [
                    (
                        torch.from_numpy(np.reshape(proj_data[i], (3, 32, 32))),
                        proj_time["0"][i],
                    )
                    for i in range(n_proj)
                ]
                projloader = DataLoader(projset, batch_size=1, shuffle=False)

                # get the losses for each data set
                # on various experiments to have representative statistics
                proj_avg_losses, _, tot_proj_losses, _ = evaluate(
                    cvae_model, projloader, projset, device, criterion, pixel_wise_criterion
                )

                for i in range(1, n_avg):
                    proj_avg_loss, _, proj_losses, _ = evaluate(
                        cvae_model,
                        projloader,
                        projset,
                        device,
                        criterion,
                        pixel_wise_criterion,
                    )
                    tot_proj_losses = list(map(add, tot_proj_losses, proj_losses))
                    proj_avg_losses += proj_avg_loss

                tot_proj_losses = np.array(tot_proj_losses) / n_avg
                proj_avg_losses = proj_avg_losses / n_avg

                # save the losses time series
                pd.DataFrame(tot_proj_losses).to_csv(
                    output_path + f"/proj{scenario}_losses_{season}_1d_allssp.csv"
                )
                print(
                    f"SSP{scenario} Projection average loss:",
                    proj_avg_losses,
                    "for",
                    season[:-1],
                )
