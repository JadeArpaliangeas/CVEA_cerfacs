import configparser as cp
import yaml
import json
from operator import add

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from itwinai.plugins.xtclim.src import model
from itwinai.plugins.xtclim.src.engine import evaluate
from itwinai.plugins.xtclim.src.initialization import initialization


def anomaly2(config_path="./config.yaml", input_path="./inputs", output_path="./outputs"):
    # Configuration file

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # pick the season to study among:
    # '' (none, i.e. full dataset), 'winter_', 'spring_', 'summer_', 'autumn_'
    seasons = config["GENERAL"]["seasons"]
    n_memb = config["TRAIN"]["n_memb"]

    # choose wether to evaluate train and test data, and/or projections
    past_evaluation = config["MODEL"]["past_evaluation"]
    future_evaluation = config["MODEL"]["future_evaluation"]
    def str_to_bool(v):
        return str(v).lower() == "true"

    past_evaluation = str_to_bool(past_evaluation)
    future_evaluation = str_to_bool(future_evaluation)
    print("future_evaluation =", future_evaluation, type(future_evaluation))

    # number of evaluations for each dataset
    n_avg = int(config["MODEL"]["n_avg"])
    # n_avg = 20

    device, criterion, pixel_wise_criterion = initialization()

    if past_evaluation:

        for season in seasons:
            # load previously trained model
            cvae_model = model.ConvVAE().to(device)
            cvae_model.load_state_dict(
                torch.load(output_path + f"/cvae_model_MSEloss_{season}_3d_{n_memb}memb_mod1.pth")
            )

            # train set and data loader
            train_time = pd.read_csv(input_path + f"/dates_train_{season}_data.csv")
            train_data = np.load(
                input_path + f"/preprocessed_3d_train_{season}_data_{n_memb}memb_mod1.npy"
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
            test_data = np.load(input_path + f"/preprocessed_3d_test_{season}_data_{n_memb}memb_mod1.npy")
            n_test = len(test_data)
            testset = [
                (torch.from_numpy(np.reshape(test_data[i], (3, 32, 32))), test_time["0"][i])
                for i in range(n_test)
            ]
            testloader = DataLoader(testset, batch_size=1, shuffle=False)

            # average over a few iterations
            # for a better reconstruction estimate
            train_avg_losses, tot_train_recon, tot_train_losses, tot_train_pixel_wise_losses = evaluate(
                cvae_model, trainloader, trainset, device, criterion, pixel_wise_criterion
            )
            #print(len(tot_train_recon))
            #print(tot_train_recon[0].shape)
            #print(tot_train_pixel_wise_losses[0])
            #print(tot_train_pixel_wise_losses[0].shape)

            test_avg_losses, tot_test_recon, tot_test_losses, tot_test_pixel_wise_losses = evaluate(
                cvae_model, testloader, testset, device, criterion, pixel_wise_criterion
            )
            #print(tot_test_recon)
            for i in range(1, n_avg):
                train_avg_loss, train_recon, train_losses, train_pixel_wise_losses = evaluate(
                    cvae_model,
                    trainloader,
                    trainset,
                    device,
                    criterion,
                    pixel_wise_criterion,
                )
                tot_train_losses = list(map(add, tot_train_losses, train_losses))
                train_avg_losses += train_avg_loss
                tot_train_recon = list(map(add, tot_train_recon, train_recon))
                tot_train_pixel_wise_losses = list(map(add, tot_train_pixel_wise_losses, train_pixel_wise_losses))
                #print(tot_train_recon[0].shape)
                #print(tot_train_recon[0])
                #tot_train_pixel_wise_losses = 

                test_avg_loss, test_recon, test_losses, test_pixel_wise_losses = evaluate(
                    cvae_model, testloader, testset, device, criterion, pixel_wise_criterion
                )
                tot_test_losses = list(map(add, tot_test_losses, test_losses))
                test_avg_losses += test_avg_loss
                tot_test_recon = list(map(add, tot_test_recon, test_recon))
                tot_test_pixel_wise_losses = list(map(add, tot_test_pixel_wise_losses, test_pixel_wise_losses))

            tot_train_losses = np.array(tot_train_losses) / n_avg
            tot_test_losses = np.array(tot_test_losses) / n_avg
            train_avg_losses = train_avg_losses / n_avg
            test_avg_losses = test_avg_losses / n_avg

            tot_train_recon = [torch.div(tensor, n_avg) for tensor in tot_train_recon]
            print(tot_train_recon[0])
            print(tot_train_recon[0].shape)
            tot_test_recon = [torch.div(tensor, n_avg) for tensor in tot_test_recon]
            tot_train_pixel_wise_losses = [torch.div(tensor, n_avg) for tensor in tot_train_pixel_wise_losses]
            tot_test_pixel_wise_losses = [torch.div(tensor, n_avg) for tensor in tot_test_pixel_wise_losses]

            pd.DataFrame(tot_train_losses).to_csv(
                output_path + f"/train_losses_MSEloss_{season}_3d_{n_memb}memb_mod1.csv"
            )
            pd.DataFrame(tot_test_losses).to_csv(
                output_path + f"/test_losses_MSEloss_{season}_3d_{n_memb}memb_mod1.csv"
            )
            torch.save(tot_train_recon, output_path + f"/tot_train_recon_MSEloss_{season}_3d_{n_memb}memb_mod1.pt")
            torch.save(tot_test_recon, output_path + f"/tot_test_recon_MSEloss_{season}_3d_{n_memb}memb_mod1.pt")
            torch.save(tot_train_pixel_wise_losses, output_path + f"/tot_train_pixel_wise_losses_MSEloss_{season}_3d_{n_memb}memb_mod1.pt")
            torch.save(tot_test_pixel_wise_losses, output_path + f"/tot_test_pixel_wise_losses_MSEloss_{season}_3d_{n_memb}memb_mod1.pt")
            
            print("Train average loss:", train_avg_losses)
            print("Test average loss:", test_avg_losses)

    if future_evaluation:
        scenarios = config["GENERAL"]["scenarios"]

        for season in seasons:
            # load previously trained model
            cvae_model = model.ConvVAE().to(device)
            cvae_model.load_state_dict(
                torch.load(output_path + f"/cvae_model_{season}_3d_{n_memb}memb_mod1.pth")
            )

            for scenario in scenarios:
                # projection set and data loader
                proj_time = pd.read_csv(input_path + f"/dates_proj_{season}_data.csv")
                proj_data = np.load(
                    input_path + f"/preprocessed_3d_proj{scenario}_{season}_data_{n_memb}memb_mod1.npy"
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
                    output_path + f"/proj{scenario}_losses_{season}_3d_{n_memb}memb_mod1.csv"
                )
                print(
                    f"SSP{scenario} Projection average loss:",
                    proj_avg_losses,
                    "for",
                    season[:-1],
                )

#if __name__ == __main__:
 #   anomaly2(config_path="/home/globc/arpaliangeas/scratchdir/config_test1.yaml", input_path="/home/globc/arpaliangeas/scratchdir/input", output_path="/home/globc/arpaliangeas/scratchdir/output")
 #   print('ok')