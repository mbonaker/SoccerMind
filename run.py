import numpy as np
import torch
import torch.nn.functional as F
import argparse
from pathlib import Path

from train import read_corpus, standardize

if __name__ == '__main__':
    # Load Teams to predict for
    parser = argparse.ArgumentParser(description='Predict the outcome of a football match.')
    parser.add_argument('team1', type=str, help='The first team')
    parser.add_argument('team2', type=str, help='The second team')
    # Load the model from the file
    parser.add_argument('model', type=Path, help='The model file')
    args = parser.parse_args()

    # Load the model
    checkpoint = torch.load(args.model)
    W1 = checkpoint['W1']
    b1 = checkpoint['b1']
    W2 = checkpoint['W2']
    b2 = checkpoint['b2']
    all_teams = checkpoint['all_teams']
    all_settings = checkpoint['all_settings']

    X_train, y_train, _, _ = read_corpus()

    team1 = args.team1
    team2 = args.team2
    setting = 'EM'
    if team1 not in all_teams:
        raise ValueError(f'Team {team1} not found in the dataset')
    if team2 not in all_teams:
        raise ValueError(f'Team {team2} not found in the dataset')

    # Find the team indices
    team1_idx = all_teams.index(team1)
    team2_idx = all_teams.index(team2)
    setting_idx = all_settings.index(setting)
    year = standardize(2024)

    X_inference = torch.zeros(X_train.shape[1], dtype=torch.float32)
    X_inference[team1_idx] = 1
    X_inference[team2_idx] = -1
    X_inference[setting_idx + len(all_teams)] = 1
    X_inference[-1] = year

    # Predict the outcome
    y_pred = torch.matmul(F.tanh(torch.matmul(X_inference, W1) + b1), W2) + b2
    delta, total = y_pred[0].item(), y_pred[1].item()

    print(f'The predicted goal difference is {delta:.2f} and the total goals are {total:.2f}')

    total = max(0, total)
    delta = max(-total, min(total, delta))

    print(f'{team1} vs. {team2} will end in a score of {(total + delta) / 2:.0f}:{(total - delta) / 2:.0f}')
