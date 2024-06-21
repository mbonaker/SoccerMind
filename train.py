import wandb
import pandas as pd
import torch
import torch.optim as optim
import numpy as np
import torch.nn.functional as F


def read_corpus():
    # Load the data
    data = pd.read_csv('~/Library/Mobile Documents/com~apple~Numbers/Documents/Fussballgeschichte.csv', sep=';', header=0, index_col=False, names=['year', 'team1', 'total', 'delta', 'setting', 'team2'])

    # One-hot encode categorical columns
    team1_cols = pd.get_dummies(data['team1'])
    team2_cols = pd.get_dummies(data['team2'])
    settings_cols = pd.get_dummies(data['setting'], dtype=int) * 2 - 1

    all_teams = list({*team1_cols.columns, *team2_cols.columns})
    all_settings = list(settings_cols.columns)
    team_participants = pd.DataFrame(columns=all_teams)
    for col in all_teams:
        team_participants[col] = pd.Series([0] * data.shape[0])
        if col in team1_cols.columns:
            team_participants.loc[team1_cols[col], col] = 1
        if col in team2_cols.columns:
            team_participants.loc[team2_cols[col], col] += -1

    # Drop categorical columns and replace with encoded
    data = pd.concat([data, team_participants, settings_cols], axis=1)
    data = data.drop(['team1', 'team2', 'setting'], axis=1)

    # Split into features and targets
    X = np.array(data[all_teams + all_settings + ['year']].values, dtype=float)
    y = np.array(data[['delta', 'total']].values, dtype=float)
    return X, y, all_teams, all_settings


def standardize(year):
    # Standardize the features
    return (year - 2023) / 2


if __name__ == '__main__':
    # Initialize Weights and Biases
    wandb.init(project="soccer-mind")

    X, y, all_teams, all_settings = read_corpus()

    X[:, -1] = standardize(X[:, -1])

    # Convert to PyTorch tensors
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)

    # Randomize the data order
    g = torch.Generator().manual_seed(42)
    perm = torch.randperm(X.shape[0], generator=g)
    X = X[perm]
    y = y[perm]

    # Training parameters
    learning_rate = 0.000002
    epochs = 1000000
    n_neurons = 3
    test_prop = 0.1

    # Split into training and testing sets
    train_size = int((1 - test_prop) * len(X))
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    # Initialize weight matrices
    W1 = torch.randn(X.shape[1], n_neurons, requires_grad=True, generator=g)
    b1 = torch.randn(n_neurons, requires_grad=True, generator=g)
    W2 = torch.randn(n_neurons, 2, requires_grad=True, generator=g)
    b2 = torch.randn(2, requires_grad=True, generator=g)

    # Initialize the optimizer
    optimizer = optim.RMSprop([W1, W2, b1, b2], lr=learning_rate)

    # Add weights, biases and other hyperparameters to wandb config
    wandb.config.learning_rate = learning_rate
    wandb.config.epochs = epochs
    wandb.config.hidden_neurons = n_neurons
    wandb.config.optimizer = 'RMSprop'
    wandb.config.activation = 'tanh'

    # Augment training data with opposite match-ups
    X_train_opposite = X_train.clone()
    X_train_opposite[:, :len(all_teams)] *= -1
    y_train_opposite = y_train.clone()
    y_train_opposite[:, 0] *= -1
    X_train = torch.cat([X_train, X_train_opposite], dim=0)
    y_train = torch.cat([y_train, y_train_opposite], dim=0)

    # Training loop
    for epoch in range(epochs):
        # Forward pass
        hidden = F.tanh(torch.mm(X_train, W1) + b1)
        output = torch.mm(hidden, W2) + b2

        # Compute loss
        loss = F.mse_loss(output, y_train)

        # Punish big weights
        total_loss = loss + 0.0001 * (W1 ** 2).sum() + 0.0001 * (W2 ** 2).sum()

        # Backward pass and optimization
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        if (epoch + 1) % 100 == 0:
            wandb.log({"Loss": loss.item(), "Epoch": epoch + 1}, step=epoch + 1)

        if (epoch + 1) % 1000 == 0:
            print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}')

            if X_test.shape[0] != 0:
                # Evaluation on test set
                with torch.no_grad():
                    hidden_test = F.tanh(torch.mm(X_test, W1) + b1)
                    output_test = torch.mm(hidden_test, W2) + b2
                    test_loss = F.mse_loss(output_test, y_test)
                    wandb.log({"Test Loss": test_loss.item()})
                    print(f'Test Loss: {test_loss.item():.4f}')

    # Save the model
    torch.save({'W1': W1, 'W2': W2, 'b1': b1, 'b2': b2, 'all_teams': all_teams, 'all_settings': all_settings}, f'model-{wandb.run.name}.pth')
