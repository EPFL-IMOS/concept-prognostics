import torch


def latent_cnn_code_generator_model(output_dim=256):
    if output_dim is None:
        output_dim = 256
    return torch.nn.Sequential(
        #torch.nn.Flatten(),
        torch.nn.Conv1d(in_channels=18, out_channels=20, kernel_size=3, padding='same'),
        torch.nn.ReLU(),
        torch.nn.Conv1d(in_channels=20, out_channels=20, kernel_size=3, padding='same'),
        torch.nn.ReLU(),
        torch.nn.Conv1d(in_channels=20, out_channels=10, kernel_size=3, padding='same'),
        torch.nn.ReLU(),
        torch.nn.Conv1d(in_channels=10, out_channels=10, kernel_size=3, padding='same'),
        torch.nn.ReLU(),
        torch.nn.Flatten(),
        torch.nn.Linear(500, output_dim),
        #torch.nn.ReLU(),
    )


# def latent_cnn_code_generator_model(output_dim=50):
#     if output_dim is None:
#         output_dim = 50
#     return torch.nn.Sequential(
#         #torch.nn.Flatten(),
#         torch.nn.Conv1d(in_channels=18, out_channels=10, kernel_size=3, padding='same'),
#         torch.nn.ReLU(),
#         torch.nn.Conv1d(in_channels=10, out_channels=10, kernel_size=3, padding='same'),
#         torch.nn.ReLU(),
#         torch.nn.Conv1d(in_channels=10, out_channels=1, kernel_size=3, padding='same'),
#         torch.nn.ReLU(),
#         torch.nn.Flatten(),
#         torch.nn.Linear(50, output_dim),
#         #torch.nn.ReLU(),
#     )


def latent_code_generator_model(output_dim=50):
    if output_dim is None:
        output_dim = 50
    return torch.nn.Sequential(
        #torch.nn.Flatten(),
        torch.nn.Linear(18, 200),
        torch.nn.ReLU(),
        torch.nn.Linear(200, 200),
        torch.nn.ReLU(),
        torch.nn.Linear(200, 200),
        torch.nn.ReLU(),
        #torch.nn.Linear(200, 200),
        #torch.nn.ReLU(),
        torch.nn.Linear(200, output_dim),
    )


def latent_mlp_code_generator_model(output_dim=256):
    if output_dim is None:
        output_dim = 256
    return torch.nn.Sequential(
        torch.nn.Linear(18, 128),
        torch.nn.ReLU(),
        torch.nn.Linear(128, output_dim),
    )
