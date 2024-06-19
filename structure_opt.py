from models import Decomposition
from utils import train_for_one_epoch, evaluation
import torch
from fvcore.nn import FlopCountAnalysis
import mlflow


def __search_space(trial, n_hidden_layer, n_input, network_name):

    layers = [n_input]
    for l in range(n_hidden_layer):
        n_unit = trial.suggest_int(f"n_unit_{l+1}_of_{network_name}", 10, 100)
        mlflow.log_param(f"n_unit_{l+1}_of_{network_name}", n_unit)
        layers.append(n_unit)
    layers.append(n_input)
    return tuple(layers)


def search_space_instantiation(trial, n_input) -> torch.nn.Module:

    n_filter_layers = trial.suggest_int("n_filter_layers", 1, 3)
    n_noise_extractor_layers = trial.suggest_int("n_noise_extractor_layers", 1, 3)
    mlflow.log_params({
        "n_filter_layers": n_filter_layers,
        "n_noise_extractor_layers": n_noise_extractor_layers
    })
    filter_shape = __search_space(trial, n_filter_layers, n_input, "filter")
    noise_extractor_shape = __search_space(trial, n_noise_extractor_layers, n_input, "noise extractor")

    return Decomposition(
        filter_shape=filter_shape, noise_extractor_shape=noise_extractor_shape
    )


def objective(trial, n_input, train_loader, val_loader, params):

    if not hasattr(objective, "num_calls"):
        objective.num_calls = 1
    else:
        objective.num_calls += 1

    epochs = params["epochs"]

    with mlflow.start_run(run_name=f"run_{objective.num_calls}", nested=True):

        model = search_space_instantiation(trial, n_input)
        flops = FlopCountAnalysis(model.to(params["device"]), inputs=torch.randn(1, n_input).to(params["device"])).total()

        mlflow.log_metric("flops", flops)

        val_loss = 0
        for epoch in range(1, epochs + 1):
            print(f"--------------------Epoch: {epoch}/{epochs}----------------------------")
            train_loss = train_for_one_epoch(
                model, train_loader, params
            )
            val_loss = evaluation(
                model, val_loader, params
            )

        mlflow.log_metrics({
            f"train_loss": train_loss,
            f"val_loss": val_loss
        })

    return val_loss, flops
