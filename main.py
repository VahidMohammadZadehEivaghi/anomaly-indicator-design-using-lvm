if __name__ == "__main__":
    import torch
    import optuna
    from structure_opt import objective
    from functools import partial

    from torch.utils.data import DataLoader

    import json

    input_dim = 16
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    epochs = 100
    lr = 0.0001
    batch_size = 16

    samples = 1000 * batch_size
    train_data = 2 + torch.randn((samples, input_dim), device=device)
    val_data = 2 + torch.randn((int(samples/100), input_dim), device=device)

    train_loader = DataLoader(train_data, batch_size=batch_size)
    val_loader = DataLoader(val_data, batch_size=batch_size)

    params = {
        "input_dim": input_dim,
        "batch_size": batch_size,
        "x_m": 2,
        "x_s": 1,
        "lr": 0.0001,
        "epochs": 1,
        "l1": 0.5,
        "l2": 0.25,
        "l3": 0.125,
        "l4": 0.125,
        "device": str(device)
    }
    with open("params.json", "w") as p:
        json.dump(params, p, indent=4)

    structure_opt = partial(
        objective,
        n_input=input_dim,
        train_loader=train_loader,
        val_loader=val_loader,
        params=params
    )

    study = optuna.create_study(directions=["minimize", "minimize"])
    study.optimize(structure_opt, n_trials=2)

    optimum_trial = max(study.best_trials, key=lambda t: t.values[1])
    hyp_opt_result = {
        "number": optimum_trial.number,
        "params": optimum_trial.params,
        "values": optimum_trial.values
    }
    with open("hyp_opt_result.json", "w") as re:
        json.dump(hyp_opt_result, re, indent=4)







