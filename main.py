if __name__ == "__main__":
    import torch
    import optuna
    from structure_opt import objective
    from functools import partial

    from torch.utils.data import DataLoader

    input_dim = 16
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    epochs = 100
    lr = 0.0001
    batch_size = 16
    print_every = 10
    samples = 1000 * batch_size
    train_data = 2 + 0.1 * torch.randn((samples, input_dim), device=device) + \
        torch.randn((samples, input_dim), device=device)
    val_data = 2 + 0.1 * torch.randn((int(samples/100), input_dim), device=device) + \
        torch.randn((int(samples/100), input_dim), device=device)

    train_loader = DataLoader(train_data, batch_size=batch_size)
    val_loader = DataLoader(val_data, batch_size=batch_size)

    params = {
        "lr": 0.0001,
        "epochs": 100,
        "l1": 1,
        "l2": 1,
        "l3": 1,
        "l4": 1,
        "device": torch.device("cuda" if torch.cuda.is_available() else "cpu")
    }

    structure_opt = partial(
        objective,
        n_input=input_dim,
        train_loader=train_loader,
        val_loader=val_loader,
        params=params
    )

    study = optuna.create_study(directions=["minimize", "minimize"])
    study.optimize(structure_opt, n_trials=2)




