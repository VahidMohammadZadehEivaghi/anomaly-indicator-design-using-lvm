from torch import nn
import torch
from torch.utils.data import DataLoader
from typing import Dict


class DecompositionLoss(nn.Module):
    def __init__(self, l1: int = 1, l2: int = 1, l3: int = 1, l4: int = 1):
        super(DecompositionLoss, self).__init__()
        self.l1 = l1
        self.l2 = l2
        self.l3 = l3
        self.l4 = l4

    @staticmethod
    def correlation_coefficient(z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        z1 = z1 - torch.mean(z1, dim=1)
        z2 = z2 - torch.mean(z2, dim=1)
        corr_ = torch.sum(z1 * z2) / (torch.sqrt(torch.sum(z1 ** 2)) * torch.sqrt(torch.sum(z2 ** 2)) + 1e-6)
        return corr_

    @staticmethod
    def reconstruction_error(z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        return nn.MSELoss()(z1, z2)

    @staticmethod
    def energy_preservation(z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        energy_z1 = torch.norm(torch.linalg.svdvals(z1), p=2)**2
        energy_z2 = torch.norm(torch.linalg.svdvals(z2), p=2)**2
        return (energy_z1 - energy_z2) ** 2

    @staticmethod
    def log_of_singular_values(input_: torch.Tensor) -> torch.Tensor:
        singular_values = torch.linalg.svdvals(input_)
        return torch.sum(1 + torch.log(singular_values ** 2))

    def forward(self, z_hat: torch.Tensor, z: torch.Tensor, x: torch.Tensor, n: torch.Tensor) -> torch.Tensor:
        total_embedding = torch.cat((x, n), dim=1)

        reconstruction_error = DecompositionLoss.reconstruction_error(z, z_hat)
        correlation_coefficient = DecompositionLoss.correlation_coefficient(x, n)
        energy_preservation = DecompositionLoss.energy_preservation(z, total_embedding)
        singular_values = DecompositionLoss.log_of_singular_values(x)
        loss = self.l1 * reconstruction_error + \
            self.l2 * correlation_coefficient + \
            self.l3 * energy_preservation + \
            self.l4 * singular_values

        return loss


def train_for_one_epoch(
        model: nn.Module,
        train_loader: DataLoader,
        params: Dict
):
    device = params["device"]
    lr = params["lr"]
    l1 = params["l1"]
    l2 = params["l2"]
    l3 = params["l3"]
    l4 = params["l4"]

    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = DecompositionLoss(l1=l1, l2=l2, l3=l3).to(device)
    running_loss = 0
    model.train()
    for batch, input_ in enumerate(train_loader):
        input_ = input_.to(device)

        # z = x + n
        z_hat, x, n = model(input_)
        optimizer.zero_grad(set_to_none=True)
        loss_value = loss_fn(z_hat=z_hat, z=input_, x=x, n=n)
        loss_value.backward()
        optimizer.step()

        running_loss += loss_value.item()

        if batch % 99 == 1:
            print(f"batch: {batch}/{len(train_loader)}, loss: {loss_value.item()}")

    return running_loss / len(train_loader)


def evaluation(
        model: nn.Module,
        val_loader: DataLoader,
        params: Dict
):
    device = params["device"]
    l1 = params["l1"]
    l2 = params["l2"]
    l3 = params["l3"]
    l4 = params["l4"]

    model.to(device)
    loss_fn = DecompositionLoss(l1=l1, l2=l2, l3=l3).to(device)
    model.eval()

    running_loss = 0
    with torch.no_grad():
        for batch, input_ in enumerate(val_loader):
            input_ = input_.to(device)

            # z = x + n
            z_hat, x, n = model(input_)
            loss_value = loss_fn(z_hat=z_hat, z=input_, x=x, n=n)

            running_loss += loss_value.item()

    val_loss = running_loss/len(val_loader)
    print(f"validation loss: {val_loss}")

    return val_loss
