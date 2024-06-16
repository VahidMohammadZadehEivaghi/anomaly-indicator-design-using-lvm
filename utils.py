from torch import nn
import torch


class DecompositionLoss(nn.Module):
    def __init__(self, l1: int = 1, l2: int = 1, l3: int = 1):
        super(DecompositionLoss, self).__init__()
        self.l1 = l1
        self.l2 = l2
        self.l3 = l3

    @staticmethod
    def correlation_coefficient(z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        z1 = z1 - torch.mean(z1, dim=2)
        z2 = z2 - torch.mean(z2, dim=2)
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

    def forward(self, z_hat: torch.Tensor, z: torch.Tensor, x: torch.Tensor, n: torch.Tensor) -> torch.Tensor:
        total_embedding = torch.cat((x, n), dim=2)

        reconstruction_error = DecompositionLoss.reconstruction_error(z, z_hat)
        correlation_coefficient = DecompositionLoss.correlation_coefficient(x, n)
        energy_preservation = DecompositionLoss.energy_preservation(z, total_embedding)

        loss = self.l1 * reconstruction_error + self.l2 * correlation_coefficient + self.l3 * energy_preservation

        return loss


if __name__ == "__main__":
    x = torch.randn(10, 1, 10)
    n = torch.randn(10, 1, 10)
    z = torch.randn(10, 1, 10)
    z_hat = torch.randn(10, 1, 10)
    loss = DecompositionLoss()
    print(loss(z_hat=z_hat, z=z, x=x, n=n))
