from torch import nn
import torch
from torch.utils.data import DataLoader
from typing import Dict
import torch.distributions as tdist


class DecompositionLoss(nn.Module):
    def __init__(self, l1: int = 1, l2: int = 1, l3: int = 1, l4: int = 1, l5: int = 1):
        super(DecompositionLoss, self).__init__()
        self.l1 = l1
        self.l2 = l2
        self.l3 = l3
        self.l4 = l4
        self.l5 = l5

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
    def log_likelihood(out: torch.Tensor, out_dist: tdist.Normal) -> torch.Tensor:
        return torch.sum(out_dist.log_prob(out))

    @staticmethod
    def energy_preservation(z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        energy_z1 = torch.sum(torch.linalg.svdvals(z1) ** 2)
        energy_z2 = torch.sum(torch.linalg.svdvals(z2) ** 2)
        return (energy_z1 - energy_z2) ** 2

    @staticmethod
    def log_of_singular_values(input_: torch.Tensor) -> torch.Tensor:
        singular_values = torch.linalg.svdvals(input_)
        return torch.norm(singular_values, p=1)

    @staticmethod
    def kld_gauss(mu_q, log_var_q, mu_p, log_var_p):
        term1 = log_var_p - log_var_q - 1
        term2 = (torch.exp(log_var_q) + (mu_q - mu_p) ** 2) / torch.exp(log_var_p)
        kld = 0.5 * torch.sum(term1 + term2)
        return kld

    def forward(self,
                out: torch.Tensor,
                out_mean: torch.Tensor,
                out_log_var: torch.Tensor,
                noise_mean: torch.Tensor,
                noise_log_var: torch.Tensor,
                x: torch.Tensor) -> torch.Tensor:
        torch.manual_seed(1234)

        cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
        out_dist = tdist.Normal(out_mean, out_log_var.exp().sqrt())
        noise_dist = tdist.Normal(noise_mean, noise_log_var.exp().sqrt())

        n = tdist.Normal.rsample(noise_dist)

        # total_embedding = torch.cat((x, n), dim=1)

        log_likelihood = DecompositionLoss.log_likelihood(out, out_dist) / x.shape[0]
        cosine_similarity = torch.sum(cos(x, n))
        # energy_preservation = DecompositionLoss.energy_preservation(out, total_embedding)
        # singular_values = DecompositionLoss.log_of_singular_values(x)
        kl_term = DecompositionLoss.kld_gauss(noise_mean,
                                              noise_log_var,
                                              torch.zeros_like(noise_mean),
                                              torch.zeros_like(noise_mean))
        # print(cosine_similarity, kl_term, log_likelihood)

        loss = -self.l1 * log_likelihood + \
            torch.abs(self.l2 * cosine_similarity) + \
            self.l5 * kl_term

        return loss / x.shape[0]


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
    l5 = params["l5"]

    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = DecompositionLoss(l1=l1, l2=l2, l3=l3, l4=l4, l5=l5).to(device)
    running_loss = 0
    model.train()
    for batch, input_ in enumerate(train_loader):
        input_ = input_.to(device)

        # out = x + n
        out_mean, out_var, mean_of_noise, log_var_of_noise, x = model(input_)
        optimizer.zero_grad(set_to_none=True)
        loss_value = loss_fn(out=input_,
                             out_mean=out_mean,
                             out_log_var=out_var,
                             noise_mean=mean_of_noise,
                             noise_log_var=log_var_of_noise,
                             x=x
                             )
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
    l5 = params["l5"]

    model.to(device)
    loss_fn = DecompositionLoss(l1=l1, l2=l2, l3=l3, l4=l4, l5=l5).to(device)
    model.eval()

    running_loss = 0
    with torch.no_grad():
        for batch, input_ in enumerate(val_loader):
            input_ = input_.to(device)

            # out = x + n
            out_mean, out_var, mean_of_noise, log_var_of_noise, x = model(input_)
            loss_value = loss_fn(out=input_,
                                 out_mean=out_mean,
                                 out_log_var=out_var,
                                 noise_mean=mean_of_noise,
                                 noise_log_var=log_var_of_noise,
                                 x=x
                                 )

            running_loss += loss_value.item()

    val_loss = running_loss/len(val_loader)
    print(f"validation loss: {val_loss}")

    return val_loss
