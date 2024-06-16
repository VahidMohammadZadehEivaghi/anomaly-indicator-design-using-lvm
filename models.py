from torch import nn
import torch.distributions as tdist
import torch
from typing import Tuple


class Filter(nn.Module):

    def __init__(self, network_shape: Tuple[int, ...]):
        super(Filter, self).__init__()

        __modules = nn.ModuleList()
        input_ = network_shape[0]
        for m in network_shape[1:-1]:
            __modules.extend(
                [
                    nn.Linear(input_, m),
                    nn.ReLU()
                ]
            )
            input_ = m

        __modules.append(
            nn.Linear(input_, network_shape[-1])
        )
        self.embedding = nn.Sequential(*__modules)

    def forward(self, input_: torch.Tensor) -> torch.Tensor:
        return self.embedding(input_)


class NoiseExtractor(nn.Module):

    def __init__(self, network_shape: Tuple[int, ...]):
        super(NoiseExtractor, self).__init__()

        self.embedding_mean = Filter(network_shape)
        self.embedding_log_var = nn.Sequential(
            Filter(network_shape),
            nn.ReLU()
        )

    def forward(self, input_: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.embedding_mean(input_), self.embedding_log_var(input_)


class Decomposition(nn.Module):

    def __init__(self,
                 filter_shape: Tuple[int, ...],
                 noise_extractor_shape: Tuple[int, ...]):
        super(Decomposition, self).__init__()

        self.deterministic_part = Filter(filter_shape)
        self.stochastic_part = NoiseExtractor(noise_extractor_shape)

    def forward(self, input_: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        torch.manual_seed(42)

        # z = x + n
        x = self.deterministic_part(input_)
        mean_of_noise, log_var_of_noise = self.stochastic_part(input_)
        dist_of_noise = tdist.Normal(mean_of_noise, log_var_of_noise.exp().sqrt())
        n = tdist.Normal.rsample(dist_of_noise)
        z_hat = x + n
        return z_hat, x, n


if __name__ == "__main__":
    f_shape = (10, 100, 20)
    n_extractor_shape = (10, 50, 20)
    model = Decomposition(f_shape, n_extractor_shape)
    x = torch.rand(10, 1, 10)
    y, z, _ = model(x)
    print(z.shape)
