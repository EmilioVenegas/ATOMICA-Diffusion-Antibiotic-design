#keep

import math
from typing import Dict

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch_scatter import scatter_add, scatter_mean

import utils


class EnVariationalDiffusion(nn.Module):
    """
    The E(n) Diffusion Module.
    """

    def __init__(
            self,
            dynamics: nn.Module, atom_nf: int, residue_nf: int,
            n_dims: int, size_histogram: Dict,
            timesteps: int = 1000, parametrization='eps',
            noise_schedule='learned', noise_precision=1e-4,
            loss_type='vlb', norm_values=(1., 1.), norm_biases=(None, 0.),
            virtual_node_idx=None):
        super().__init__()

        assert loss_type in {'vlb', 'l2'}
        self.loss_type = loss_type
        if noise_schedule == 'learned':
            assert loss_type == 'vlb', 'A noise schedule can only be learned' \
                                       ' with a vlb objective.'

        # Only supported parametrization.
        assert parametrization == 'eps'

        if noise_schedule == 'learned':
            self.gamma = GammaNetwork()
        else:
            self.gamma = PredefinedNoiseSchedule(noise_schedule,
                                                 timesteps=timesteps,
                                                 precision=noise_precision)

        # The network that will predict the denoising.
        self.dynamics = dynamics

        self.atom_nf = atom_nf
        self.residue_nf = residue_nf
        self.n_dims = n_dims
        self.num_classes = self.atom_nf

        self.T = timesteps
        self.parametrization = parametrization

        self.norm_values = norm_values
        self.norm_biases = norm_biases
        self.register_buffer('buffer', torch.zeros(1))

        #  distribution of nodes
        self.size_distribution = DistributionNodes(size_histogram)

        # indicate if virtual nodes are present
        self.vnode_idx = virtual_node_idx

        if noise_schedule != 'learned':
            self.check_issues_norm_values()

    def check_issues_norm_values(self, num_stdevs=8):
        zeros = torch.zeros((1, 1))
        gamma_0 = self.gamma(zeros)
        sigma_0 = self.sigma(gamma_0, target_tensor=zeros).item()

        # Checked if 1 / norm_value is still larger than 10 * standard
        # deviation.
        norm_value = self.norm_values[1]

        if sigma_0 * num_stdevs > 1. / norm_value:
            raise ValueError(
                f'Value for normalization value {norm_value} probably too '
                f'large with sigma_0 {sigma_0:.5f} and '
                f'1 / norm_value = {1. / norm_value}')

    def sigma_and_alpha_t_given_s(self, gamma_t: torch.Tensor,
                                  gamma_s: torch.Tensor,
                                  target_tensor: torch.Tensor):
        """
        Computes sigma t given s, using gamma_t and gamma_s. Used during sampling.
        These are defined as:
            alpha t given s = alpha t / alpha s,
            sigma t given s = sqrt(1 - (alpha t given s) ^2 ).
        """
        sigma2_t_given_s = self.inflate_batch_array(
            -torch.expm1(F.softplus(gamma_s) - F.softplus(gamma_t)), target_tensor
        )

        # alpha_t_given_s = alpha_t / alpha_s
        log_alpha2_t = F.logsigmoid(-gamma_t)
        log_alpha2_s = F.logsigmoid(-gamma_s)
        log_alpha2_t_given_s = log_alpha2_t - log_alpha2_s

        alpha_t_given_s = torch.exp(0.5 * log_alpha2_t_given_s)
        alpha_t_given_s = self.inflate_batch_array(
            alpha_t_given_s, target_tensor)

        sigma_t_given_s = torch.sqrt(sigma2_t_given_s)

        return sigma2_t_given_s, sigma_t_given_s, alpha_t_given_s



    def compute_x_pred(self, net_out, zt, gamma_t, batch_mask):
        """Commputes x_pred, i.e. the most likely prediction of x."""
        if self.parametrization == 'x':
            x_pred = net_out
        elif self.parametrization == 'eps':
            sigma_t = self.sigma(gamma_t, target_tensor=net_out)
            alpha_t = self.alpha(gamma_t, target_tensor=net_out)
            eps_t = net_out
            x_pred = 1. / alpha_t[batch_mask] * (zt - sigma_t[batch_mask] * eps_t)
        else:
            raise ValueError(self.parametrization)

        return x_pred

    def log_constants_p_x_given_z0(self, n_nodes, device):
        """Computes p(x|z0)."""

        batch_size = len(n_nodes)
        degrees_of_freedom_x = self.subspace_dimensionality(n_nodes)

        zeros = torch.zeros((batch_size, 1), device=device)
        gamma_0 = self.gamma(zeros)

        # Recall that sigma_x = sqrt(sigma_0^2 / alpha_0^2) = SNR(-0.5 gamma_0).
        log_sigma_x = 0.5 * gamma_0.view(batch_size)

        return degrees_of_freedom_x * (- log_sigma_x - 0.5 * np.log(2 * np.pi))







    def log_pN(self, N_lig, N_pocket):
        """
        Prior on the sample size for computing
        log p(x,h,N) = log p(x,h|N) + log p(N), where log p(x,h|N) is the
        model's output
        Args:
            N: array of sample sizes
        Returns:
            log p(N)
        """
        log_pN = self.size_distribution.log_prob_n1_given_n2(N_lig, N_pocket)
        return log_pN

    def delta_log_px(self, num_nodes):
        return -self.subspace_dimensionality(num_nodes) * \
               np.log(self.norm_values[0])



    def xh_given_zt_and_epsilon(self, z_t, epsilon, gamma_t, batch_mask):
        """ Equation (7) in the EDM paper """
        alpha_t = self.alpha(gamma_t, z_t)
        sigma_t = self.sigma(gamma_t, z_t)
        xh = z_t / alpha_t[batch_mask] - epsilon * sigma_t[batch_mask] / \
             alpha_t[batch_mask]
        return xh













    @staticmethod
    def gaussian_KL(q_mu_minus_p_mu_squared, q_sigma, p_sigma, d):
        """Computes the KL distance between two normal distributions.
            Args:
                q_mu_minus_p_mu_squared: Squared difference between mean of
                    distribution q and distribution p: ||mu_q - mu_p||^2
                q_sigma: Standard deviation of distribution q.
                p_sigma: Standard deviation of distribution p.
                d: dimension
            Returns:
                The KL distance
            """
        return d * torch.log(p_sigma / q_sigma) + \
               0.5 * (d * q_sigma ** 2 + q_mu_minus_p_mu_squared) / \
               (p_sigma ** 2) - 0.5 * d

    @staticmethod
    def inflate_batch_array(array, target):
        """
        Inflates the batch array (array) with only a single axis
        (i.e. shape = (batch_size,), or possibly more empty axes
        (i.e. shape (batch_size, 1, ..., 1)) to match the target shape.
        """
        target_shape = (array.size(0),) + (1,) * (len(target.size()) - 1)
        return array.view(target_shape)

    def sigma(self, gamma, target_tensor):
        """Computes sigma given gamma."""
        return self.inflate_batch_array(torch.sqrt(torch.sigmoid(gamma)),
                                        target_tensor)

    def alpha(self, gamma, target_tensor):
        """Computes alpha given gamma."""
        return self.inflate_batch_array(torch.sqrt(torch.sigmoid(-gamma)),
                                        target_tensor)

    @staticmethod
    def SNR(gamma):
        """Computes signal to noise ratio (alpha^2/sigma^2) given gamma."""
        return torch.exp(-gamma)

    def normalize(self, ligand=None, pocket=None):
        if ligand is not None:
            ligand['x'] = ligand['x'] / self.norm_values[0]
            ligand['one_hot'] = \
                (ligand['one_hot'].float() - self.norm_biases[1]) / \
                self.norm_values[1]

        if pocket is not None:
            pocket['x'] = pocket['x'] / self.norm_values[0]
            
            # The 'dynamics' object is available here. We check its class to infer the mode.
            if self.dynamics.__class__.__name__ != 'AtomicaDynamics':
                 pocket['one_hot'] = \
                    (pocket['one_hot'].float() - self.norm_biases[1]) / \
                    self.norm_values[1]
            # If it IS AtomicaDynamics, we do nothing, leaving the embeddings as they are.

        return ligand, pocket

    def unnormalize(self, x, h_cat):
        x = x * self.norm_values[0]
        h_cat = h_cat * self.norm_values[1] + self.norm_biases[1]

        return x, h_cat

    def unnormalize_z(self, z_lig, z_pocket):
        # Parse from z
        x_lig, h_lig = z_lig[:, :self.n_dims], z_lig[:, self.n_dims:]
        x_pocket, h_pocket = z_pocket[:, :self.n_dims], z_pocket[:, self.n_dims:]

        # Unnormalize
        x_lig, h_lig = self.unnormalize(x_lig, h_lig)
        x_pocket, h_pocket = self.unnormalize(x_pocket, h_pocket)
        return torch.cat([x_lig, h_lig], dim=1), \
               torch.cat([x_pocket, h_pocket], dim=1)

    def subspace_dimensionality(self, input_size):
        """Compute the dimensionality on translation-invariant linear subspace
        where distributions on x are defined."""
        return (input_size - 1) * self.n_dims

    @staticmethod
    def remove_mean_batch(x, indices):
        mean = scatter_mean(x, indices, dim=0)
        x = x - mean[indices]
        return x

    @staticmethod
    def assert_mean_zero_with_mask(x, node_mask, eps=1e-10):
        largest_value = x.abs().max().item()
        error = scatter_add(x, node_mask, dim=0).abs().max().item()
        rel_error = error / (largest_value + eps)
        assert rel_error < 1e-2, f'Mean is not zero, relative_error {rel_error}'

    @staticmethod
    def sample_center_gravity_zero_gaussian_batch(size, lig_indices,
                                                  pocket_indices):
        assert len(size) == 2
        x = torch.randn(size, device=lig_indices.device)

        # This projection only works because Gaussian is rotation invariant
        # around zero and samples are independent!
        x_projected = EnVariationalDiffusion.remove_mean_batch(
            x, torch.cat((lig_indices, pocket_indices)))
        return x_projected

    @staticmethod
    def sum_except_batch(x, indices):
        return scatter_add(x.sum(-1), indices, dim=0)

    @staticmethod
    def cdf_standard_gaussian(x):
        return 0.5 * (1. + torch.erf(x / math.sqrt(2)))

    @staticmethod
    def sample_gaussian(size, device):
        x = torch.randn(size, device=device)
        return x

class DistributionNodes(object):
    def __init__(self, histogram):
        histogram = torch.tensor(histogram).float()
        histogram = histogram + 1e-3  # for numerical stability

        self.is_2d = (len(histogram.shape) == 2)

        if self.is_2d:
            # --- 2D Histogram Logic (for conditional models) ---
            self.prob_joint = histogram / histogram.sum()
            self.prob_n2 = self.prob_joint.sum(dim=0) # Marginal probability of pocket size
            self.prob_n1 = self.prob_joint.sum(dim=1) # Marginal probability of ligand size

            # Precompute conditional probabilities P(n1 | n2)
            self.n1_given_n2 = [
                torch.distributions.Categorical(self.prob_joint[:, j] / (self.prob_n2[j] + 1e-8), validate_args=False)
                for j in range(self.prob_joint.shape[1])
            ]
            self.m = torch.distributions.Categorical(self.prob_joint.view(-1), validate_args=False)

        else:
            # --- Original 1D Histogram Logic (for unconditional models) ---
            self.prob = histogram / histogram.sum()
            self.m = torch.distributions.Categorical(self.prob, validate_args=False)

    def sample(self, n_samples=1):
        # This is for unconditional sampling, which we are not using.
        # We can just sample from the marginal ligand distribution for simplicity.
        prob_to_sample = self.prob_n1 if self.is_2d else self.prob
        n1 = torch.distributions.Categorical(prob_to_sample, validate_args=False).sample((n_samples,))
        # The node counts are 1-based, so add 1 to the sampled index.
        return n1 + 1
        
    def sample_conditional(self, n1=None, n2=None, n_samples=1):
        if not self.is_2d:
            # Fallback for 1D histogram
            print("Warning: Attempting conditional sampling with a 1D histogram. Sampling unconditionally.")
            return self.sample(n_samples)
            
        assert (n1 is None) and (n2 is not None), "Must provide n2 (pocket sizes) for conditional sampling."
        
        n1_samples = []
        for pocket_size in n2:
            # Clamp pocket size to be within the histogram's bounds
            pocket_idx = min(pocket_size.item() - 1, len(self.n1_given_n2) - 1)
            
            if pocket_idx < 0:
                print(f"Warning: Pocket size {pocket_size.item()} is invalid. Sampling from marginal.")
                dist = torch.distributions.Categorical(self.prob_n1, validate_args=False)
            else:
                dist = self.n1_given_n2[pocket_idx]

            # Sample the index (0-based) and add 1 to get the node count (1-based)
            sampled_n1 = dist.sample() + 1
            n1_samples.append(sampled_n1)
        
        n1 = torch.stack(n1_samples)
        # Return a tensor of shape [n_samples, 2] for compatibility, even though n2 was input
        n_nodes = torch.stack((n1, n2.to(n1.device)), dim=1)
        return n_nodes

    def log_prob_n1_given_n2(self, n1, n2):
        # Calculates log P(n1 | n2)
        if not self.is_2d:
            # For 1D, P(n1|n2) = P(n1)
            n1_idx = (n1 - 1).clamp(0, len(self.prob) - 1)
            return torch.log(self.prob[n1_idx.long()] + 1e-8)

        log_probs = []
        for i, c in zip(n1, n2):
            # Clamp indices to be within bounds
            n1_idx = min(i.item() - 1, self.prob_joint.shape[0] - 1)
            n2_idx = min(c.item() - 1, self.prob_joint.shape[1] - 1)

            if n1_idx < 0 or n2_idx < 0:
                log_probs.append(torch.tensor(-30.0)) # Very small log probability
                continue

            dist = self.n1_given_n2[n2_idx]
            log_probs.append(dist.log_prob(torch.tensor(n1_idx).to(dist.logits.device)))

        log_probs = torch.stack(log_probs)
        return log_probs.to(n1.device)

class PositiveLinear(torch.nn.Module):
    """Linear layer with weights forced to be positive."""

    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 weight_init_offset: int = -2):
        super(PositiveLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = torch.nn.Parameter(
            torch.empty((out_features, in_features)))
        if bias:
            self.bias = torch.nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter('bias', None)
        self.weight_init_offset = weight_init_offset
        self.reset_parameters()

    def reset_parameters(self) -> None:
        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        with torch.no_grad():
            self.weight.add_(self.weight_init_offset)

        if self.bias is not None:
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            torch.nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        positive_weight = F.softplus(self.weight)
        return F.linear(input, positive_weight, self.bias)


class GammaNetwork(torch.nn.Module):
    """The gamma network models a monotonic increasing function.
    Construction as in the VDM paper."""
    def __init__(self):
        super().__init__()

        self.l1 = PositiveLinear(1, 1)
        self.l2 = PositiveLinear(1, 1024)
        self.l3 = PositiveLinear(1024, 1)

        self.gamma_0 = torch.nn.Parameter(torch.tensor([-5.]))
        self.gamma_1 = torch.nn.Parameter(torch.tensor([10.]))
        self.show_schedule()

    def show_schedule(self, num_steps=50):
        t = torch.linspace(0, 1, num_steps).view(num_steps, 1)
        gamma = self.forward(t)
        print('Gamma schedule:')
        print(gamma.detach().cpu().numpy().reshape(num_steps))

    def gamma_tilde(self, t):
        l1_t = self.l1(t)
        return l1_t + self.l3(torch.sigmoid(self.l2(l1_t)))

    def forward(self, t):
        zeros, ones = torch.zeros_like(t), torch.ones_like(t)
        # Not super efficient.
        gamma_tilde_0 = self.gamma_tilde(zeros)
        gamma_tilde_1 = self.gamma_tilde(ones)
        gamma_tilde_t = self.gamma_tilde(t)

        # Normalize to [0, 1]
        normalized_gamma = (gamma_tilde_t - gamma_tilde_0) / (
                gamma_tilde_1 - gamma_tilde_0)

        # Rescale to [gamma_0, gamma_1]
        gamma = self.gamma_0 + (self.gamma_1 - self.gamma_0) * normalized_gamma

        return gamma


def cosine_beta_schedule(timesteps, s=0.008, raise_to_power: float = 1):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 2
    x = np.linspace(0, steps, steps)
    alphas_cumprod = np.cos(((x / steps) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    betas = np.clip(betas, a_min=0, a_max=0.999)
    alphas = 1. - betas
    alphas_cumprod = np.cumprod(alphas, axis=0)

    if raise_to_power != 1:
        alphas_cumprod = np.power(alphas_cumprod, raise_to_power)

    return alphas_cumprod


def clip_noise_schedule(alphas2, clip_value=0.001):
    """
    For a noise schedule given by alpha^2, this clips alpha_t / alpha_t-1.
    This may help improve stability during
    sampling.
    """
    alphas2 = np.concatenate([np.ones(1), alphas2], axis=0)

    alphas_step = (alphas2[1:] / alphas2[:-1])

    alphas_step = np.clip(alphas_step, a_min=clip_value, a_max=1.)
    alphas2 = np.cumprod(alphas_step, axis=0)

    return alphas2


def polynomial_schedule(timesteps: int, s=1e-4, power=3.):
    """
    A noise schedule based on a simple polynomial equation: 1 - x^power.
    """
    steps = timesteps + 1
    x = np.linspace(0, steps, steps)
    alphas2 = (1 - np.power(x / steps, power))**2

    alphas2 = clip_noise_schedule(alphas2, clip_value=0.001)

    precision = 1 - 2 * s

    alphas2 = precision * alphas2 + s

    return alphas2


class PredefinedNoiseSchedule(torch.nn.Module):
    """
    Predefined noise schedule. Essentially creates a lookup array for predefined
    (non-learned) noise schedules.
    """
    def __init__(self, noise_schedule, timesteps, precision):
        super(PredefinedNoiseSchedule, self).__init__()
        self.timesteps = timesteps

        if noise_schedule == 'cosine':
            alphas2 = cosine_beta_schedule(timesteps)
        elif 'polynomial' in noise_schedule:
            splits = noise_schedule.split('_')
            assert len(splits) == 2
            power = float(splits[1])
            alphas2 = polynomial_schedule(timesteps, s=precision, power=power)
        else:
            raise ValueError(noise_schedule)

        sigmas2 = 1 - alphas2

        log_alphas2 = np.log(alphas2)
        log_sigmas2 = np.log(sigmas2)

        log_alphas2_to_sigmas2 = log_alphas2 - log_sigmas2

        self.gamma = torch.nn.Parameter(
            torch.from_numpy(-log_alphas2_to_sigmas2).float(),
            requires_grad=False)

    def forward(self, t):
        t_int = torch.round(t * self.timesteps).long()
        return self.gamma[t_int]
