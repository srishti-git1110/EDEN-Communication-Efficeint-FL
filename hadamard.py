import torch
import torch.nn.functional as F
from math import floor, log2

from base import Transform


# refer section 5 of the paper (pg. 8)
def hadamard_walsh_transform_(vec):
  '''This function is used to apply an inplace randomized hadamard transform to the gradient vector prior to its quantization.

  :grads: a vector with even no. of elements containing the gradients wrt all model parameters
  :return: Hadamard transformed gradient vector
  '''
  
  d = vec.numel()
  original_shape = vec.shape
  h = 2
  while h <= d:
    hf = h // 2
    vec = vec.view(d // h, h)
    vec[:, :hf] = vec[:, :hf] + vec[:, hf:2 * hf]
    vec[:, hf:2 * hf] = vec[:, :hf] - 2 * vec[:, hf:2 * hf]
    h *= 2
  vec *= d ** -0.5

  return vec.view(*original_shape)


def rademacher_like(x, generator):
  '''generates a tensor with the same size as x following the Rademacher distribution'''
  return 2 * torch.torch.empty_like(x).bernoulli_(generator=generator) - 1


def randomized_hadamard_transform_(x, generator):
  d = rademacher_like(x, generator)
  return hadamard_transform_(x * d)


def inverse_randomized_hadamard_transform_(tx, generator):
  d = rademacher_like(tx, generator)
  return hadamard_transform_(tx) * d


class RandomizedHadamard(Transform):
  def __init__(self, device='cpu'):
    self.prng = torch.Generator(device=device)

  def forward(self, x):
    
    


  

 
