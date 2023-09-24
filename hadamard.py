import torch
import torch.nn.functional as F
from math import floor, log2


# refer section 5 of the paper (pg. 8)
def hadamard_walsh_transform_(grads):
  '''This function is used to apply an inplace randomized hadamard transform to the gradient vector prior to its quantization.

  :grads: a vector with even no. of elements containing the gradients wrt all model parameters
  :return: Hadamard transformed gradient vector
  '''

 
