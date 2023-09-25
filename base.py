from abc import ABC, abstractmethod
from typing import Sequence

class Transform(ABC):
  @abstractmethod
  def forward(self, x):
    '''
    :param x: The tensor to be transformed
    :return: (tx, info) The transformed tensor tx and supplementary info to be used for reverse transformation by backward
    '''

  def backward(self, tx, info):
    '''
    :param tx: The transformed tensor to be reverse transformed
    :info: supplementary info to be used for reverse transformation
    :returns: an approx of the original tensor x and additional stats related to the transformation
    '''
    return tx, None # defaults to no-op

  def roundtrip(self, x):
    return self.backward(*self.forward(x))


    

    
  
