import torch
from scipy.stats import beta

import torch.distributions.beta as beta




# beta.cdf(0)

toy_x = torch.rand((50,10))
toy_y = torch.rand((50,4))

poopie = torch.Tensor([9])
snoopie = torch.Tensor([11])

poopie *= 4
snoopie *= 5

distribution = beta.Beta(poopie,snoopie)
print(distribution.rsample().requires_grad)