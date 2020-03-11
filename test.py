#TODO test scripts
import torch
import random
import matplotlib.pyplot as plt
import numpy as np
# Set random seed for reproducibility
manualSeed = 999
#manualSeed = random.randint(1, 10000) # use if you want new results
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)

netG = torch.load(r'./netG_300.pth')
noise = torch.randn(1, 100, 1, 1)
output = netG(noise)
plt.axis("off")
plt.title("Fake Images")
plt.imshow(np.transpose(output[-1].detach().numpy(),(1,2,0)))
plt.show()
