# TODO traing loop
import torch
import torch.nn as nn
import torch.optim as optim
from networks import Generator, Disclaimer, weights_init
from dataset import FaceDataset
from torchvision import transforms
import torchvision.utils as vutils
from torch.utils.data import DataLoader



dataset_dir = r'C:\Users\vincent.xu\PycharmProjects\faces'

n_vector = 100
n_feature_channels = 64
num_epochs = 400
lr = 0.0002
beta1 = 0.5
batch_size = 16
num_workers = 0

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

netD = Disclaimer(3, n_feature_channels)
netG = Generator(n_vector, n_feature_channels, 3)

netD = netD.to(device)
netG = netG.to(device)

netD.apply(weights_init)
netG.apply(weights_init)

criterion = nn.BCELoss()
fixed_noise = torch.randn(64, n_vector, 1, 1, device=device)

real_label = 1
fake_label = 0

optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))

facedataset = FaceDataset(dataset_dir=dataset_dir,
                          transform=transforms.Compose([
                              transforms.Resize(64),
                              transforms.CenterCrop(64),
                              transforms.ToTensor(),
                              transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                          ]))

dataloader = DataLoader(facedataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

# Training loop
img_list = []
G_losses = []
D_losses = []
iters = 0

print("Starting Training Loop...")

for epoch in range(num_epochs):
    for i, data in enumerate(dataloader):
        netD.zero_grad()
        real_cpu = data.to(device)
        b_size = real_cpu.size(0)
        label = torch.full((b_size,), real_label, device=device)
        output = netD(real_cpu).view(-1)
        errD_real = criterion(output, label)
        errD_real.backward()
        D_x = output.mean().item()

        noise = torch.randn(b_size, n_vector, 1, 1, device=device)
        fake = netG(noise)
        label.fill_(fake_label)
        output = netD(fake.detach()).view(-1)
        errD_fake = criterion(output, label)
        errD_fake.backward()
        D_G_z1 = output.mean().item()
        errD = errD_real + errD_fake
        optimizerD.step()

        netG.zero_grad()
        label.fill_(real_label)
        output = netD(fake).view(-1)
        errG = criterion(output, label)
        errG.backward()
        D_G_z2 = output.mean().item()
        optimizerG.step()

        if i % 50 == 0:
            print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                  % (epoch, num_epochs, i, len(dataloader),
                     errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))
            G_losses.append(errG.item())
            D_losses.append(errD.item())

        if epoch % 50 == 0:
            torch.save(netG, 'netG_{}.pth'.format(epoch))
            torch.save(netD, 'netD_{}.pth'.format(epoch))

            if (iters % 500 == 0) or ((epoch == num_epochs-1) and
                                      (i == len(dataloader)-1)):
                with torch.no_grad():
                    fake = netG(fixed_noise).detach().cpu()
                img_list.append((vutils.make_grid(fake, padding=2, normalize=True)))
            iters += 1


