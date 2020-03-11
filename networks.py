import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, vector_length, n_features_in_generator, output_channel):
        super(Generator, self).__init__()
        self.network = nn.Sequential(
            nn.ConvTranspose2d(vector_length, n_features_in_generator * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(n_features_in_generator * 8),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(n_features_in_generator * 8, n_features_in_generator * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(n_features_in_generator * 4),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(n_features_in_generator * 4, n_features_in_generator * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(n_features_in_generator * 2),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(n_features_in_generator * 2, n_features_in_generator, 4, 2, 1, bias=False),
            nn.BatchNorm2d(n_features_in_generator),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(n_features_in_generator, output_channel, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, input):
        return self.network(input)


class Disclaimer(nn.Module):
    def __init__(self, n_input_channel, n_feature_disclaimer):
        super(Disclaimer, self).__init__()
        self.network = nn.Sequential(
            nn.Conv2d(n_input_channel, n_feature_disclaimer, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(n_feature_disclaimer, n_feature_disclaimer * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(n_feature_disclaimer * 2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(n_feature_disclaimer * 2, n_feature_disclaimer * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(n_feature_disclaimer * 4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(n_feature_disclaimer * 4, n_feature_disclaimer * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(n_feature_disclaimer * 8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(n_feature_disclaimer * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.network(input)

if __name__ == '__main__':
    import torch
    netD = Disclaimer(3, 64)
    netG = Generator(100, 64, 3)
    input_G = torch.zeros(size=(4, 100, 1, 1))
    input_D = torch.ones(size=(4, 3, 64, 64))
    out_G = netG(input_G)
    out_D = netD(input_D)







#TODO Generator network and Disclaimer network