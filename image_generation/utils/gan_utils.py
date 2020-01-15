from math import log2
import torch.nn as nn


# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


class Generator(nn.Module):
    def __init__(self, ngpu, nc, nz, ngf, img_sz=128):  # GPUs, num colors (2 or 3), dim of Z init vector, dim of generator
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.nc = nc
        self.ngf = ngf
        self.nz = nz
        self.img_sz = img_sz
        self._init_main()

    def _init_main(self):
        # one less layer than powers of 2 of the image size, and first and last layer are different. Reverse of the discriminator (since it's output is disc input)
        num_extra_convolutions = int(log2(self.img_sz)) - 2
        conv_seq = []
        for conv_layer in range(num_extra_convolutions, -1, -1):
            if conv_layer == num_extra_convolutions:  # first layer takes noise vector z
                conv_seq.extend([nn.ConvTranspose2d(self.nz, self.ngf * pow(2, conv_layer-1), 4, 1, 0, bias=False),
                                 nn.BatchNorm2d(self.ngf * pow(2, conv_layer-1)),
                                 nn.ReLU(True)])
            elif conv_layer == 0: # last layer goes to colour channels and has Tanh
                conv_seq.extend([nn.ConvTranspose2d(self.ngf, self.nc, 4, 2, 1, bias=False),
                                nn.Tanh()])
            else:
                conv_seq.extend([nn.ConvTranspose2d(self.ngf * pow(2, conv_layer),
                                                    self.ngf * pow(2, conv_layer-1), 4, 2, 1, bias=False),
                                 nn.BatchNorm2d(self.ngf * pow(2, conv_layer-1)),
                                 nn.ReLU(True)])

        self.main = nn.Sequential(*conv_seq)


    def forward(self, input):
        if input.is_cuda and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)
        return output


class Discriminator(nn.Module):
    def __init__(self, ngpu, nc, ndf, img_sz=128): # GPUs, num colors (2 or 3), dim of discriminator
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.nc = nc
        self.ndf = ndf
        self.img_sz = img_sz
        self._init_main()

    def _init_main(self):
        # one less layer than powers of 2 of the image size, and first and last layer are different
        num_extra_convolutions = int(log2(self.img_sz)) - 2
        # init first one as not dependent on sequence
        conv_seq = [nn.Conv2d(self.nc, self.ndf, 4, stride=2, padding=1, bias=False),
                    nn.LeakyReLU(0.2, inplace=True)]  # first layer takes num color channels and has no batch norm

        for conv_layer in range(num_extra_convolutions):
            if conv_layer == num_extra_convolutions - 1:  # last layer has just a conv and a sigmoid
                conv_seq.extend([nn.Conv2d(self.ndf * pow(2, conv_layer), 1, 4, stride=1, padding=0, bias=False),
                                 nn.Sigmoid()])
            else:
                conv_seq.extend([nn.Conv2d(self.ndf * pow(2, conv_layer), self.ndf * pow(2, conv_layer + 1), 4, stride=2, padding=1, bias=False),
                                 nn.BatchNorm2d(self.ndf * pow(2, conv_layer + 1)),
                                 nn.LeakyReLU(0.2, inplace=True)])
        self.main = nn.Sequential(*conv_seq)




    def forward(self, input):
        if input.is_cuda and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)

        return output.view(-1, 1).squeeze(1)