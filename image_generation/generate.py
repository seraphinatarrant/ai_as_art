import argparse

import torch
import torchvision.utils as vutils

from utils.gan_utils import Generator
from utils.general_utils import read_yaml_config


def setup_argparse():
    p = argparse.ArgumentParser()
    p.add_argument('-c', dest='config_file', default='config/gan_e88.yaml',
                   help='a yaml config containing necessary information to load and run generation')
    return p.parse_args()

if __name__ == "__main__":
    args = setup_argparse()
    print("Reading Config...")
    config = read_yaml_config(args.config_file)
    generator = config["generator"]
    ngpu, nc, nz, ngf = config.get("ngpu", 0), config.get("nc", 3), config['nz'], config['ngf']
    num_samples, output_dir = config.get("num_samples", 10), config.get("output_dir", ".")

    print("Loading Model...")
    device = torch.device("cuda:0" if ngpu > 0 else "cpu")
    netG = Generator(ngpu, nc, nz, ngf).to(device)
    if not ngpu:
        netG.load_state_dict(torch.load(generator, map_location = torch.device('cpu')))
    else:
        netG.load_state_dict(torch.load(generator))

    print("Generating {} images to {}...".format(num_samples, output_dir))
    for i in range(num_samples):
        noise = torch.randn(1, nz, 1, 1, device=device) # first arg is batch size
        gen_img = netG(noise)
        vutils.save_image(gen_img.detach(),
                          '%s/generated_image_%s.png' % (output_dir, i),
                          normalize=True)