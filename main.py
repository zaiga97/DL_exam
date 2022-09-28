from Models import Generator, Discriminator, train, DataloaderGenerator
import torch.backends.cudnn as cudnn
import argparse
import os
from datetime import datetime

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--depth', type=int, default=2)
    parser.add_argument('--epochs', type=int, default=10000)
    parser.add_argument('--hidden_dim', type=int, default=32)
    parser.add_argument('--latent_dim', type=int, default=64)
    opts = parser.parse_args()

    date_time = datetime.now()
    log_dir = f'./ProGAN{date_time.date()}_{date_time.hour}_{date_time.minute}'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
        os.makedirs(log_dir + '/images')
        os.makedirs(log_dir + '/model')

    g = Generator(input_size=opts.latent_dim, hidden_channels=opts.hidden_dim, nc=3)
    d = Discriminator(hidden_channels=opts.hidden_dim, nc=3)
    schedule = [opts.epochs for _ in range(opts.depth)]
    dl_g = DataloaderGenerator()
    # Improve performance by testing and using the best convolution algorithm
    cudnn.benchmark = True
    train(g, d, dl_g, schedule, log_dir)
