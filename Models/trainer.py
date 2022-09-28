from typing import List

import torchvision
import torch
from torch.autograd import grad
from tqdm import tqdm
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from .models import Generator, Discriminator


class DataloaderGenerator:
    def __call__(self, dim: int) -> DataLoader:
        transform = transforms.Compose([
            transforms.Resize(dim + int(dim * 0.2) + 1),
            transforms.RandomCrop(dim),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5), std=(0.5))
        ])
        dataset = torchvision.datasets.MNIST('Mnist', download=False, transform=transform)
        dataloader = DataLoader(dataset, shuffle=True, batch_size=8, num_workers=4, pin_memory=True)
        return dataloader


def get_alpha(iteration: int, max_iteration: int) -> float:
    # alpha will be equal to one in half of max_iterations then be stable at one
    return min(1., iteration / (max_iteration / 2))


def train(g: Generator, d: Discriminator, dataloader_gen: DataloaderGenerator, iterations_schedule: List[int],
          log_dir: str = None):
    cum_g_losses = 0
    cum_d_losses = 0
    max_level = len(iterations_schedule)
    assert g.max_depth == d.max_depth, "generator and discriminator must have same max depth"
    assert g.max_depth >= max_level, "schedule dimension is too large for this generator"

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    g.to(device)
    d.to(device)
    g_opt = torch.optim.Adam(g.parameters(), lr=0.001, betas=(0.0, 0.99))
    d_opt = torch.optim.Adam(d.parameters(), lr=0.001, betas=(0.0, 0.99))

    for level in range(0, max_level):
        # Change the resolution of the dataloader and create a "new" dataset
        img_size = (2 ** level) * 8
        dataloader = dataloader_gen(img_size)
        dataiter = iter(dataloader)
        # Create the progression bar for this level
        pbar = tqdm(range(iterations_schedule[level]), desc=f"Level[{level}]: img_size={img_size}x{img_size}")
        # Train at the current resolution
        for i in pbar:
            alpha = get_alpha(i, iterations_schedule[level])
            # Train discriminator
            d.zero_grad()

            # Real data
            try:
                real_img, _ = next(dataiter)
            except (OSError, StopIteration):
                dataiter = iter(dataloader)
                real_img, _ = next(dataiter)
            real_img = real_img.to(device)
            batch_size = real_img.size(0)
            real_predict = d(real_img, level=level, alpha=alpha).mean()
            real_predict = real_predict - 0.001 * (real_predict ** 2)

            # Fake data
            z = torch.randn(batch_size, g.input_size, 1, 1, device=device)
            fake_img = g(z, level=level, alpha=alpha)
            fake_predict = d(fake_img.detach(), level=level, alpha=alpha).mean()

            # Gradient penalty for improved stability
            beta = torch.rand(batch_size, 1, 1, 1, device=device)
            image_mix = real_img.data + beta * (fake_img.detach().data - real_img.data)
            image_mix.requires_grad = True
            mix_predict = d(image_mix, level=level, alpha=alpha)
            mix_grad = grad(outputs=mix_predict, inputs=image_mix,
                            grad_outputs=torch.ones(mix_predict.size(), device=device),
                            create_graph=True, retain_graph=True, only_inputs=True)[0].view(batch_size, -1)
            gp = 10 * ((mix_grad.view(mix_grad.size(0), -1).norm(2, dim=1) - 1) ** 2).mean()

            # Calculate loss, propagate back and step
            d_loss = fake_predict - real_predict + gp
            d_loss.backward()
            d_opt.step()
            cum_d_losses += d_loss.item()

            # Train generator
            g.zero_grad()
            fake_predict = d(fake_img, level=level, alpha=alpha).mean()
            g_loss = -fake_predict
            g_loss.backward()
            g_opt.step()
            cum_g_losses += g_loss.item()

            # Save models and performance
            if log_dir is not None:
                if (i + 1) % 500 == 0:
                    # Save an image sample
                    with torch.no_grad():
                        images = g(torch.randn(5 * 10, g.input_size, 1, 1).to(device), level=level,
                                   alpha=alpha).data.cpu()
                        save_image(
                            images,
                            f'{log_dir}/images/l{level}i{i + 1}.png',
                            nrow=10,
                            normalize=True,
                            value_range=(-1, 1))

                if (i + 1) % 500 == 0:
                    # Save losses
                    log_file = log_dir + "/log.txt"
                    f = open(log_file, 'a+')
                    f.write(f"{level},{i + 1},{cum_d_losses / 500},{cum_g_losses / 500}\n")
                    f.close()
                    cum_g_losses = 0
                    cum_d_losses = 0

                if (i + 1) % 50000 == 0:
                    # Save models
                    model_dir = log_dir + "/model"
                    torch.save(g.state_dict(), f"{model_dir}/generator_model{level}")
                    torch.save(d.state_dict(), f"{model_dir}/discriminator_model{level}")
