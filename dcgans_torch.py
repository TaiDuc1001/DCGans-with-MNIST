import torch
from torch import nn, optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision.utils import make_grid

import matplotlib.pyplot as plt

# Define class models
class Discriminator(nn.Module):
  def __init__(self, channels_image, feature_d):
    super(Discriminator, self).__init__()
    self.model = nn.Sequential(
        nn.Conv2d(channels_image, feature_d, kernel_size=4, stride=2, padding=1),
        nn.LeakyReLU(0.2),
        self._block(feature_d * 1, feature_d * 2, 4, 2, 1),
        self._block(feature_d * 2, feature_d * 4, 4, 2, 1),
        self._block(feature_d * 4, feature_d * 8, 4, 2, 1),
        nn.Conv2d(feature_d * 8, 1, 4, 2, 0),
        nn.Sigmoid()
    )

  def _block(self, in_channels, out_channels, kernel_size, strides, padding):
    return nn.Sequential(
        nn.Conv2d(in_channels = in_channels,
                  out_channels = out_channels,
                  kernel_size = kernel_size,
                  stride = strides,
                  padding = padding,
                  bias = False),
        nn.BatchNorm2d(out_channels),
        nn.LeakyReLU(0.2)
    )

  def forward(self, x):
    return self.model(x)

class Generator(nn.Module):
  def __init__(self, z_dim, feature_g, channels_image):
    super(Generator, self).__init__()
    self.model = nn.Sequential(
        nn.ConvTranspose2d(z_dim, feature_g * 16, 4, 1, 0),
        nn.ReLU(),
        self._block(feature_g * 16, feature_g * 8, 4, 2, 1),
        self._block(feature_g * 8, feature_g * 4, 4, 2, 1),
        self._block(feature_g * 4, feature_g * 2, 4, 2, 1),
        nn.ConvTranspose2d(feature_g * 2, channels_image, 4, 2, 1),
        nn.Tanh()
    )

  def _block(self, in_channels, out_channels, kernel_size, stride, padding):
    return nn.Sequential(
        nn.ConvTranspose2d(in_channels = in_channels,
                           out_channels = out_channels,
                           kernel_size = kernel_size,
                           stride = stride,
                           padding = padding,
                           bias = False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU()
    )

  def forward(self, x):
    return self.model(x)

# Initialize weights for model layers
def init_weights(model):
  for m in model.modules():
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
      nn.init.normal_(m.weight.data, 0.0, 0.02)

# Hyperparameters
device = "cuda" if torch.cuda.is_available() else "cpu"
lr = 2e-4
batch_size = 128
image_size = 64
channels_image = 1
z_dim = 100
epochs = 5
feature_d = 64
feature_g = 64

# Transforms
transforms = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    transforms.Normalize([0.5 for _ in range(channels_image)], [0.5 for _ in range(channels_image)])
])

# Load dataset MNIST
dataset = datasets.MNIST(root='dataset', train=True, download=True, transform=transforms)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)


# Define variables for training
view_result_interval = 200
print_loss_interval = 5

# Create instances
dist = Discriminator(channels_image, feature_d).to(device)
gen = Generator(z_dim, feature_g, channels_image).to(device)

# Initialize weights
init_weights(dist)
init_weights(gen)

# Criterion
opt_gen = optim.Adam(gen.parameters(), lr=lr, betas=(0.5, 0.999))
opt_dist = optim.Adam(dist.parameters(), lr=lr, betas=(0.5, 0.999))
criterion = nn.BCELoss()

# Main training loop
for epoch in range(epochs):
  for batch_idx, (real, _) in enumerate(loader):
    # Set real images to gpu
    real = real.to(device)

    # Create noise
    noise = torch.randn(batch_size, z_dim, 1, 1).to(device)

    # Get fake images
    fake = gen(noise).to(device)

    # Get discriminator predictions from real and fake images
    dist_real = dist(real).reshape(-1)
    dist_fake = dist(fake).reshape(-1)

    ### TRAIN DISCRIMINATOR
    # Discriminator loss from real and fake images
    real_loss = criterion(dist_real, torch.ones_like(dist_real))
    fake_loss = criterion(dist_fake, torch.zeros_like(dist_fake))
    dist_loss = 0.5 * (real_loss + fake_loss)

    # Gradient for discriminator
    opt_dist.zero_grad()
    dist_loss.backward(retain_graph=True)
    opt_dist.step()

    ### TRAIN GENERATOR
    # Reforward noise and get fake predictions
    output = dist(fake).reshape(-1)
    gen_loss = criterion(output, torch.ones_like(output))

    # Gradient for generator
    opt_gen.zero_grad()
    gen_loss.backward()
    opt_gen.step()

    ## Print loss
    if batch_idx % print_loss_interval == 0:
      print(f"Epoch {epoch}/{epochs} Batch {batch_idx}/{len(loader)} Loss_D: {dist_loss:.4f}, Loss_G: {gen_loss:.4f}.")

    ## View generator's result
    if batch_idx % view_result_interval == 0:
      with torch.no_grad():
        fake = gen(noise).to(device)

        # Make grid of fake images
        fake_grid = make_grid(fake[:32], normalize=True)

        # Plot generated numbers
        plt.figure(figsize=(10, 5))
        plt.imshow(fake_grid.permute(1, 2, 0).cpu().numpy())
        plt.axis("off")
        plt.show()

