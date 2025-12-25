import math
import numbers
import torch
from torch import nn
from torch.nn import functional as F

class TimeMasking(nn.Module):
    """
    Masks contiguous temporal chunks of neural activity.
    Reference: SpecAugment / Brain-to-Text Benchmark suggestions.
    This augmentation forces the model to rely on contextual cues rather than
    memorizing specific local features.
    """
    def __init__(self, p=0.5, max_mask_len=25):
        super().__init__()
        self.p = p  # Probability of applying the mask to a batch/sample
        self.max_mask_len = max_mask_len # Maximum duration (in time bins) to mask out

    def forward(self, x):
        # Input x shape: [batch_size, time_steps, channels]
        
        # Only apply augmentation during training mode
        if not self.training:
            return x
            
        B, T, C = x.shape
        # Create a mask tensor initialized with 1s (keep data)
        mask = torch.ones_like(x)
        
        # Iterate through each sample in the batch
        for b in range(B):
            # Apply masking with probability p (e.g., 50% chance)
            if torch.rand(1) > self.p:
                continue
                
            # Randomly select the length of the mask (between 1 and max_mask_len)
            mask_len = torch.randint(1, self.max_mask_len, (1,)).item()
            
            # If the sequence is long enough, apply the mask
            if mask_len > 0 and T > mask_len:
                # Randomly select the starting position of the mask
                t_start = torch.randint(0, T - mask_len, (1,)).item()
                
                # Set the selected time chunk to 0 (masking it out)
                mask[b, t_start : t_start + mask_len, :] = 0
                
        # Apply the mask to the input data
        return x * mask
        
class WhiteNoise(nn.Module):
    def __init__(self, std=0.1):
        super().__init__()
        self.std = std

    def forward(self, x):
        noise = torch.randn_like(x) * self.std
        return x + noise

class MeanDriftNoise(nn.Module):
    def __init__(self, std=0.1):
        super().__init__()
        self.std = std

    def forward(self, x):
        _, C = x.shape
        noise = torch.randn(1, C) * self.std
        return x + noise

class GaussianSmoothing(nn.Module):
    """
    Apply gaussian smoothing on a
    1d, 2d or 3d tensor. Filtering is performed seperately for each channel
    in the input using a depthwise convolution.
    Arguments:
        channels (int, sequence): Number of channels of the input tensors. Output will
            have this number of channels as well.
        kernel_size (int, sequence): Size of the gaussian kernel.
        sigma (float, sequence): Standard deviation of the gaussian kernel.
        dim (int, optional): The number of dimensions of the data.
            Default value is 2 (spatial).
    """

    def __init__(self, channels, kernel_size, sigma, dim=2):
        super(GaussianSmoothing, self).__init__()
        if isinstance(kernel_size, numbers.Number):
            kernel_size = [kernel_size] * dim
        if isinstance(sigma, numbers.Number):
            sigma = [sigma] * dim

        # The gaussian kernel is the product of the
        # gaussian function of each dimension.
        kernel = 1
        meshgrids = torch.meshgrid(
            [torch.arange(size, dtype=torch.float32) for size in kernel_size]
        )
        for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
            mean = (size - 1) / 2
            kernel *= (
                1
                / (std * math.sqrt(2 * math.pi))
                * torch.exp(-(((mgrid - mean) / std) ** 2) / 2)
            )

        # Make sure sum of values in gaussian kernel equals 1.
        kernel = kernel / torch.sum(kernel)

        # Reshape to depthwise convolutional weight
        kernel = kernel.view(1, 1, *kernel.size())
        kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))

        self.register_buffer("weight", kernel)
        self.groups = channels

        if dim == 1:
            self.conv = F.conv1d
        elif dim == 2:
            self.conv = F.conv2d
        elif dim == 3:
            self.conv = F.conv3d
        else:
            raise RuntimeError(
                "Only 1, 2 and 3 dimensions are supported. Received {}.".format(dim)
            )

    def forward(self, input):
        """
        Apply gaussian filter to input.
        Arguments:
            input (torch.Tensor): Input to apply gaussian filter on.
        Returns:
            filtered (torch.Tensor): Filtered output.
        """
        return self.conv(input, weight=self.weight, groups=self.groups, padding="same")
