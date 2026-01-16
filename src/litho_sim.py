"""
Differentiable Lithography Simulator

This module implements a simplified, fully differentiable lithography simulator.
It simulates the core physical processes from mask pattern to final printed pattern on wafer.

Core Components:
1. Optical Model: Simulates diffraction effects when light passes through the optical system.
   In this simplified version, Gaussian blur is used to approximate the Point Spread Function (PSF) effect.
   
2. Photoresist Model: Simulates the exposure and development process of photoresist.
   Uses Sigmoid function to simulate threshold effect, i.e., only regions with exposure exceeding
   a certain threshold will be developed.

Differentiability:
The entire simulation flow is implemented in PyTorch, with all operations supporting automatic differentiation.
This allows us to compute gradients of output (e.g., CD) with respect to input parameters (e.g., Dose, Focus),
laying the foundation for future gradient-based optimization methods.

References:
- TorchLitho: https://github.com/TorchOPC/TorchLitho
- LithoBench: https://github.com/shelljane/lithobench
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class DifferentiableLithoSim(nn.Module):
    """
    Differentiable Lithography Simulator
    
    This simulator takes process parameters (exposure Dose, Focus),
    and simulates the lithography process for a simple line pattern,
    ultimately outputting the printed Critical Dimension (CD).
    
    Attributes:
        target_cd (float): Target critical dimension (nm)
        mask_cd (float): Critical dimension on mask (nm)
        pixel_size (float): Pixel size of simulation grid (nm)
        image_size (int): Side length of simulation image (pixels)
    """
    
    def __init__(self, target_cd=45.0, mask_cd=50.0, pixel_size=2.0, image_size=128):
        super().__init__()
        self.target_cd = target_cd
        self.mask_cd = mask_cd
        self.pixel_size = pixel_size
        self.image_size = image_size
        
        # Create mask pattern (a simple vertical line)
        self.mask = self._create_line_mask()
        
    def _create_line_mask(self):
        """
        Create a simple line mask pattern
        
        Returns:
            torch.Tensor: Mask tensor of shape (1, 1, H, W)
        """
        mask = torch.zeros(1, 1, self.image_size, self.image_size)
        
        # Calculate line width in pixels on mask
        line_width_pixels = int(self.mask_cd / self.pixel_size)
        
        # Create a vertical line at image center
        center = self.image_size // 2
        half_width = line_width_pixels // 2
        
        mask[:, :, :, center - half_width : center + half_width] = 1.0
        
        return mask
    
    def _optical_model(self, mask, focus, dose):
        """
        Optical Model: Simulate the aerial image after light passes through optical system
        
        Uses Gaussian blur to approximate diffraction effects. Larger defocus (Focus) results in more blur.
        Exposure dose (Dose) acts directly as an intensity scaling factor.
        
        Args:
            mask (torch.Tensor): Mask pattern
            focus (torch.Tensor): Defocus amount (um), typically in range [-0.2, 0.2]
            dose (torch.Tensor): Exposure dose (mJ/cm^2), typically in range [15, 25]
        
        Returns:
            torch.Tensor: Aerial image
        """
        # Base blur kernel size (simulates inherent resolution limit of optical system)
        base_sigma = 3.0
        
        # Defocus increases blur (simplified model)
        # Larger absolute value of focus means larger sigma
        sigma = base_sigma + torch.abs(focus) * 10.0
        
        # Create Gaussian kernel
        kernel_size = int(6 * sigma.item()) + 1
        if kernel_size % 2 == 0:
            kernel_size += 1
        
        # Generate Gaussian kernel
        x = torch.arange(kernel_size).float() - kernel_size // 2
        gaussian_1d = torch.exp(-x**2 / (2 * sigma**2))
        gaussian_1d = gaussian_1d / gaussian_1d.sum()
        gaussian_2d = gaussian_1d.unsqueeze(0) * gaussian_1d.unsqueeze(1)
        gaussian_2d = gaussian_2d.unsqueeze(0).unsqueeze(0)
        
        # Apply Gaussian blur
        padding = kernel_size // 2
        aerial_image = F.conv2d(mask, gaussian_2d, padding=padding)
        
        # Apply dose scaling (normalized to reasonable range)
        # Assume standard dose is 20 mJ/cm^2
        dose_factor = dose / 20.0
        aerial_image = aerial_image * dose_factor
        
        return aerial_image
    
    def _resist_model(self, aerial_image, threshold=0.5):
        """
        Photoresist Model: Simulate the development process of photoresist
        
        Uses Sigmoid function to simulate threshold effect.
        
        Args:
            aerial_image (torch.Tensor): Aerial image
            threshold (float): Development threshold
        
        Returns:
            torch.Tensor: Printed image (binarized pattern)
        """
        # Use Sigmoid function for soft thresholding
        # steepness controls the steepness of the threshold
        steepness = 20.0
        printed_image = torch.sigmoid(steepness * (aerial_image - threshold))
        
        return printed_image
    
    def _measure_cd(self, printed_image):
        """
        Measure Critical Dimension (CD) of printed image
        
        Measures CD by calculating line width at center row of image.
        
        Args:
            printed_image (torch.Tensor): Printed image
        
        Returns:
            torch.Tensor: Measured CD value (nm)
        """
        # Get profile at center row of image
        center_row = self.image_size // 2
        profile = printed_image[0, 0, center_row, :]
        
        # Use soft threshold to calculate line width
        # Count pixels greater than 0.5 (using soft method to maintain differentiability)
        soft_count = torch.sum(torch.sigmoid(20 * (profile - 0.5)))
        
        # Convert to physical dimension
        cd = soft_count * self.pixel_size
        
        return cd
    
    def forward(self, dose, focus):
        """
        Forward pass: Execute complete lithography simulation flow
        
        Args:
            dose (torch.Tensor): Exposure dose (mJ/cm^2)
            focus (torch.Tensor): Defocus amount (um)
        
        Returns:
            dict: Dictionary containing the following keys:
                - 'cd': Measured critical dimension (nm)
                - 'cd_error': CD deviation from target value (nm)
                - 'aerial_image': Aerial image
                - 'printed_image': Printed image
        """
        # Ensure inputs are tensors
        if not isinstance(dose, torch.Tensor):
            dose = torch.tensor(dose, dtype=torch.float32)
        if not isinstance(focus, torch.Tensor):
            focus = torch.tensor(focus, dtype=torch.float32)
        
        # 1. Optical model: Generate aerial image
        aerial_image = self._optical_model(self.mask, focus, dose)
        
        # 2. Photoresist model: Generate printed image
        printed_image = self._resist_model(aerial_image)
        
        # 3. Measure CD
        cd = self._measure_cd(printed_image)
        
        # 4. Calculate CD error
        cd_error = torch.abs(cd - self.target_cd)
        
        return {
            'cd': cd,
            'cd_error': cd_error,
            'aerial_image': aerial_image,
            'printed_image': printed_image
        }
    
    def simulate(self, params):
        """
        Convenience interface: Accept parameter dictionary and return CD error
        
        Args:
            params (dict): Parameter dictionary containing 'dose' and 'focus'
        
        Returns:
            float: CD error value
        """
        dose = params.get('dose', 20.0)
        focus = params.get('focus', 0.0)
        
        result = self.forward(dose, focus)
        return result['cd_error'].item()


def demo():
    """
    Demonstrate basic functionality of the simulator
    """
    print("=" * 60)
    print("Differentiable Lithography Simulator Demo")
    print("=" * 60)
    
    # Create simulator
    sim = DifferentiableLithoSim(target_cd=45.0)
    
    # Test different parameter combinations
    test_params = [
        {'dose': 18.0, 'focus': 0.0},
        {'dose': 20.0, 'focus': 0.0},
        {'dose': 22.0, 'focus': 0.0},
        {'dose': 20.0, 'focus': -0.1},
        {'dose': 20.0, 'focus': 0.1},
    ]
    
    print(f"\nTarget CD: {sim.target_cd} nm")
    print("-" * 60)
    print(f"{'Dose (mJ/cm²)':<15} {'Focus (um)':<15} {'CD (nm)':<15} {'CD Error (nm)':<15}")
    print("-" * 60)
    
    for params in test_params:
        result = sim.forward(params['dose'], params['focus'])
        print(f"{params['dose']:<15.1f} {params['focus']:<15.2f} {result['cd'].item():<15.2f} {result['cd_error'].item():<15.2f}")
    
    # Demonstrate differentiability
    print("\n" + "=" * 60)
    print("Differentiability Demonstration")
    print("=" * 60)
    
    dose = torch.tensor(20.0, requires_grad=True)
    focus = torch.tensor(0.0, requires_grad=True)
    
    result = sim.forward(dose, focus)
    cd_error = result['cd_error']
    
    # Backward pass to compute gradients
    cd_error.backward()
    
    print(f"\nAt Dose={dose.item():.1f}, Focus={focus.item():.2f}:")
    print(f"  CD Error = {cd_error.item():.4f} nm")
    print(f"  ∂(CD Error)/∂Dose = {dose.grad.item():.4f}")
    print(f"  ∂(CD Error)/∂Focus = {focus.grad.item():.4f}")
    print("\nGradient information can be used for gradient-based optimization methods!")


if __name__ == "__main__":
    demo()
