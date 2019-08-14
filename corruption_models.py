import numpy as np
from numpy import random
from scipy.fftpack import fft2, ifft2, fftshift, ifftshift

def add_gaussian_noise(input_image, sigma):
    noisy_image = input_image + np.random.normal(loc=0.0,scale=sigma, size=np.shape(input_image))
    return np.clip(noisy_image,a_min=0,a_max=1)

def get_inpainting_map(image_size, p):
    pixel_conservation_map = random.binomial(n=1, p=p, size=image_size)
    pixel_deletion_map = np.logical_not(pixel_conservation_map)
    return pixel_deletion_map

def apply_inpainting_map(input_image, pixel_deletion_map):
    corrupted_image = np.copy(input_image)
    corrupted_image[pixel_deletion_map==1] = 0.0
    return corrupted_image

def blur_and_add_noise(input_image, blur_kernel, noise_sigma):
    corrupted_image = np.real(ifft2(fft2(input_image) * fft2(fftshift(blur_kernel))))
    corrupted_image = corrupted_image + np.random.normal(loc=0.0, scale=noise_sigma, size=np.shape(corrupted_image))
    return corrupted_image

def gaussian_blur(input_image, blur_kernel):
    corrupted_image = np.real(ifft2(fft2(input_image) * fft2(fftshift(blur_kernel))))
    return corrupted_image

def gaussian_blur_gramian(input_image, blur_kernel):
    return gaussian_blur(gaussian_blur(input_image, blur_kernel), blur_kernel)

def inpainting_gramian(input_image, pixel_deletion_map):
    return apply_inpainting_map(input_image, pixel_deletion_map)