import numpy as np
import pywt
from scipy import signal
from skimage import color

def is_input_valid(input_image):
    if input_image.ndim is not 2:
        print('We operate only on greyscale images.')
        return 0

    image_shape = np.shape(input_image)
    if not image_shape[0]==image_shape[1]:
        print('Please use a square image.')
        return 0
    n_levels = np.int_(np.log2(image_shape[0]))
    if not 2**n_levels == image_shape[0]:
        print('Image side length should be a power of 2')
        return 0
    return 1

def is_input_valid_color(input_image):
    image_shape = np.shape(input_image)
    if not image_shape[0] == image_shape[1]:
        print('Please use a square image.')
        return 0
    n_levels = np.int_(np.log2(image_shape[0]))
    if not 2 ** n_levels == image_shape[0]:
        print('Image side length should be a power of 2')
        return 0
    return 1

def waveletDenoiseColor(input_image, shrinkage_parameter):
    if not is_input_valid_color(input_image):
        return

    # Do denoising in ycbcr for aesthetic reasons
    ycbcr_image = color.rgb2ycbcr(input_image)
    output_image = np.zeros_like(ycbcr_image)
    # We multiply the thresholds by the new maxima for the channels
    output_image[:,:,0] = waveletDenoise(ycbcr_image[:,:,0],235.*shrinkage_parameter)
    output_image[:,:,1] = waveletDenoise(ycbcr_image[:,:,1],230.*shrinkage_parameter)
    output_image[:,:,2] = waveletDenoise(ycbcr_image[:,:,2],240.*shrinkage_parameter)

    output_image = color.ycbcr2rgb(output_image)
    return output_image


def waveletDenoise(input_image, shrinkage_parameter, thresholding='soft'):
    # if not is_input_valid(input_image):
    #     return

    max_n_levels = np.int_(np.log2(np.shape(input_image)[0]))

    wavelet_coeffs = forward_wavelet_transform(input_image, max_n_levels)

    if thresholding=='soft':
        thresholded_coeffs = soft_threshold(wavelet_coeffs, shrinkage_parameter)
    elif thresholding=='hard':
        thresholded_coeffs = hard_threshold(wavelet_coeffs, shrinkage_parameter)
    else:
        print('Thresholding must be "hard" or "soft".')

    # Alternately! Try playing with the mode parameter here. Test out 'hard', 'greater', or 'garrote'
    # thresholded_coeffs = pywt.threshold(wavelet_coeffs, shrinkage_parameter, mode='soft')

    estimated_image = inverse_wavelet_transform(thresholded_coeffs, max_n_levels)
    return estimated_image

def get_inverse_2d_haar_matrices():
    h00 = np.array([[1,1],[1,1]]) / 2.
    h01 = np.array([[1,-1],[1,-1]]) / 2.
    h10 = np.array([[1,1],[-1,-1]]) / 2.
    h11 = np.array([[1,-1],[-1,1]]) / 2.
    return [h00, h01, h10, h11]

def get_forward_2d_haar_matrices():
    h00 = np.array([[1,1],[1,1]]) / 2.
    h01 = np.array([[-1,1],[-1,1]]) / 2.
    h10 = np.array([[-1,-1],[1,1]]) / 2.
    h11 = np.array([[1,-1],[-1,1]]) / 2.
    return [h00, h01, h10, h11]

def upsample_2x_nofilter(input_data):
    upsampler = np.zeros((np.shape(input_data)[0]*2,np.shape(input_data)[1]*2))
    upsampler[::2,::2] = input_data
    return upsampler

def downsample_2x_nofilter(input_image):
    return input_image[::2,::2]

def forward_wavelet_transform(input_image, n_levels):
    if not is_input_valid(input_image):
        return
    L = np.int_(np.log2(np.shape(input_image)[0]))
    number_iterations = np.minimum(n_levels, L)
    transform = input_image

    for ii in range(number_iterations):
        transform_depth = 2**(L-ii)
        transform[:transform_depth, :transform_depth] = forward_transform_onelevel(
            transform[:transform_depth, :transform_depth])

    return transform


def forward_transform_onelevel(input_image):
    if not is_input_valid(input_image):
        return
    h00,h01,h10,h11 = get_forward_2d_haar_matrices()

    w00 = signal.convolve2d(input_image, h00, mode='full'); w00 = w00[1:,1:]
    w00 = downsample_2x_nofilter(w00)
    w01 = signal.convolve2d(input_image, h01, mode='full'); w01 = w01[1:,1:]
    w01 = downsample_2x_nofilter(w01)
    w10 = signal.convolve2d(input_image, h10, mode='full'); w10 = w10[1:,1:]
    w10 = downsample_2x_nofilter(w10)
    w11 = signal.convolve2d(input_image, h11, mode='full'); w11 = w11[1:,1:]
    w11 = downsample_2x_nofilter(w11)

    output_transform = np.block([[w00, w01],[w10,w11]])
    return output_transform

def cycle_spin_denoise(input_image, shrinkage_parameter):
    # if not is_input_valid(input_image):
    #     return

    max_n_levels = np.int_(np.log2(np.shape(input_image)[0]))
    # print(max_n_levels)
    cycle_spin_shift = np.random.randint(np.shape(input_image)[0], size=(2,))
    # Shift the image
    input_image = np.roll(input_image, cycle_spin_shift)
    wavelet_coeffs = forward_wavelet_transform(input_image, max_n_levels)
    thresholded_coeffs = soft_threshold(wavelet_coeffs, shrinkage_parameter)
    estimated_image = inverse_wavelet_transform(thresholded_coeffs, max_n_levels)
    # Un-shift the denoised image
    estimated_image = np.roll(estimated_image, -cycle_spin_shift)
    return estimated_image


def soft_threshold(theta, threshold):
    # normalized_theta = theta / np.abs(theta)
    theta_abs = np.abs(theta)
    normalized_theta = np.sign(theta)
    return normalized_theta * np.maximum(theta_abs - np.abs(threshold),0)

def hard_threshold(theta, threshold):
    theta_abs = np.abs(theta)
    normalized_theta = np.sign(theta)
    return normalized_theta * (np.clip(theta_abs,a_min=threshold, a_max=None))

def inverse_wavelet_transform(wavelet_coeffs, n_levels):
    L = np.int_(np.log2(np.shape(wavelet_coeffs)[0]))
    number_iterations = np.minimum(n_levels, L)
    reconstruction = wavelet_coeffs

    for ii in range(number_iterations,0,-1):
        inv_transform_depth = 2 ** (L - ii + 1)
        reconstruction[:inv_transform_depth, :inv_transform_depth] = inverse_transform_onelevel(
            reconstruction[:inv_transform_depth, :inv_transform_depth])

    return reconstruction

def inverse_transform_onelevel(input_data):
    if not is_input_valid(input_data):
        return
    original_size = np.shape(input_data)[0]
    upsampled = upsample_2x_nofilter(input_data)
    h00, h01, h10, h11 = get_inverse_2d_haar_matrices()

    w00 = upsampled[:original_size, :original_size]
    w00 = signal.convolve2d(w00, h00, mode='full'); w00 = w00[:-1,:-1]
    w01 = upsampled[:original_size, original_size:2*original_size]
    w01 = signal.convolve2d(w01, h01, mode='full'); w01 = w01[:-1,:-1]
    w10 = upsampled[original_size:2*original_size, :original_size]
    w10 = signal.convolve2d(w10, h10, mode='full'); w10 = w10[:-1,:-1]
    w11 = upsampled[original_size:2*original_size, original_size:2*original_size]
    w11 = signal.convolve2d(w11, h11, mode='full'); w11 = w11[:-1,:-1]

    reconstruction = w00 + w01 + w10 + w11

    return reconstruction
