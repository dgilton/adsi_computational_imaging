import utils, plotting_utils, corruption_models, bilateral_filtering, wavelet_denoising
import imageio, os
import numpy as np
from skimage.restoration import denoise_bilateral


def main():
    cwd = os.getcwd()
    image_folder = cwd + "/images/"

    test_image_path = image_folder + "owls.jpg"
    test_image = utils.normalize_to_01(imageio.imread(test_image_path))
    test_image = test_image[20:276,0:371, :]

    print(np.shape(test_image))
    return
    corrupted_image = corruption_models.add_gaussian_noise(test_image, sigma=0.1)
    corrupted_image_copy = np.copy(corrupted_image)
    # print(np.max(corrupted_image[:]))
    # denoised_image = bm3d.bm3d_denoise(corrupted_image, sigma=0.05)
    # denoised_image = bilateral_filtering.bilateral_filter(corrupted_image, sigma_r=0.5, sigma_d=3, filter_width=5)
    denoised_image = wavelet_denoising.waveletDenoiseColor(corrupted_image, 0.1)

    # skimage_denoised = denoise_bilateral(corrupted_image, win_size=5, sigma_color=0.5, sigma_spatial=3, multichannel=True, mode='edge')
    # plotting_utils.singleimage_color_plot(corrupted_image)
    # plotting_utils.before_after_plot(corrupted_image=corrupted_image, restored_image=test_image)
    plotting_utils.before_corrupted_after_plot(true_image=test_image,
                                               corrupted_image=corrupted_image_copy,
                                               restored_image=denoised_image)
    # plotting_utils.singleimage_color_plot(denoised_image)


if __name__=="__main__":
    main()