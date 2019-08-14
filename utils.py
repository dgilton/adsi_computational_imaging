import numpy as np

def normalize_to_01(input_image):
    color_channel_max = np.max(input_image, axis=(0,1))
    zero_one_range = np.divide(input_image, color_channel_max)
    return zero_one_range

def normalize_imagenet_method(input_image):
    mean = np.asarray([0.485, 0.456, 0.406])
    std = np.asarray([0.229, 0.224, 0.225])
    return np.divide(input_image - mean, std)

def calculate_psnr(test_image, true_image):
    if not np.array_equal(np.shape(test_image), np.shape(true_image)):
        print('The arrays must be the same size to calculate PSNR')
        exit()

    max_intensity = np.max(true_image[:])
    image_size = np.prod(np.shape(test_image))
    mse = np.sum(np.square(true_image - test_image)[:]) / image_size
    psnr = 20. * np.log10(max_intensity) - 10. * np.log10(mse)
    return psnr