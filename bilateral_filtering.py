import numpy as np
from skimage import color

def gaussian_filter_coeffs(size, sigma):
    x, y = np.mgrid[-size//2 + 1:size//2 + 1, -size//2 + 1:size//2 + 1]
    g = np.exp(-((x**2 + y**2)/(2.0*sigma**2)))
    return g

def gaussian_intensity_weights(center, surrounding_region, sigma):
    return np.exp(-np.square(surrounding_region - center) / (2.0*sigma**2))

def bilateral_filter_grey(input_image, sigma_r, sigma_d, filter_width):
    gaussian_dist_weights = gaussian_filter_coeffs(size=2*filter_width, sigma=sigma_d)
    image_size = np.shape(input_image)
    output_image = np.zeros_like(input_image)
    for x in range(image_size[0]):
        xrange_min = max((x - filter_width, 0))
        xrange_max = min((x + filter_width, image_size[0] - 1))
        for y in range(image_size[1]):
            yrange_min = max((y - filter_width, 0))
            yrange_max = min((y + filter_width, image_size[1] - 1))
            relevant_region = input_image[xrange_min:xrange_max, yrange_min:yrange_max]
            intensity_weights = gaussian_intensity_weights(center=input_image[x,y],
                                                           surrounding_region=relevant_region, sigma=sigma_r)
            filter_xrange = [ii-x+filter_width for ii in range(xrange_min, xrange_max)]
            filter_yrange = [ii-y+filter_width for ii in range(yrange_min, yrange_max)]
            clipped_filter = gaussian_dist_weights[filter_xrange,:][:, filter_yrange]
            weights = intensity_weights*clipped_filter
            weights_normalization_term = np.sum(weights[:])
            output_image[x,y] = np.sum((weights*relevant_region)[:]) / weights_normalization_term
    return output_image

def bilateral_filter_color(input_image, sigma_r, sigma_d, filter_width):

    gaussian_dist_weights = gaussian_filter_coeffs(size=2*filter_width, sigma=sigma_d)
    image_size = np.shape(input_image)
    output_image = np.zeros_like(input_image)
    for x in range(image_size[0]):
        xrange_min = max((x - filter_width, 0))
        xrange_max = min((x + filter_width, image_size[0] - 1))
        for y in range(image_size[1]):
            yrange_min = max((y - filter_width, 0))
            yrange_max = min((y + filter_width, image_size[1] - 1))
            relevant_region = input_image[xrange_min:xrange_max, yrange_min:yrange_max,:]

            luminance_means = relevant_region[...,0] - input_image[x,y,0]
            a_means = relevant_region[...,1] - input_image[x,y,1]
            b_means = relevant_region[...,2] - input_image[x,y,2]
            total_weights = np.square(luminance_means) + np.square(a_means) + np.square(b_means)

            intensity_weights = np.exp(-total_weights / (2 * sigma_r**2))

            filter_xrange = [ii-x+filter_width for ii in range(xrange_min, xrange_max)]
            filter_yrange = [ii-y+filter_width for ii in range(yrange_min, yrange_max)]
            clipped_filter = gaussian_dist_weights[filter_xrange,:][:, filter_yrange]
            weights = intensity_weights*clipped_filter
            weights_normalization_term = np.sum(weights[:])
            output_image[x,y,0] = np.sum((weights*relevant_region[...,0])[:]) / weights_normalization_term
            output_image[x,y,1] = np.sum((weights*relevant_region[...,1])[:]) / weights_normalization_term
            output_image[x,y,2] = np.sum((weights*relevant_region[...,2])[:]) / weights_normalization_term
    return output_image

def bilateral_filter(input_image, sigma_r, sigma_d, filter_width=5):
    if input_image.ndim==2:
        return bilateral_filter_grey(input_image, sigma_r, sigma_d, filter_width)
    elif input_image.ndim==3:
        if np.shape(input_image)[2]==3:
            return bilateral_filter_color(input_image, sigma_r, sigma_d, filter_width)
        else:
            print('This function requires 3 color channels')
            return
    else:
        print('Image must be either greyscale or rgb')
        return
