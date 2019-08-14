import pybm3d

def bm3d_denoise(input_image, sigma):
    denoised_img = pybm3d.bm3d.bm3d(input_image, sigma)
    return denoised_img

