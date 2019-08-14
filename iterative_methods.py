import numpy as np

from scipy.fftpack import fft2, ifft2, fftshift, ifftshift
import corruption_models

def psd_estimate(size):
    alpha = 1.8
    x, y = np.mgrid[-size//2 + 1:size//2+1, -size//2 + 1:size//2+1]
    x = np.array(x, dtype=np.complex); y = np.array(y, dtype=np.complex)
    g = 1. / ((x*2./size)**alpha + (y*2./size)**alpha + 1e-13)
    return g

def wiener_filter(input_data, corruption_kernel, noise_sigma):
    size = np.shape(input_data)[0]
    S = fftshift(psd_estimate(size))
    filter_shifted = fft2(fftshift(corruption_kernel))
    fourier_data = fft2(input_data)
    weiner_filter = np.conj(filter_shifted)*S / (np.square(np.abs(filter_shifted))*S + noise_sigma**2 + 1e-13)
    return np.real(ifft2(fourier_data * weiner_filter))

def inpaint_gradient_step(input_data, inpaint_mask, initial_point, stepsize):
    gradient_inner_term = initial_point - corruption_models.apply_inpainting_map(input_data, inpaint_mask)
    gradient_step = corruption_models.apply_inpainting_map(gradient_inner_term, inpaint_mask)
    return input_data + stepsize*gradient_step

def blur_gradient_step(input_data, blur_kernel, initial_point, stepsize):
    input_fft = fft2(input_data)
    filter_fft = fft2(fftshift(blur_kernel))
    y_fft = fft2(initial_point)
    gradient_step = np.real(ifft2(np.conj(filter_fft) * (y_fft - filter_fft*input_fft)))

    return input_data + stepsize*gradient_step

def simple_alternating_optimizer(initial_point, gradient_step, denoiser, n_steps, delta_tolerance):
    f_t = initial_point
    for ii in range(n_steps):
        f_tminus1 = f_t
        z_t = gradient_step(f_t)
        f_t = denoiser(z_t)

        delta = np.mean(np.square(f_t - f_tminus1)[:])
        if delta < delta_tolerance:
            break
        if ii % 10 == 0:
            print('Delta: ' + str(delta))

    return f_t

def tikhonov_fft(size):
    output_filter = np.zeros((size, size))
    laplacian = np.asarray([[1, 3, 1], [3, -20, 3], [1, 3, 1]])
    output_filter[:3, :3] = laplacian
    output_filter = np.roll(output_filter, (-1, -1))
    return fft2(output_filter)


def tikhonov_filter(input_data, corruption_kernel, tikhonov_term):
    size = np.shape(input_data)[0]
    S = tikhonov_fft(size)
    filter_shifted = fft2(fftshift(corruption_kernel))
    fourier_data = fft2(input_data)

    weiner_filter = np.conj(filter_shifted) / (
    np.square(np.abs(filter_shifted)) + tikhonov_term * np.square(np.abs(S)) + 1e-13)
    return np.real(ifft2(fourier_data * weiner_filter))


def projected_gradient_descent(initial_point, forward_gramian, xTy, projection_operator, n_iterations = 500,
                               step_size = 0.01):
    f_i = initial_point

    for ii in range(n_iterations):
        z_i = f_i - step_size * (forward_gramian(f_i) - xTy)
        f_i = projection_operator(z_i)

    return f_i

def vectorized_matrix_norm(img_1, img_2):
    return np.sum((img_1*img_2)[:])

# This is designed to work with 2d inputs
def conjugate_gradient_descent(initial_point, lhs_operator, rhs, n_iterations = 500, tolerance = 1e-8):
    x_k = initial_point
    r_k = rhs - lhs_operator(x_k)
    p_k = r_k
    norm_rk = vectorized_matrix_norm(r_k, r_k)
    for k in range(n_iterations):
        alpha_k = vectorized_matrix_norm(r_k, r_k) / vectorized_matrix_norm(r_k, lhs_operator(r_k))
        x_k = x_k + alpha_k * p_k
        r_k1 = r_k - alpha_k * lhs_operator(p_k)
        norm_rk1 = vectorized_matrix_norm(r_k1,r_k1)
        if np.sqrt(norm_rk1) < tolerance:
            break
        beta_k = norm_rk1 / norm_rk
        p_k = r_k1 + beta_k * p_k

    return x_k

def blur_consistency(input_data, corruption_kernel, noise_sigma):
    # return tikhonov_filter(input_data, corruption_kernel, noise_sigma)
    return wiener_filter(input_data, corruption_kernel, noise_sigma)

def tikhonov_conjugate_gradient_descent(initial_point, lhs_operator, rhs, tikhonov_weight,
                                        n_iterations = 100, tolerance = 1e-8):
    x_k = initial_point
    r_k = rhs - lhs_operator(x_k)
    p_k = r_k
    norm_rk = vectorized_matrix_norm(r_k, r_k)
    for k in range(n_iterations):
        A = lhs_operator(r_k) + tikhonov_weight * r_k
        alpha_k = vectorized_matrix_norm(r_k, r_k) / (vectorized_matrix_norm(r_k, A) + 1e-10)
        x_k = x_k + alpha_k * p_k
        r_k1 = r_k - alpha_k * A
        norm_rk1 = vectorized_matrix_norm(r_k1,r_k1)
        if np.sqrt(norm_rk1) < tolerance:
            break
        beta_k = norm_rk1 / (norm_rk + 1e-10)
        p_k = r_k1 + beta_k * p_k

    return x_k

def plug_and_play(initial_point, gramian, denoiser, aTy, tikhonov_weight, n_iterations=100, tolerance = 1e-8):
    zeros = np.zeros_like(initial_point)
    data_consistency = lambda x: tikhonov_conjugate_gradient_descent(initial_point, gramian, x, tikhonov_weight)
    x_hat = initial_point
    v_hat = x_hat
    u = np.copy(zeros)

    for k in range(n_iterations):
        x_tilde = v_hat - u
        x_hat = data_consistency(aTy + x_tilde)
        v_tilde = x_hat + u
        v_hat = denoiser(v_tilde)
        u = u + x_hat - v_hat
        # print(k)
        if k % 10 == 0:
            print(np.sum(np.isnan(x_hat)))

    return x_hat

def red_pg(initial_point, gramian, denoiser, aTy, tikhonov_weight, L, n_iterations=100, tolerance = 1e-8):
    zeros = np.zeros_like(initial_point)
    data_consistency = lambda x: tikhonov_conjugate_gradient_descent(zeros, gramian, x, tikhonov_weight)
    x_hat = initial_point

    v = np.copy(zeros)

    for k in range(n_iterations):
        x_hat = data_consistency(aTy + v)
        v = (1 / L) * denoiser(x_hat) - ((1-L)/L) * x_hat
        # print(k)
        if k % 10 == 0:
            print(np.sum(np.isnan(x_hat)))

    return x_hat

def red_pg_analytic(initial_point, blur_kernel, denoiser, aTy, tikhonov_weight, L, sigma, n_iterations=100, tolerance = 1e-8):
    zeros = np.zeros_like(initial_point)
    data_consistency = lambda x, sigma: blur_consistency(x, blur_kernel, sigma)
    x_hat = initial_point

    v = np.copy(zeros)

    for k in range(n_iterations):
        x_hat = data_consistency((1/sigma**2)*aTy + tikhonov_weight*v, sigma)
        v = (1 / L) * denoiser(x_hat) - ((1-L)/L) * x_hat
        # print(k)
        if k % 10 == 0:
            # print(np.sum(np.isnan(x_hat)))
            print(np.max(x_hat[:]))


    return x_hat