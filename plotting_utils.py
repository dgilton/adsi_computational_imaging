import matplotlib.pyplot as plt

def singleimage_color_plot(input_image):
    plt.figure()
    plt.imshow(input_image, vmin=0, vmax=1); plt.colorbar()
    plt.show()

def before_after_plot(corrupted_image, restored_image):
    # plt.figure()
    _, ax_array = plt.subplots(1,2,sharey=True)
    ax_array[0].imshow(corrupted_image, vmin=0, vmax=1)
    ax_array[0].set_title('Corrupted Image')
    ax_array[1].imshow(restored_image, vmin=0, vmax=1)
    ax_array[1].set_title('Restored Image')
    plt.show()

def true_noisy_plot(true_image, corrupted_image):
    # plt.figure()
    _, ax_array = plt.subplots(1,2)
    ax_array[1].imshow(corrupted_image, vmin=0, vmax=1)
    ax_array[1].set_title('Corrupted Image')
    ax_array[0].imshow(true_image, vmin=0, vmax=1)
    ax_array[0].set_title('Original Image')
    plt.show()

def before_corrupted_after_plot(true_image, corrupted_image, restored_image):
    # plt.figure()
    _, ax_array = plt.subplots(3)
    ax_array[0].imshow(true_image, vmin=0, vmax=1)
    ax_array[0].set_title('True Image')
    ax_array[1].imshow(corrupted_image, vmin=0, vmax=1)
    ax_array[1].set_title('Corrupted Image')
    ax_array[2].imshow(restored_image, vmin=0, vmax=1)
    ax_array[2].set_title('Restored Image')
    plt.show()
