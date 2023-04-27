import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from numpy.typing import NDArray
from typing import Tuple
from pprint import pprint


def Conv2D(image: NDArray[np.uint], out_channels: int, kernel_size: Tuple[int], padding: int=0, strides: int=1):
    '''
    input = (C, H, W)
    kernel_size = (k, k)
    ouptput = (out_channels, output_height, output_width)
    '''
    channel, image_height, image_width = image.shape
    kernel_height, kernel_width = kernel_size[0], kernel_size[1]
    kernel = np.random.random((channel, kernel_size[0], kernel_size[1]))
    
    output_height = int(((image_height - kernel_height + 2 * padding) / strides) + 1)
    output_width= int(((image_width - kernel_width + 2 * padding) / strides) + 1)
    
    output = np.zeros((out_channels, output_height, output_width))

    if padding != 0:
        imagePadded = np.zeros((channel, image_height + padding * 2, image_width + padding * 2))
        imagePadded[:, padding:(-1*padding), padding:(-1*padding)] = image
    
    # pprint('='*50)
    # pprint(f'padded image shape : {imagePadded.shape}')
    # pprint(np.round(imagePadded, 2))

    # convolution 2D
    for z in range(0, out_channels):
        output_per_channel = np.zeros((output_height, output_width))
        
        for y in range(0, output_height):
            if (y*strides + kernel_height) <= imagePadded.shape[1]:

                for x in range(0, output_width):                
                    if (x*strides + kernel_width) <= imagePadded.shape[2]:
                        output_per_channel[y][x] = np.sum(imagePadded[:,
                                                               y*strides : y*strides + kernel_height,
                                                               x*strides : x*strides + kernel_width] * kernel).astype(np.float32)
        output[z, :, :] = output_per_channel
    
    # pprint('='*50)
    # pprint(f'output image shape : {output.shape}')
    # pprint(output)
    # pprint('='*50)

    return output

IMAGE_PATH = "source_image/cat_224x224.jpg"
OUTPUT_CHANNELS = 6
KERNEL_SIZE = (3, 3)
PADDING = 1
STRIDE = 1

raw_image = Image.open(IMAGE_PATH)
raw_image = np.array(raw_image)
image = raw_image.transpose((2, 0, 1))
output = Conv2D(image, out_channels=OUTPUT_CHANNELS, kernel_size=KERNEL_SIZE, padding=PADDING, strides=STRIDE)
print(raw_image.shape)
print(output.shape)

plt.imshow(raw_image)
for channel_i, channel in enumerate(output):
    channel = np.expand_dims(channel, axis=0).astype(int)
    channel = np.transpose(channel, (1,2,0))
    channel_scaled = channel * (255.0/channel.max())
    channel_scaled = channel_scaled.astype(int)
    print(channel_scaled.shape)
    plt.subplot(1, OUTPUT_CHANNELS, channel_i+1)
    plt.axis('off')
    plt.title(f"channel: {channel_i}")
    plt.imshow(channel_scaled, cmap='gray', vmin=0, vmax=255)