from pathlib import Path
from torch.nn import Conv2d
from numpy.typing import NDArray
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import cv2

Conv2d

SCRIPT_PATH = Path(__file__).parent
INPUT_PATH = SCRIPT_PATH / "cat.jpg"

def main():
    bgr_img = cv2.imread(str(INPUT_PATH))
    rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB) 

    resized_rgb_img = cv2.resize(rgb_img, (224, 224), interpolation=cv2.INTER_AREA) # cv2 interpolation 참조: https://docs.opencv.org/3.4/da/d54/group__imgproc__transform.html#ga5bb5a1fea74ea38e1a5445ca803ff121
    # plt.imshow(final_rgb_img)
    # plt.show()

    final_rgb_img = np.transpose(resized_rgb_img, (2, 0, 1))
    print("final_rgb_img's shape:", final_rgb_img.shape)

    convolution(
        input_mat=final_rgb_img,
        in_channels=3,
        out_channels=8,
        kernel_size=(3, 3)
    )


def convolution(input_mat: NDArray, in_channels: int, out_channels: int, kernel_size: Tuple[int] , bias: bool=False, strides: int=1, pad: int=0) -> NDArray:    
    kernels_shape = (out_channels, in_channels, kernel_size[0], kernel_size[1])
    init_kernels = np.random.random(kernels_shape)
    print("init_kernels' shape:", init_kernels.shape)
    # print("init_kernels' value:", init_kernels)


def sliding_window(input_mat: NDArray, kernel_size: Tuple[int]):
    input_mat[[]]

    

main()


# cv2.imshow("img", img_rgb)
# cv2.waitKey(0)

