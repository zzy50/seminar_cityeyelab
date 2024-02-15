import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

INPUT_PATH = "img/cat.jpg"

bgr_img = cv2.imread(str(INPUT_PATH))
rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB) 

resized_rgb_img = cv2.resize(rgb_img, (40, 40), interpolation=cv2.INTER_AREA)
mean_subtracted = resized_rgb_img - resized_rgb_img.mean(axis=(0, 1), keepdims=True)

min_val = resized_rgb_img.min(axis=(0, 1), keepdims=True)
max_val = resized_rgb_img.max(axis=(0, 1), keepdims=True)
min_max_scaled = (resized_rgb_img - min_val) / (max_val - min_val)

# fig, axes = plt.subplots(1, 3, figsize=(14, 10))

# axes[0].imshow(resized_rgb_img)
# axes[0].set_title('Original Image')
# axes[0].axis('off')

# mean_subtracted_display = mean_subtracted - mean_subtracted.min()
# mean_subtracted_display = (mean_subtracted_display / mean_subtracted_display.max()) * 255
# axes[1].imshow(mean_subtracted_display.astype('uint8'))
# axes[1].set_title('Mean-Subtracted')
# axes[1].axis('off')

# axes[2].imshow((min_max_scaled).astype('uint8'))
# axes[2].set_title('Min-Max Scaling')
# axes[2].axis('off')

# plt.show()

change_view = False

fig = plt.figure(figsize=(14, 10))
ax1 = fig.add_subplot(131, projection='3d')
x, y = np.meshgrid(range(resized_rgb_img.shape[1]), range(resized_rgb_img.shape[0]))
for i in range(3):
    ax1.scatter(x.flatten(), y.flatten(), resized_rgb_img[:, :, i].flatten(), label=f'Channel {i+1}', alpha=0.2)
ax1.set_title('Raw Image')
ax1.legend()

ax2 = fig.add_subplot(132, projection='3d')
x, y = np.meshgrid(range(mean_subtracted.shape[1]), range(mean_subtracted.shape[0]))
for i in range(3):
    ax2.scatter(x.flatten(), y.flatten(), mean_subtracted[:, :, i].flatten(), label=f'Channel {i+1}', alpha=0.2)
ax2.set_title('Mean-Subtracted')
ax2.legend()

ax3 = fig.add_subplot(133, projection='3d')
for i in range(3):
    ax3.scatter(x.flatten(), y.flatten(), min_max_scaled[:, :, i].flatten(), label=f'Channel {i+1}', alpha=0.2)
ax3.set_title('Min-Max Scaling')
ax3.legend()

if change_view:
    ax1.view_init(260, -90)
    ax2.view_init(260, -90)
    ax3.view_init(260, -90)

plt.show()