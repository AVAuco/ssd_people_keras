import numpy as np
from imageio import imread
import matplotlib.pyplot as plt
from keras.preprocessing import image
from keras_ssd512 import ssd_512

# Setting image size and creating the model.

img_height = 512
img_width = 512

model = ssd_512(image_size=(img_height, img_width, 3), n_classes=1, min_scale=0.1, max_scale=1, mode='inference')

# Loading stored weights.

model.load_weights('head-detector.h5')

# Setting input image path.

img_path = 'data/people_drinking.jpg'

input_images = []

img = image.load_img(img_path, target_size=(img_height, img_width))
img = image.img_to_array(img)
input_images.append(img)
input_images = np.array(input_images)

# Detecting heads.

y_pred = model.predict(input_images)

# Setting confidence threshold and filtering detections.

confidence_threshold = 0.5

y_pred_thresh = [y_pred[k][y_pred[k, :, 1] > confidence_threshold] for k in range(y_pred.shape[0])]

# Showing detections over original input image.

ori_image = imread(img_path)

colors = plt.cm.hsv(np.linspace(0, 1, 21)).tolist()
classes = ['background',
           'head']

plt.figure(figsize=(15, 15))
plt.imshow(ori_image)

current_axis = plt.gca()

for box in y_pred_thresh[0]:
    xmin = box[-4] * ori_image.shape[1] / img_width
    ymin = box[-3] * ori_image.shape[0] / img_height
    xmax = box[-2] * ori_image.shape[1] / img_width
    ymax = box[-1] * ori_image.shape[0] / img_height
    color = colors[int(box[0])]
    label = '{}: {:.2f}'.format(classes[int(box[0])], box[1])
    current_axis.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, color=color, fill=False, linewidth=2))
    current_axis.text(xmin, ymin, label, size='x-large', color='white', bbox={'facecolor': color, 'alpha': 1.0})

plt.show()
