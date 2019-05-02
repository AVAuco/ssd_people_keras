SSD-based people detector for Keras
===================================
![detections](https://github.com/AVAuco/ssd_people_keras/blob/master/sample_detections.png)

Conversion made by Rafael Berral-Soler, based on the [original work](https://github.com/AVAuco/ssd_people) by Pablo Medina-Suarez and Manuel J. Marin-Jimenez.

This model relies on the Keras implementation of Single-Shot Multibox Detector by Pierluigi Ferrari found [here](https://github.com/pierluigiferrari/ssd_keras).

## Notes
* In order to use this model, you **MUST** import models/keras_ssd512.py, bounding_box_utils and keras_layers from its [repository](https://github.com/pierluigiferrari/ssd_keras).
* Because of the size of the converted model, to clone the repository [git-lfs](https://git-lfs.github.com/) is needed; the model can also be converted from the original MatConvNet model using convert_ssd_512.py. 
  Instructions for setting up git-lfs alongside git (from ssd_people [repository](https://github.com/AVAuco/ssd_people)):
  
  ```
  Install git:     
      sudo apt-get install git
  Install git-lfs:
      sudo apt-get install git-lfs
  Set up git-lfs:
      git lfs install
  Clone ssd_people_keras from GitHub using the method of your choice: 
      git clone https://github.com/AVAuco/ssd_people_keras.git (HTTPS)
      git clone git@github.com:AVAuco/ssd_people_keras.git (SSH)
  ```
* Script convert_ssd_512.py may be able to convert other SSD-512 models, provided they use the same MatConvNet toolbox used in ssd_people [repository](https://github.com/AVAuco/ssd_people). Slight modifications to the script and the layer mapping (layers.csv) could make possible to convert also SSD-256 models.
* Detections using the converted model could not match detections obtained with the original MatConvNet implementation. Adjusting confidence threshold should improve performance.

## Software requirements
In order to run demo.py:
* Python packages: numpy, imageio, matplotlib, keras.
* [Keras SSD-512 implementation](https://github.com/pierluigiferrari/ssd_keras):
  * models/ssd512.py
  * bounding_box_utils
  * keras_layers

In order to use convert_ssd_512.py:
* Python packages: numpy, pandas, keras, scipy.
* loadmat_stackoverflow.py, code obtained from [here](https://stackoverflow.com/a/8832212) (accessed on April 11, 2019).
* [Keras SSD-512 implementation](https://github.com/pierluigiferrari/ssd_keras):
  * models/ssd512.py
  * bounding_box_utils
  * keras_layers

## Acknowledgements
Picture used in demo by [Ross Broadstock](https://www.flickr.com/people/figurepainting/). Licensed under [CC BY 2.0](https://creativecommons.org/licenses/by/2.0/) license.
