SSD-based people detector for Keras
===================================
![detections](https://github.com/AVAuco/ssd_people_keras/blob/master/sample_detections.png)

Conversion made by Rafael Berral-Soler, based on the [original work](https://github.com/AVAuco/ssd_people) by Pablo Medina-Suarez and Manuel J. Marin-Jimenez.

This model relies on the Keras implementation of Single-Shot Multibox Detector by Pierluigi Ferrari found [here](https://github.com/pierluigiferrari/ssd_keras).

## Notes
* In order to use this model, you need to import models/keras_ssd512.py, bounding_box_utils and keras_layers from its [repository](https://github.com/pierluigiferrari/ssd_keras).
* Script convert_ssd_512.py may be able to convert other SSD-512 models, provided they use the same MatConvNet toolbox used in [ssd_people repository](https://github.com/AVAuco/ssd_people). Slight modifications to the script and the layer mapping (
* 
