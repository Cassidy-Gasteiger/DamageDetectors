# Damage Detectors

## Introduction and Background
Natural disaster zones are difficult to navigate, and it may take weeks or months to understand the full scope of structural and human damage. An efficient and accurate classification model could take available satellite imagery from affected areas and classify the level of structural damage. This would greatly reduce the speed of response and decrease the risk for relief workers. 

Previous research has highlighted the effectiveness of neural networks as they reduce high dimensionality of images without losing information and demonstrate high accuracy in detecting post-disaster building and vegetation damage. Existing damage classification models also incorporate additional features like geolocation data and flood risk index to provide a highly granular map of damage.

## Problem Definition and Dataset

In this project, we want to:

1.	Accurately classify damaged and undamaged buildings from post-hurricane satellite imagery 
2.	Test generalizability of our classification model on other post-disaster datasets

Our main dataset is satellite images depicting structures in the aftermath of Hurricane Harvey, pre-labeled to indicate damage. The geolocation data consists of distance from nearest water body and building elevation level. We will test the model’s generalizability on Stanford’s xBD building dataset that spans a range of geographies and natural disasters.

## Methods

We’ll begin with preprocessing the data by removing any images where the buildings can’t be seen clearly. We’ll then resize, adjust for noise or rotate the images if necessary and combine the geolocation data with the image data using geo-coordinates.

For image classification, we will build a Convolutional Neural Network. The network will be composed of an image encoder, connected layers that encode the geolocation features, a layer to combine the encoded terms and finally, a multi-class prediction layer. The combined image and geolocation embedding will be appended to a softmax layer resulting in a one-hot encoded vector.

To achieve optimal results, we’ll do hyperparameter tuning, e.g., number of layers, embedding size, Adam Optimizer parameters, etc. When testing generalizability on the xBD dataset, we will:
1)	Use unsupervised algorithms like k-Means + Transfer Learning, DBSCAN and GMM to generate clusters of images with and without buildings. 
2)	In images with buildings, use a segmentation approach to capture each building’s polygon.
3)	Apply the CNN model on the encoded building images and analyze the results across each hurricane.

We’ll experiment with PyTorch and TensorFlow and use the PACE COC-ICE cluster.

![image](https://user-images.githubusercontent.com/95386379/219880890-f71051e4-094b-46a7-afb5-b80021993729.png)

## Potential Results and Discussion

With the classification model, we hope to obtain results where images are accurately classified into one of the sub-categories. To measure its performance, we will use the accuracy and F1-score functions from scikit-learn. For the clustering model, we will use the elbow method for k selection and silhouette scores for tuning.

When testing generalizability, we expect to see a decrease in the F1-score. Previous research suggests that generalizing the model to identify damage from new disasters is challenging due to various reasons like differing pixel distributions. In addition, the clustering algorithm may recognize hidden features, e.g., geographical coloring, which may not be related to building presence.

## References
Berezina, Polina and Desheng Liu. “Hurricane damage assessment using couple convolutional neural networks: a case study of hurricane Michael.” Geomatics, Natural Hazards and Risks: pp. 414-31. 2021.

Chen, Xiao. “Using Satellite Imagery to Automate Building Damage Assessment: A case study of the xBD dataset.” Department of Civil and Environmental Engineering, Stanford University, Stanford, CA. 2021.

Cao, Quoc Dung and Youngjun Choe. “Post-Hurricane Damage Assessment Using Satellite Imagery and Geolocation Features.” Department of Industrial and Systems Engineering, University of Washington, Seattle, WA. 2020.

Khajwal, Asim B., Chih-Shen Cheng, and Arash Noshadravan. “Multi-view Deep Learning for Reliable Post-Disaster Damage Classification.” ArxIv, Cornell University. 2022.

Sharma, Neha, Vibhor Jain, and Anju Mishra. “An Analysis of Convolutional Neural Networks for Image Classification.” Procedia Computer Science (132): 377-384. 2018.
