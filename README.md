# Plant Disease Detection Using Deep Learning

## Overview
This project explores the application of deep learning techniques to detect and classify plant diseases using images. Utilizing Convolutional Neural Networks (CNNs), our study compares the efficacy of different architectures—GoogleNet, MobileNetV2, and EfficientNet—across various datasets with a focus on agricultural applications. The goal is to develop a robust model that enhances the detection of plant diseases at early stages, potentially improving crop management and productivity.

## Datasets
We employ three primary datasets in our analysis:

1. Cassava Dataset [https://tensorflow.google.cn/datasets/catalog/cassava]: The Cassava dataset features leaf images of the cassava plant, showcasing healthy leaves and four disease conditions: Cassava Mosaic Disease (CMD), Cassava Bacterial Blight (CBB), Cassava Green Mite (CGM), and Cassava Brown Streak Disease (CBSD). It includes 9,430 labeled images

2. Crop Pest and Disease Dataset [https://data.mendeley.com/datasets/bwh3zbpkpv/1]: This dataset, obtained from local farms in Ghana, is composed of two parts: raw images totaling 24,881, divided into 22 classes. These images are distributed among four types of crops as follows: 6,549 images of Cashew, 7,508 of Cassava, 5,389 of Maize, and 5,435 of Tomato.[2]

3. PlantVillage Dataset [https://www.tensorflow.org/datasets/catalog/plant_village]: The PlantVillage dataset features 54,303 images, showcasing both healthy and diseased leaves, organized into 38 distinct categories according to plant species and disease type.

All datasets are split into training (80%), validation (10%), and testing (10%) sets to ensure rigorous evaluation.


## Methodology
Our methodology consists of:
1. **Data Preprocessing**: Standardizing image dimensions, applying various augmentations, and normalizing the data to improve model training.
2. **Model Training**: Evaluating three CNN architectures—GoogleNet, MobileNetV2, and EfficientNet. Models are trained from scratch and also fine-tuned using transfer learning approaches to compare performance.
3. **Performance Metrics**: Models are assessed based on accuracy, precision, recall, and F1 scores. Special attention is given to the robustness and generalization capability across different conditions and datasets.

## Results
The models demonstrate varying effectiveness across datasets, with EfficientNet generally providing the best balance between performance and computational efficiency. Detailed results of the experiments, including performance metrics and training times, are discussed comprehensively in the project report.

## Usage
The models developed can be utilized by researchers and practitioners in the agricultural sector for early detection of plant diseases, facilitating timely and effective treatment strategies. This could be particularly beneficial in regions where access to expert knowledge and resources is limited.

## Supplementary Material
Figures and additional data supporting the findings are included in the final report, providing insights into the models' performance under various conditions.

## References
A comprehensive list of references supporting this study is available in the final report, detailing the datasets used, methodologies applied, and previous work in the field.

## Install Dependencies
Run the below command to install all the dependencies:

```
pip install -r requirements.txt
```


## Usage
Run the below command to run entire pipeline or just inference pipeline:

```
python main.py --dataset [dataset] --model [model] --train_mode [True/False] --pretrained [True/False] --max_epochs [max_epochs] --batch_size [batch_size] --lr [learning rate] --tune_hyperparams [True/False]

dataset: cassava, crop_diease, plant_village
model: googlenet, efficientnet_b0, mobilenet_v2
train_mode: True for running in training else False
pretrained: True for using pre-trained weights else False
max_epochs: number of epochs
batch_size: batch size
lr: learning rate
tune_hyperparams: True for tuning hyperparams else False
```

