# Plant-Disease-Detection-and-Classification-using-Deep-Learning
## Overview
This project explores the application of deep learning techniques to detect and classify plant diseases using images. Utilizing Convolutional Neural Networks (CNNs), our study compares the efficacy of different architectures—GoogleNet, MobileNetV2, and EfficientNet—across various datasets with a focus on agricultural applications. The project's goal is to develop a robust model that enhances the detection of plant diseases at early stages, potentially improving crop management and productivity.

## Datasets
We employ three primary datasets in our analysis:

### Cassava Dataset: Features images of cassava plant leaves, crucial for identifying common diseases affecting this crop.
### Crop Disease Dataset: Consists of diverse leaf images sourced from farms, representing multiple disease classes.
### Plant Village Dataset: Includes images categorized into several classes of plant diseases across multiple species.
All datasets are split into training (80%), validation (10%), and testing (10%) sets to ensure rigorous evaluation.

## Methodology
Our methodology consists of:

## Data Preprocessing: Standardizing image dimensions, applying various augmentations, and normalizing the data to improve model training.
Model Training: Evaluating three CNN architectures—GoogleNet, MobileNetV2, and EfficientNet. Models are trained from scratch and also fine-tuned using transfer learning approaches to compare performance.
Performance Metrics: Models are assessed based on accuracy, precision, recall, and F1 scores. Special attention is given to the robustness and generalization capability across different conditions and datasets.
## Results
The models demonstrate varying effectiveness across datasets, with EfficientNet generally providing the best balance between performance and computational efficiency. Detailed results of the experiments, including performance metrics and training times, are discussed comprehensively in the project report.

## Usage
The models developed can be utilized by researchers and practitioners in the agricultural sector for early detection of plant diseases, facilitating timely and effective treatment strategies. This could be particularly beneficial in regions where access to expert knowledge and resources is limited.

## Supplementary Material
Figures and additional data supporting the findings are included in the final report, providing insights into the models' performance under various conditions.
