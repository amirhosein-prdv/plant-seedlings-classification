# Plant Seedlings Classification

This project aims to classify plant seedlings into various species using convolutional neural networks (CNNs) implemented in PyTorch. The dataset used is from the Plant Seedlings Classification competition on Kaggle, which can be accessed [here](https://www.kaggle.com/competitions/plant-seedlings-classification/data?select=train).

## Objectives

1. **CNN Model**: Initially, a CNN model was built from scratch without using any augmentation or transfer learning techniques. The model's performance metrics including loss, accuracy, and sample plots of model performance on the test data were reported.

2. **Data Augmentation**: In the second phase, data augmentation techniques were employed to enhance the model's performance. The results, including improvements in loss, accuracy, and sample plots, were documented.

3. **Transfer Learning**: Finally, transfer learning was utilized by employing a pre-trained VGG network. The model's performance metrics were reported similarly to the previous sections.

## Project Structure

- `dataset/`: Contains the dataset used for training and testing.
- `main.ipynb`: Jupyter notebooks for each phase of the project, including data preprocessing, model training, evaluation, and visualization.
- `Src/`: Utility scripts for data loading, preprocessing, and visualization.

## How to Use

1. Clone this repository:
   ```bash
   git clone https://github.com/amirhosein-prdv/plant-seedlings-classification.git
   ```
2. Navigate to the project directory:
   ```bash
   cd plant-seedlings-classification
   ```
3. Run the Jupyter notebook to replicate the experiments and analyze the results.

## Results

Detailed results including loss, accuracy, and visualizations of model performance are provided in the respective sections of the Jupyter notebook. 
