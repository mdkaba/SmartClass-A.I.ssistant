# SmartClass-A.I.ssistant

![image](https://github.com/mdkaba/SmartClass-A.I.ssistant/assets/23130811/51ae01bc-2c1c-4332-a2eb-c86516be65f5)

## Objectives

The primary objective of the SmartClass A.I.ssistant project is to develop a Convolutional Neural Network (CNN) using PyTorch to analyze and classify student images into four distinct states: Neutral, Engaged/Focused, Angry/Irritated, and Happy , the fourth state we have chosen. This involves collecting, cleaning, and labeling suitable datasets, building and training the CNN model, evaluating its performance using various metrics, and conducting bias analysis to ensure ethical AI application. The project aims to provide real-time feedback during lectures by recognizing students' facial expressions, enhancing the learning experience dynamically.

### Group Information

* **Mamadou Kaba (github.com/mdkaba)**: Training Specialist  
  As the Training Specialist, I will be the main person crafting, tuning, and training the Convolutional Neural Network. While I will be involved in all stages of model development, my primary focus will be on setting hyperparameters, monitoring training progress, and troubleshooting convergence issues.

* **Jaskirat Kaur (github.com/jaskiratkaur1906)**: Evaluation Specialist  
  As the Evaluation Specialist, my main task will be to rigorously test and evaluate our model's performance. I will analyze predictions, identify strengths and weaknesses, and detect any biases. My work will guide the refinement of the model and ensure it performs accurately and fairly across all metrics.

* **Kaloyan Kirilov (github.com/kalo2711)**: Data Specialist  
  As the Data Specialist, my primary responsibility is managing the dataset lifecycle, which includes sourcing, pre-processing, and loading data. I will ensure data quality and integrity, while also collaborating with the team to enhance our understanding and troubleshooting of data-related issues.
    
## Contents and Purpose of Each File

### Data_Cleaning Folder

* **duplicate_check.py**: This script detects and removes duplicate images from the dataset to ensure a unique and diverse dataset.
* **normalize.py**: This script normalizes pixel intensity values of images, ensuring consistent intensity distribution across all images.
* **resize.py**: This script resizes images to 96x96 pixels, standardizing the image dimensions.
* **bias_dataset_label.py**: This script labels all images in the bias datasets in the format Expression_Race_Index or Expression_Gender_Index.
* **definitive_dataset_label.py**: This script labels all images in the definitive dataset in the format Expression_Race_Gender_Index.

### Data_Visualization Folder

* **visualize_dataset.py**: This script visualizes the class distribution and sample images, providing insights into the dataset characteristics.
* **Original Dataset Plots**: Contains visualizations for the original dataset.
* **Definitive Dataset Plots**: Contains visualizations for the definitive dataset.

### Data_Training Folder

* **data_train.py**: Script to train the CNN models.
* **dataset_utils.py**: Utility functions for loading and processing datasets.
* **main_model.py**: Implementation of the main CNN model.
* **model_variant1.py**: Implementation of the first CNN model variant.
* **model_variant2.py**: Implementation of the second CNN model variant.
* **model_utils.py**: Utility functions for model evaluation.

### Data_Evaluation Folder

* **saved_kfolds**: Contains the results and models from k-fold cross-validation.
* **saved_models**: Contains the best model weights and metrics for each model variant.
* **bias_detect.py**: Script to train the models with bias detection and mitigation.
* **kfold_validate.py**: Script to perform k-fold cross-validation.
* **run_kfold.py**: Script to evaluate models using k-fold cross-validation.
* **run_model.py**: Script to evaluate the trained models on datasets or single images.



### Dataset Folder

* **Bias Dataset**: Contains the bias datasets for different groups.
* **Definitive Dataset**: Contains the final, cleaned, and labeled dataset used for training the definitive model.
* **Original Dataset**: Contains the original dataset before bias mitigation.

## Execution Steps

1. Clone the Repository: 
    ```bash
    git clone git@github.com:your_username/SmartClass-A.I.ssistant.git
    ```
    **Note**: Replace `your_username` with your actual GitHub username.

2. Install Dependencies:
    * Make sure you have Python installed on your system.
    * Navigate to the project directory:
      ```bash
      cd SmartClass-A.I.ssistant
      ```
    * Install dependencies:
      ```bash
      pip install -r requirements.txt
      ```

### Data Cleaning

1. Detect and Remove Duplicates:
    ```bash
    python Data_Cleaning/duplicate_check.py
    ```
    This will detect and remove any duplicate images from the dataset.

2. Normalize Images:
    ```bash
    python Data_Cleaning/normalize.py
    ```
    This will normalize the pixel intensity values inside the folder of each class.

3. Resize Images:
    ```bash
    python Data_Cleaning/resize.py
    ```
    This will resize all images in the folder to 96x96 pixels, ensuring uniform dimensions.

4. Label Bias Dataset:
    ```bash
    python Data_Cleaning/bias_dataset_label.py
    ```
    This script will label all images in the bias datasets in the format Expression_Race_Index or Expression_Gender_Index.

5. Label Definitive Dataset:
    ```bash
    python Data_Cleaning/definitive_dataset_label.py
    ```
    This script will label all images in the definitive dataset in the format Expression_Race_Gender_Index.
   
    **Note**: The Dataset inside this repo is already cleaned. These previous scripts are to be used on the Team folder for demonstration. Or, please follow the next step instead.

### Data Visualization

1. Visualize Dataset:
    ```bash
    python Data_Visualization/visualize_dataset.py
    ```
    This script will generate visualizations for class distribution and sample images with corresponding pixel intensity histograms. The visualizations help in understanding the dataset characteristics and ensuring that the data cleaning process was successful.

### Data Training

1. Train the models:
    ```bash
    python Data_Training/data_train.py
    ```
    You will be prompted to choose the model to train (`main`, `variant1`, `variant2`, or `definitive_model`).

    **Note**: The Models inside this repo are already trained and can be found in the `saved_models` folder. Please follow the next step instead.

### Data Evaluation

1. Detect and Mitigate Bias:
    ```bash
    python Data_Evaluation/bias_detect.py
    ```
    This script will perform bias detection and mitigation on the specified model.

    **Note**: The bias detection and mitigation for the models are already done and the metrics can be found in the `saved_models` folder.

2. Perform k-fold cross-validation:
    ```bash
    python Data_Evaluation/kfold_validate.py
    ```
    This script will perform k-fold cross-validation on the specified model.

    **Note**: The results of k-fold cross-validation are already saved and can be found in the `saved_kfolds` folder.

3. Evaluate the k-folds:
    ```bash
    python Data_Evaluation/run_kfold.py
    ```
    This script will evaluate models using k-fold cross-validation.


4. Evaluate the models:
    ```bash
    python Data_Evaluation/run_model.py
    ```
    You will be prompted to choose the model to evaluate (`main`, `variant1`, `variant2`, `definitive_model`, `main_male`, `main_female`, `main_asian`, `main_black`, `main_white`, `main_male_2.0`, `main_black_2.0`, `main_asian_2.0`) and the mode (`dataset` or `image`).


### Summary of Findings

| Model                | Macro Precision | Macro Recall | Macro F1-Score | Micro Precision | Micro Recall | Micro F1-Score | Accuracy |
|----------------------|-----------------|--------------|----------------|-----------------|--------------|----------------|----------|
| **Main Model**       | 86.73%          | 86.67%       | 86.67%         | 86.67%          | 86.67%       | 86.67%         | 86.67%   |
| **Variant 1**        | 81.20%          | 80.76%       | 80.78%         | 80.67%          | 80.67%       | 80.67%         | 80.67%   |
| **Variant 2**        | 84.15%          | 84.00%       | 83.99%         | 84.00%          | 84.00%       | 84.00%         | 84.00%   |
| **Definitive Model** | 85.96%          | 86.00%       | 86.30%         | 86.32%          | 86.00%       | 86.32%         | 86.32%   |
