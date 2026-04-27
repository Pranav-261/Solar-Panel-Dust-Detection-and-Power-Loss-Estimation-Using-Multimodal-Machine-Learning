# Solar Panel Power Loss Estimation

## Description
This project aims to estimate power loss in solar panels due to soiling (dirt, dust, etc.) using image analysis and environmental data. It develops a regression model to predict power loss percentage based on panel images and irradiance levels.

## Dataset: DeepSolarEye
The **DeepSolarEye** dataset is a specialized collection designed for solar panel soiling detection and power loss prediction. It addresses the maintenance challenges of solar plants, where accumulated dirt or debris reduces efficiency.

### Key Dataset Details
- **Content:** Over 45,000 RGB images (~45,754) capturing various soiling conditions on PV panels.
- **Soiling Types:** Simulated natural conditions including dust, gray powder, red and brown sand, and varying thick patches of dirt (ranging from clean to 90% power loss).
- **Environmental Factors:** Includes irradiance levels and timestamps for each image.
- **Data Resolution:** RGB images often at $150 \times 150$ or $192 \times 192$ resolution.
- **Labels:** Includes power loss labels, enabling weakly supervised training without complex pixel-level segmentation maps.
- **Purpose:** Power loss prediction, soiling detection, and localization via CNNs.
- **License:** Creative Commons.
- **Associated Paper:** *"DeepSolarEye: Power Loss Prediction and Weakly Supervised Soiling Localization via Fully Convolutional Networks for Solar Panels"* (2018).


## Technologies Used
- **Programming Language:** Python
- **Data Analysis:** Pandas, NumPy, H5py
- **Visualization:** Matplotlib, Seaborn
- **Image Processing:** OpenCV, Scikit-image, Pillow, Albumentations
- **Machine Learning/Deep Learning:** TensorFlow, Keras, Scikit-learn (Random Forest, HistGradientBoostingRegressor, PCA, StandardScaler, SimpleImputer)
- **Model Explainability:** SHAP
- **Utilities:** Joblib

## Project Structure
- `results/`: Folder for saving output data and visualizations.
- `model/`: Folder for saving trained model files.
- `Final_Deep_Solar_Eye_Predicting_Energy_Loss_in_Solar_Panels_Using_ML_Based_Dust_Detection.ipynb`: Main project notebook.

## Getting Started

### Prerequisites
Make sure you have Python installed. It is recommended to use a virtual environment.

### Installation
Install the required dependencies using pip:

```bash
pip install -r requirements.txt
```

## Usage
Open the Jupyter notebook `Final_Deep_Solar_Eye_Predicting_Energy_Loss_in_Solar_Panels_Using_ML_Based_Dust_Detection.ipynb` and run the cells to reproduce the analysis and model training.
