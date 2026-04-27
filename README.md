# Solar Panel Power Loss Estimation & Soiling Detection

## Description
This project focuses on predicting energy efficiency loss in photovoltaic (PV) panels due to soiling (dust, dirt, and debris). Unlike traditional image-only systems, this pipeline implements a **multimodal machine learning approach** that fuses high-dimensional visual features with environmental data (solar irradiance). 

This integration effectively addresses the "lighting-driven appearance change" problem, allowing the model to distinguish between genuine soiling and simple changes in ambient light, which is critical for real-world UAV or edge-monitored solar farm maintenance.

## Key Results
- **Predictive Accuracy ($R^2$):** **0.8126** (High correlation between predicted and actual power loss).
- **Mean Absolute Error (MAE):** **6.40** (Percentage points), indicating high precision in estimating energy degradation.
- **Explainability (XAI):** Integrated **SHAP** to identify that the fusion of mid-range irradiance and specific texture-based visual embeddings are the primary drivers for accurate loss estimation.

## Technologies Used
* **Programming & Core:** Python (Pandas, NumPy, H5py)
* **Deep Learning & Vision:** TensorFlow, Keras, OpenCV, Scikit-image, Albumentations
* **Machine Learning:** Scikit-learn (HistGradientBoosting, Random Forest, PCA, StandardScaler)
* **Interpretability & Utilities:** SHAP, Joblib
* **Visualization:** Matplotlib, Seaborn

## Dataset: DeepSolarEye
The project utilizes the **DeepSolarEye** dataset, a specialized collection for PV maintenance research.
- **Size:** ~45,754 RGB images capturing diverse soiling conditions.
- **Conditions:** Simulated natural dust (red sand, brown sand, gray powder) and varying patches of dirt.
- **Labels:** Continuous power-loss labels (ranging from 0% to 90%) and matched irradiance/timestamps.
- **Resolution:** Images processed at $150 \times 150$ or $192 \times 192$ pixels.
- **Significance:** Enables weakly supervised training for power loss prediction without requiring expensive pixel-level segmentation.

## Project Architecture
1.  **Feature Extraction:** Generates 1280-dimensional visual embeddings using a deep CNN backbone.
2.  **Dimensionality Reduction:** Applies **PCA** to reduce visual features to 32 principal components, retaining maximum variance while reducing noise.
3.  **Multimodal Fusion:** Concatenates reduced visual embeddings with engineered image features and numerical solar irradiance data.
4.  **Regression Pipeline:** Trains an ensemble of **HistGradientBoosting** and **Random Forest** regressors to output the final power loss percentage.

## Getting Started

## Project Structure
- `results/`: Folder for saving output data and visualizations.
- `model/`: Folder for saving trained model files.
- `Final_Deep_Solar_Eye_Predicting_Energy_Loss_in_Solar_Panels_Using_ML_Based_Dust_Detection.ipynb`: Main project notebook.
- 
### Installation
Clone the repository and install the required dependencies:
```bash
git clone [https://github.com/YourUsername/solar-panel-power-loss.git](https://github.com/YourUsername/solar-panel-power-loss.git)
cd solar-panel-power-loss
pip install -r requirements.txt
```



## Usage
Open the Jupyter notebook `Final_Deep_Solar_Eye_Predicting_Energy_Loss_in_Solar_Panels_Using_ML_Based_Dust_Detection.ipynb` and run the cells to reproduce the analysis and model training.

## Citation
S. Mehta, A. P. Azad, S. A. Chemmengath, V. Raykar and S. Kalyanaraman, "DeepSolarEye: Power Loss Prediction and Weakly Supervised Soiling Localization via Fully Convolutional Networks for Solar Panels," 2018 IEEE Winter Conference on Applications of Computer Vision (WACV).
