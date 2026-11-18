# SAR-Based Deforestation Detection Using Machine Learning and Federated Learning

A multi-phase project exploring SAR-based deforestation detection through machine learning, SAR raster processing, and federated learning.

## Overview

Deforestation monitoring is important for understanding forest loss and environmental changes. Satellite remote sensing provides a way to observe large geographical regions, while machine learning techniques can help identify patterns associated with forest disturbance and deforestation.

This project is developed in three phases, progressing from traditional machine learning-based classification to SAR raster processing and federated deep learning. It explores SAR data analysis, feature extraction, model evaluation, and distributed training across multiple clients.

---

## Phase 1 — Machine Learning-Based Deforestation Detection

### Objective

The first phase implements machine learning models for deforestation detection using Sentinel-1 SAR imagery. A Random Forest classifier is trained using SAR pixel information, followed by an experiment incorporating additional spatial and statistical features to improve classification performance.

### Dataset

**Deforestation Detection Dataset – Kaggle**

Dataset: https://www.kaggle.com/datasets/akhilchibber/deforestation-detection-dataset

The dataset contains Sentinel-1 imagery and corresponding training masks used to distinguish deforestation from non-deforestation regions.

### Data Preparation

- Used Sentinel-1 SAR imagery with two input bands.
- Ground-truth masks were used to label deforestation and non-deforestation pixels.
- Label 1 represents deforestation.
- Label 2 represents non-deforestation.
- Invalid pixels were excluded from training and testing.
- Training and test data were separated by raster tiles to evaluate performance on unseen regions.

### Dataset Split

Training tiles:

`RASTER_0–11, RASTER_13, RASTER_15`

Test tiles:

`RASTER_12, RASTER_14`

A balanced training sample was created with 140,000 pixels:

| Class | Training Samples |
|---|---:|
| Non-deforestation | 70,000 |
| Deforestation | 70,000 |
| **Total** | **140,000** |

The test set contains 15,859,710 valid pixels from the two held-out tiles.

### Model 1 — Random Forest with 2 SAR Features

The baseline model uses the two original SAR bands as input features.

**Configuration:**
- Algorithm: Random Forest Classifier
- Number of trees: 200
- Class weighting: Balanced
- Random state: 42
- Parallel processing: Enabled

The model was trained using two SAR features per pixel and evaluated on the unseen test tiles.

#### Baseline Results

| Metric | Score |
|---|---:|
| Accuracy | 73.52% |
| Precision | 79.26% |
| Recall | 76.09% |
| F1-score | 77.64% |

### Model 2 — Random Forest with 8 SAR Features

To improve the baseline model, additional features were extracted from the SAR bands. This experiment combines the original band values with derived features and local spatial statistics.

#### Feature Extraction

Eight features were used for each pixel:

| Feature | Description |
|---|---|
| Band 1 | Original SAR Band 1 value |
| Band 2 | Original SAR Band 2 value |
| Band Difference | Band 1 − Band 2 |
| Band Ratio | Band 1 / Band 2 |
| Local Mean – Band 1 | 7×7 neighborhood mean |
| Local Mean – Band 2 | 7×7 neighborhood mean |
| Local Standard Deviation – Band 1 | 7×7 local variability |
| Local Standard Deviation – Band 2 | 7×7 local variability |

Local mean and standard deviation features were calculated using a 7×7 spatial window. These features introduce neighborhood information alongside the original pixel-level SAR measurements.

#### Results — 8-Feature Random Forest

| Metric | Score |
|---|---:|
| Accuracy | **80.70%** |
| Precision | **84.22%** |
| Recall | **83.76%** |
| F1-score | **84.00%** |

### Model Comparison

The comparison shows the effect of adding spatial and statistical SAR features to the baseline Random Forest.

| Metric | 2 Features | 8 Features | Improvement |
|---|---:|---:|---:|
| Accuracy | 73.52% | **80.70%** | +7.19 pp |
| Precision | 79.26% | **84.22%** | +4.96 pp |
| Recall | 76.09% | **83.76%** | +7.67 pp |
| F1-score | 77.64% | **84.00%** | +6.35 pp |

### Confusion Matrix — 8-Feature Random Forest

Rows represent actual classes, and columns represent predicted classes.

| | Predicted Non-Deforestation | Predicted Deforestation |
|---|---:|---:|
| Actual Non-Deforestation | 4,770,974 | 1,504,005 |
| Actual Deforestation | 1,556,333 | 8,028,398 |

### Results Discussion

The baseline Random Forest achieved an accuracy of 73.52% using only the two original SAR bands. After adding band difference, band ratio, local mean, and local standard deviation features, the accuracy increased to 80.70%.

The F1-score also increased from 77.64% to 84.00%, indicating improved balance between precision and recall. The additional spatial features helped the model capture local SAR patterns beyond the original pixel values.

The eight-feature model was the better-performing Random Forest configuration in this experiment, demonstrating the value of incorporating neighborhood-based information into SAR classification.

---

## Phase 2 — SAR Raster Processing

### Objective

The second phase focuses on understanding and working with SAR raster data through a completed raster-processing course/project. It covers practical handling and processing of satellite imagery and provides experience with geospatial data operations.

### Course Repository

**Raster Processing SAR Course**

Repository: https://github.com/sangati12/Raster_processing_SAR_course

### Focus Areas

- Reading and handling raster datasets
- Understanding raster dimensions and bands
- Exploring raster properties and metadata
- SAR image visualisation
- Spatial data processing
- Raster operations using Python
- Geospatial data handling using Rasterio

### Tools

- Python
- Rasterio
- NumPy
- Matplotlib

The course implementation and associated files are maintained in the linked repository.

---

## Phase 3 — Federated Learning for Deforestation Detection

### Objective

The third phase implements federated learning for SAR-based deforestation segmentation. Instead of combining all training data into one centralized dataset, training data is distributed among three clients.

Each client trains a shared model locally on its own data. The local model updates are sent to a central server and aggregated using Federated Averaging (FedAvg). The updated global model is then distributed back to the clients for subsequent training rounds.

This phase evaluates the global segmentation model using geographically distributed client data.

### Dataset

**BraDD-S1TS — Brazilian Deforestation Dataset using Sentinel-1 SAR time series**

Dataset source: https://github.com/kaankaramanofficial/BraDD-S1TS

BraDD-S1TS is used for the federated deforestation segmentation implementation. The samples are processed and distributed across clients based on geographical/state groupings.

### Dataset Split

| Subset | Samples |
|---|---:|
| Training | 15,625 |
| Validation | 5,112 |
| Test | 5,251 |
| **Total** | **25,988** |

The training data is distributed among three clients, while validation and test subsets are kept for model evaluation.

### Federated Client Distribution

Training samples are divided into three clients based on geographical/state distribution.

| Client | States | Training Samples |
|---|---|---:|
| Client 1 | PA | 7,012 |
| Client 2 | MT, AM, MA | 4,320 |
| Client 3 | RO, AC, RR, AP, TO | 4,293 |
| **Total** | **9 states** | **15,625** |

Each client maintains its own local training data and performs local model training.

### Model Architecture

**U-Net — Semantic Segmentation**

U-Net is used to predict deforestation regions at the pixel level. Its encoder extracts image features, while the decoder reconstructs spatial information to produce the segmentation map.

| Component | Configuration |
|---|---|
| Task | Binary semantic segmentation |
| Architecture | U-Net |
| Input Channels | 2 |
| Output Channels | 1 |
| Loss Function | BCE + Dice Loss |
| Framework | PyTorch |
| Clients | 3 |
| Aggregation | Federated Averaging (FedAvg) |

### Federated Training Workflow

1. Initialize the global U-Net model.
2. Distribute the global model to the three clients.
3. Each client trains locally using its own SAR training samples.
4. Clients return locally trained model parameters.
5. The server aggregates the parameters using weighted FedAvg.
6. Update the global model.
7. Distribute the updated global model back to the clients.
8. Repeat the process over multiple federated rounds.
9. Evaluate the global model using validation data.

### Federated Averaging (FedAvg)

FedAvg aggregates client models based on their respective training sample counts. Clients with more training samples contribute proportionally to the aggregated model.

The aggregation produces a shared global model while keeping the training samples separated at the client level.

### Global Model Validation Results

The global model was evaluated using the validation dataset after federated training.

| Metric | Score |
|---|---:|
| IoU | 0.1998 |
| Dice Score | 0.3330 |
| Precision | 0.2245 |
| Recall | 0.6447 |
| Accuracy | 0.5968 |
| Predicted Positive Pixels | 44.8357% |

### Results Discussion

The global model achieved a recall of 0.6447, indicating that it identified a portion of the positive deforestation pixels in the validation set. However, the precision of 0.2245 shows that many predicted positive pixels were false positives.

The IoU of 0.1998 and Dice score of 0.3330 indicate that the overlap between the predicted segmentation regions and the ground-truth masks remains limited.

The model predicted approximately 44.84% of the pixels as positive, highlighting the need to examine segmentation threshold behaviour and false-positive predictions.

These results reflect the challenges of SAR segmentation, class imbalance, and geographical differences in client data distributions during federated training.

*These values are validation metrics, not final test-set results.*

---

## Technologies Used

- Python
- Jupyter Notebook
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- PyTorch
- Rasterio
- Git
- GitHub

---

## Installation and Usage

### Clone the Repository

```bash
git clone https://github.com/sangati12/Federated_learning_for_-SAR_deforestation_detection.git
```

### Navigate to the Project Folder

```bash
cd Federated_learning_for_-SAR_deforestation_detection
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Dataset Setup

The datasets are not stored in this repository. Download the required datasets from their original sources and update the dataset paths in the notebooks to match your environment.

### Run the Notebooks

Open the notebooks using Jupyter Notebook or Google Colab and run the cells in order.

---

## Dataset Sources

**Phase 1 — Deforestation Detection Dataset**

https://www.kaggle.com/datasets/akhilchibber/deforestation-detection-dataset

**Phase 3 — BraDD-S1TS**

https://github.com/kaankaramanofficial/BraDD-S1TS
