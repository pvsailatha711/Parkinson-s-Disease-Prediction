# Parkinson's Disease Prediction

## Overview

This project develops a machine learning model to predict the presence of Parkinson's Disease using voice measurement data. By analyzing various acoustic features from voice recordings, the model can identify patterns that distinguish individuals with Parkinson's Disease from healthy individuals. The project employs advanced feature selection techniques to identify the most significant voice biomarkers for accurate disease prediction.

## Dataset

The dataset is sourced from the UCI Machine Learning Repository and contains voice measurements from individuals with and without Parkinson's Disease.

### Dataset Characteristics

- **Total Samples**: 195 voice recordings
- **Participants**: 31 individuals (23 with Parkinson's Disease, 8 healthy)
- **Features**: 24 columns including various voice measurements
- **Target Variable**: `status` (1 = Parkinson's Disease, 0 = Healthy)

### Feature Categories

**1. Fundamental Frequency Measures**
- `MDVP:Fo(Hz)`: Average vocal fundamental frequency
- `MDVP:Fhi(Hz)`: Maximum vocal fundamental frequency
- `MDVP:Flo(Hz)`: Minimum vocal fundamental frequency

**2. Jitter Measures** (Frequency variation)
- `MDVP:Jitter(%)`, `MDVP:Jitter(Abs)`: Measures of variation in fundamental frequency
- `MDVP:RAP`, `MDVP:PPQ`, `Jitter:DDP`: Additional jitter measurements

**3. Shimmer Measures** (Amplitude variation)
- `MDVP:Shimmer`, `MDVP:Shimmer(dB)`: Measures of variation in amplitude
- `Shimmer:APQ3`, `Shimmer:APQ5`, `MDVP:APQ`, `Shimmer:DDA`: Additional shimmer measurements

**4. Harmonicity Measures**
- `NHR`: Noise-to-Harmonics Ratio
- `HNR`: Harmonics-to-Noise Ratio

**5. Nonlinear Dynamical Complexity Measures**
- `RPDE`: Recurrence Period Density Entropy
- `DFA`: Detrended Fluctuation Analysis
- `spread1`, `spread2`: Nonlinear measures of fundamental frequency variation
- `D2`: Correlation dimension
- `PPE`: Pitch Period Entropy

## Methodology

### Data Preprocessing

1. **Feature Extraction**: Removed identifier column (`name`) and separated target variable (`status`)
2. **Feature Scaling**: Applied StandardScaler to normalize all features for equal contribution to model training
3. **Train-Test Split**: 75-25 split (146 training samples, 49 test samples)

### Feature Selection

The project implements **Recursive Feature Elimination with Cross-Validation (RFECV)** to identify the most predictive features:

- **Algorithm**: Random Forest estimator with 5-fold cross-validation
- **Optimal Feature Count**: 7 features selected from the original 22
- **Selected Features**:
  1. `MDVP:Fo(Hz)` - Average vocal fundamental frequency
  2. `Shimmer:APQ5` - Five-point amplitude perturbation quotient
  3. `MDVP:APQ` - Amplitude perturbation quotient
  4. `Shimmer:DDA` - Average absolute difference of differences between amplitudes
  5. `spread1` - Nonlinear measure of fundamental frequency variation
  6. `spread2` - Nonlinear measure of fundamental frequency variation
  7. `PPE` - Pitch Period Entropy

This feature selection process significantly improved model focus by reducing dimensionality while retaining the most informative voice biomarkers.

### Model Architecture

**Algorithm**: Random Forest Classifier

**Hyperparameters**:
- `n_estimators`: 680 trees
- `criterion`: 'entropy' (information gain)
- `random_state`: 0 (for reproducibility)

Random Forest was chosen for its ability to:
- Handle complex, non-linear relationships in voice data
- Provide feature importance rankings
- Reduce overfitting through ensemble learning
- Maintain high accuracy with relatively small datasets

## Results

The model demonstrates excellent performance in predicting Parkinson's Disease:

### Overall Performance
- **Accuracy**: 91.84%

### Confusion Matrix

|                    | Predicted Healthy | Predicted PD |
|--------------------|-------------------|--------------|
| **Actual Healthy** | 9                 | 2            |
| **Actual PD**      | 2                 | 36           |

### Error Analysis
- **True Positives**: 36 (correctly identified PD cases)
- **True Negatives**: 9 (correctly identified healthy individuals)
- **False Positives**: 2 (Type I Error - healthy individuals incorrectly flagged)
- **False Negatives**: 2 (Type II Error - PD cases missed)

The model shows balanced performance across both classes, successfully identifying the vast majority of Parkinson's Disease cases while maintaining low false alarm rates.

## Key Findings

1. **Feature Importance**: Voice measurements, particularly frequency and amplitude variations, serve as reliable biomarkers for Parkinson's Disease detection

2. **Optimal Feature Set**: Only 7 out of 22 features are necessary for high-accuracy prediction, demonstrating that specific voice characteristics are particularly indicative of the disease

3. **Clinical Relevance**: The selected features align with known symptoms of Parkinson's Disease, which affects motor control including vocal cord function

4. **Non-Invasive Screening**: Voice analysis provides a simple, non-invasive method for preliminary Parkinson's Disease screening

## Technologies Used

- **Python**: Core programming language
- **Scikit-learn**: Machine learning framework
  - RandomForestClassifier: Primary classification algorithm
  - RFECV: Feature selection
  - StandardScaler: Data normalization
  - train_test_split: Data partitioning
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computations
- **Matplotlib**: Data visualization
- **Seaborn**: Statistical data visualization

## Acknowledgments

Dataset sourced from the UCI Machine Learning Repository. Original data collected by Max Little of the University of Oxford, in collaboration with the National Centre for Voice and Speech, Denver, Colorado.
