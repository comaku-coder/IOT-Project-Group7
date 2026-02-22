# Solar Energy Forecasting with LSTM-Attention Models

## IOT-Project-Group7 — AAI530 Final Course Project

---

## 📋 Project Overview

This project implements a machine learning IoT application for **real-time and predictive forecasting of solar photovoltaic (PV) system performance** using high-resolution sensor data from the **Desert Knowledge Australia Solar Centre (DKASC)**.

### Background: Desert Knowledge Australia Solar Centre (DKASC)

The **DKASC** is a large multi-technology solar research and demonstration facility located in **Alice Springs, Northern Territory, Australia**. Established in 2008 with Australian government funding, it serves as an open-access testbed for diverse photovoltaic (PV) and related technologies deployed under harsh arid conditions.

**Key Facts:**

- **Location:** Desert Knowledge Precinct, Alice Springs, Northern Territory, Australia
- **Scale:** 40+ PV installations; 20+ panel manufacturers represented
- **Mission:** Provide freely accessible, high-quality, long-term datasets on solar resource, climate conditions, and system performance
- **Purpose:** Help researchers refine models, support developers in technology choice, and inform standards and policy for desert and remote-region solar applications

By leveraging very high solar irradiance and extreme temperatures, DKASC stress-tests technologies beyond what many coastal sites experience, making it an ideal source for robust ML model development.

---

## 🎯 Project Objectives

As a final team project for **AAI530 (IoT and Machine Learning)**, this application demonstrates:

1. **IoT System Design:** Complete theoretical IoT system architecture for real-world solar monitoring
2. **Data Collection & Processing:** Handling raw sensor data with 5-minute resolution from multiple instruments
3. **Feature Engineering:** Creating domain-specific features (cyclical encodings, rolling statistics, temporal differences)
4. **Machine Learning:** Training advanced deep learning models (LSTM with Attention mechanisms) for forecasting
5. **Performance Analysis:** Comprehensive evaluation metrics and visualization of model predictions
6. **Data Visualization:** Interactive Tableau Public dashboards for stakeholder insights

---

## 📊 Dataset

**Source:** Desert Knowledge Australia Solar Centre (DKASC)  
**File:** `Data/8-Site_CA-PV1-DB-CA-1A.csv`  
**Temporal Resolution:** 5-minute intervals  
**Data Coverage:** Multiple years of solar and weather measurements

### Key Sensor Measurements

**Environmental Inputs (Features):**

- Wind Speed (m/s)
- Weather Temperature (°C)
- Global Horizontal Radiation (GHI) — W/m²
- Wind Direction (degrees)
- Daily Rainfall (mm)
- Max Wind Speed (m/s)
- Air Pressure (hPa)
- Pyranometer 1 (W/m²) — Alternative radiation sensor

**Output Targets:**

- **Active Power Output (kW)** — Solar panel system power generation
- **Panel Temperature (°C)** — Average temperature of photovoltaic panels

---

## 🏗️ Technical Approach

### Data Preprocessing Pipeline

1. **Data Cleaning:**
   - Remove duplicate timestamps and invalid readings
   - Fix known sensor accuracy issues (GHI/Pyranometer outliers)
   - Remove incomplete years (2016, 2020, 2025+)
   - Filter nighttime hours (outside 4 AM-9 PM)

2. **Missing Data Handling:**
   - Reindex to regular 5-minute intervals
   - Linear interpolation with max lag of 12 steps (1 hour)
   - Drop remaining rows with missing values

3. **Feature Engineering:**
   - **Cyclical Encoding:** Hour-of-day, day-of-year, wind direction (sin/cos pairs)
   - **Rolling Statistics:** 6, 12, 24-step rolling means and standard deviations for GHI, Pyranometer, and temperature
   - **Temporal Differences:** 5, 15, 30-step differences to capture rate of change

### Model Architecture

**Primary Model: LSTM with Attention + Conv1D**

```
Input Layer (seq_len=24, features=48)
    ↓
Conv1D (32 filters, kernel=3, ReLU, 30% dropout)
    ↓
LSTM (128 units, return_sequences=True, 30% dropout)
    ↓
Attention Layer (scaled dot-product attention)
    ↓
LSTM (64 units, 30% dropout)
    ↓
Dense (16 units, ReLU)
    ↓
Dense (8 units, ReLU)
    ↓
Output Dense (2 units) → [Active Power, Panel Temperature]
```

**Key Features:**

- **Sequence Length:** 24 steps (2 hours of historical data at 5-min intervals)
- **Attention Mechanism:** Learns temporal importance weights across history
- **Dropout:** 30% regularization to prevent overfitting
- **Optimizer:** Adam with learning_rate=0.0005
- **Loss:** Mean Squared Error (MSE)

### Training Configuration

- **Train/Validation/Test Split:** 70% / 25% / 5%
- **Batch Size:** 64
- **Max Epochs:** 200
- **Early Stopping:** Patience of 7 epochs on validation loss
- **Data Scaling:** MinMax scaling [0, 1] for both inputs and outputs

---

## 📁 Repository Structure

```
IOT-Project-Group7/
├── FinalProject.ipynb                  # Main Jupyter notebook with full pipeline
├── FinalProject.py                     # Standalone Python script version
├── README.md                           # This file — project documentation
├── LICENSE                             # Project license
│
├── Data/
│   └── 8-Site_CA-PV1-DB-CA-1A.csv     # Raw sensor data (5-min intervals, multiple years)
│
├── Models/
│   ├── model_power_forecast.keras      # Trained LSTM model (future: power forecasting)
│   └── model_temp_forecast.keras       # Trained LSTM-Attention model (panel temperature)
│
├── Training Results/
│   ├── model1_metrics.csv              # Evaluation metrics for power model
│   ├── model2_metrics.csv              # Evaluation metrics for temperature model
│   ├── training_log_model1.csv         # Epoch-by-epoch training history (power)
│   └── training_log_model2.csv         # Epoch-by-epoch training history (temperature)
│
├── Images/                             # (Directory for visualization outputs)
└── training_log_model2.csv             # Training log (also at root for convenience)
```

### File Descriptions

| File                           | Purpose                                                                                                                                                            |
| ------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **FinalProject.ipynb**         | Interactive Jupyter notebook containing the full ML pipeline: data loading, EDA, preprocessing, feature engineering, model training, evaluation, and visualization |
| **FinalProject.py**            | Production-ready Python script with the same pipeline; can be run standalone without Jupyter                                                                       |
| **8-Site_CA-PV1-DB-CA-1A.csv** | Raw sensor measurements from DKASC; contains ~5-minute resolution readings for multiple years                                                                      |
| **model_temp_forecast.keras**  | Serialized TensorFlow/Keras model for panel temperature forecasting; ready for inference                                                                           |
| **model_power_forecast.keras** | Serialized model for power output (for future expansion)                                                                                                           |
| **model1/2_metrics.csv**       | Model evaluation results: MAE, RMSE, R² scores on test set                                                                                                         |
| **training_log_model1/2.csv**  | Per-epoch training logs: loss, MAE for train and validation sets                                                                                                   |

---

## 🚀 Getting Started

### Prerequisites

- **Python 3.8+**
- **pip** or **conda** package manager

### Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/comaku-coder/IOT-Project-Group7.git
   cd IOT-Project-Group7
   ```

2. **Create and activate a virtual environment (recommended):**

   ```bash
   # Using venv
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate

   # Or using conda
   conda create -n solar-ml python=3.10
   conda activate solar-ml
   ```

3. **Install required dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

   **Key packages:**
   - pandas >= 1.3.0
   - numpy >= 1.21.0
   - scikit-learn >= 1.0.0
   - tensorflow >= 2.10.0
   - matplotlib >= 3.5.0

### Running the Project

#### Option 1: Jupyter Notebook (Interactive)

```bash
jupyter notebook FinalProject.ipynb
```

- Open the notebook in your browser
- Run cells sequentially from top to bottom
- Modify parameters and re-run cells as needed for experimentation

#### Option 2: Python Script (Batch)

```bash
python FinalProject.py
```

- Runs the entire pipeline end-to-end
- Generates plots and metrics files in the project root
- No interactive exploration required

### Expected Outputs

After running the pipeline, you will generate:

1. **Visualization Plots:**
   - Daily patterns (all features across July 2019)
   - Monthly aggregates by year
   - Model loss curves over epochs
   - Prediction vs. actual for temperature (current + 30-min forecast)

2. **Training Logs:**
   - `training_log_model2.csv` — Per-epoch metrics

3. **Model Evaluation:**
   - `model2_metrics.csv` — Performance summary (MAE, RMSE, R²)
   - Serialized `.keras` models for reuse

---

## 📈 Model Performance

### Temperature Forecasting Model (model_temp_forecast.keras)

The LSTM-Attention model predicts **panel temperature** at two timescales:

| Timeframe                  | MAE (°C) | RMSE (°C) | R²  |
| -------------------------- | -------- | --------- | --- |
| Current (t=0)              | TBD      | TBD       | TBD |
| 30 Minutes Ahead (t+30min) | TBD      | TBD       | TBD |

_(Fill in actual values after first successful run)_

### Model Insights

- **Current Temperature:** Lower error due to high temporal correlation
- **30-Minute Forecast:** Captures solar ramp rates and thermal lag effects
- **Attention Mechanism:** Learns which historical windows are most predictive
- **Cyclical Features:** Critical for capturing diurnal and seasonal patterns

---

## 🔧 Code Structure & Key Sections

### Data Loading & Cleaning

- Load CSV from DKASC
- Parse timestamps and identify gaps
- Remove outliers (GHI > 1400 W/m², Pyranometer > 2000 W/m²)
- Interpolate missing values

### Exploratory Data Analysis (EDA)

- Daily patterns (one line per day in July 2019)
- Monthly averages across years
- Identify data quality issues

### Feature Engineering

- **Temporal Features:** Year/month/day/hour extraction
- **Cyclical Encoding:** sin/cos transformations for circular features
- **Aggregations:** Rolling windows (6, 12, 24 steps) for mean and std
- **Derivatives:** Temporal differences (5, 15, 30 steps)

### Model Training

- Data normalization (MinMaxScaler)
- Sequence creation (sliding windows of 24 steps = 2 hours)
- Model compilation with Adam optimizer
- Early stopping on validation loss
- CSV logging of training history

### Evaluation & Visualization

- Multi-metric assessment (MAE, RMSE, R²)
- Plot predictions vs. actuals
- Loss curves over epochs

---

## 📊 IoT System Design

This project represents a complete **IoT system architecture:**

1. **IoT Sensors (DKASC):**
   - Distributed PV array monitoring
   - Meteorological station (wind, temperature, pressure, rainfall)
   - Radiation sensors (GHI, Pyranometer)
   - Data logged at 5-minute intervals

2. **Data Aggregation:**
   - Custom CSV export from DKASC database
   - Time-series alignment and quality checks
   - Missing data imputation

3. **Machine Learning Pipeline (This Project):**
   - Feature extraction and normalization
   - LSTM-Attention deep learning model
   - Real-time inference capability
   - Performance monitoring

4. **Downstream Applications:**
   - **Tableau Public Dashboard:** Interactive visualization for stakeholders
   - **Control Systems:** Grid operation and demand response
   - **Maintenance Alerts:** Predict panel degradation
   - **Forecasting:** Feed solar predictions into grid balancing systems

---

## 🎓 Educational Outcomes

Through this project, learners will understand:

- ✅ **IoT Data Collection:** Multi-sensor systems with temporal data
- ✅ **Time Series Preprocessing:** Handling gaps, outliers, and frequency alignment
- ✅ **Feature Engineering:** Domain-specific transformations for solar data
- ✅ **Deep Learning for Sequences:** LSTM, Attention, CNN architectures
- ✅ **Machine Learning Workflow:** Train/val/test splits, early stopping, regularization
- ✅ **Model Evaluation:** Multi-metric assessment and visualization
- ✅ **Reproducibility:** Documented code, version control, modular design

---

## 📝 Code Comments & Inline Documentation

Both `FinalProject.ipynb` and `FinalProject.py` include detailed comments throughout:

```python
# Example: Feature engineering section
# Cyclical features: encode circular features as sin/cos pairs
df['hour_sin'] = np.sin(2 * np.pi * df['hour_of_day'] / 24)  # Normalize to [0, 1]
df['hour_cos'] = np.cos(2 * np.pi * df['hour_of_day'] / 24)  # Pair for uniqueness

# Rolling window features: capture local trends
roll_windows = [6, 12, 24]  # 30 min, 60 min, 120 min windows
for window in roll_windows:
    df[f'GHI_roll_mean_{window}'] = df['Global_Horizontal_Radiation'].rolling(
        window=window,
        min_periods=1,      # Handle edges
        center=False        # Forward-looking window
    ).mean()
```

Comments explain:

- **What:** Purpose of each code block
- **Why:** Rationale behind design choices
- **How:** Parameters and their impact on results

---

## 🤝 Team & Attribution

**Course:** AAI530 — IoT and Machine Learning (Spring 2026)  
**Institution:** University of San Diego (USD)  
**Team:** Group 7

**Dataset Source:** Desert Knowledge Australia Solar Centre (DKASC), Alice Springs, NT, Australia

---

## 📜 License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.

---

## 🔗 References & Further Reading

1. **DKASC Official Site:** https://dkasc.org.au
2. **TensorFlow/Keras Documentation:** https://www.tensorflow.org/
3. **Time Series Forecasting with LSTM:** https://colah.github.io/posts/2015-08-Understanding-LSTMs/
4. **Attention Mechanisms:** https://arxiv.org/abs/1706.03762
5. **Solar Energy Forecasting Review:** Energy Reviews, Solar Power Prediction Papers

---

## 💡 Next Steps & Future Work

Potential enhancements:

1. **Model Improvements:**
   - Ensemble methods combining power + temperature predictions
   - Transformer architecture (self-attention) for longer sequences
   - Transfer learning from other solar sites

2. **Feature Expansion:**
   - Cloud cover and sky condition data (when available)
   - PV panel-specific maintenance flags
   - Satellite-based irradiance forecasts

3. **Deployment:**
   - REST API for real-time inference
   - Edge deployment on IoT gateways
   - Integration with SCADA systems

4. **Tableau Dashboard:**
   - Real-time prediction updates
   - Historical performance trends
   - Anomaly detection alerts

---

## ❓ FAQ & Troubleshooting

**Q: How do I handle missing data if I run the pipeline on new DKASC data?**  
A: The notebook includes automatic interpolation (linear, max 1 hour) and gap detection. Adjust the `limit` parameter in the `interpolate()` call for more/less aggressive filling.

**Q: Can I retrain the model on new data?**  
A: Yes! The notebook is modular—simply load a new CSV, run preprocessing, and execute the training cells. The model will fit on your new data.

**Q: How do I use the pre-trained models for inference?**  
A: Load with `tf.keras.models.load_model('model_temp_forecast.keras')` and pass preprocessed sequences (shape: `[batch, 24, 48]`).

---

## 📧 Support & Contact

For questions or issues:

- Open an issue on GitHub
- Review the inline comments in the notebook
- Check the DKASC documentation for sensor specs

---

**Last Updated:** February 22, 2026  
**Status:** Final Submission for AAI530
