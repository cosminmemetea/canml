# canml: A Python Library for Preparing CAN Bus Data for Machine Learning

## Description

`canml` is a Python library designed to facilitate the preparation of Controller Area Network (CAN) bus data for machine learning applications. It provides tools to parse BLF (Bus Log Format) files using CAN.DBC files, preprocess the data, extract features, and export it in formats suitable for machine learning, such as CSV, Excel, and pandas DataFrames. Additionally, it offers visualization tools to aid in data exploration and understanding.

**Why `canml`?**

While libraries like `cantools` and `python-can` excel in parsing and decoding CAN bus data, they do not focus on preparing this data for machine learning workflows. `canml` fills this gap by providing specialized functions for:

- Preprocessing time-series CAN data (e.g., handling missing values, resampling).
- Extracting features relevant for machine learning (e.g., statistical summaries, frequency-domain features).
- Exporting data in machine learning-ready formats.
- Integrating with popular machine learning libraries like scikit-learn.

This makes `canml` particularly useful for engineers and data scientists working on applications such as anomaly detection, predictive maintenance, and driver behavior analysis in automotive and industrial settings.

## Features

- Parse BLF files using CAN.DBC files to decode CAN messages into meaningful signals.
- Preprocess data:
  - Handle missing values with interpolation or forward fill.
  - Resample time-series data to a uniform grid.
  - Normalize or standardize signals.
- Extract features:
  - Statistical features (mean, standard deviation, min, max) over sliding windows.
  - Frequency-domain features (FFT, power spectral density).
  - Custom feature engineering based on domain knowledge.
- Export data to:
  - CSV and Excel for reporting and sharing.
  - pandas DataFrames and NumPy arrays for machine learning.
- Visualization:
  - Time-series plots of signals.
  - Histograms and correlation matrices for data exploration.
- Machine learning integration:
  - Scikit-learn-compatible API for use in ML pipelines.
  - High-level functions for common tasks like anomaly detection.

## Installation

To install `canml`, use pip:

```bash
pip install canml
```

**Dependencies**:
- Python 3.7+
- pandas
- NumPy
- scikit-learn
- matplotlib
- cantools (optional, for enhanced BLF parsing)

## Usage

Below is an example of how to use `canml` to load a BLF file, preprocess the data, extract features, and export it to CSV.

```python
import canml
import pandas as pd

# Load BLF and DBC files
data = canml.io.load_blf('data.blf', 'config.dbc')

# Preprocess data
data_clean = canml.preprocess.handle_missing(data, method='interpolate')
data_resampled = canml.preprocess.resample(data_clean, freq='100ms')
data_normalized = canml.preprocess.normalize(data_resampled)

# Extract features
features = canml.features.extract_stats(data_normalized, window='1s', stats=['mean', 'std'])

# Export to CSV
canml.export.to_csv(features, 'output.csv')

# Visualize data
canml.viz.plot_timeseries(data_normalized, signals=['EngineRPM', 'VehicleSpeed'])
```

For advanced usage, such as integrating with scikit-learn for anomaly detection, refer to the [documentation](https://canml.readthedocs.io/).

## Contributing

Contributions are welcome! To contribute:

1. Fork the repository on GitHub.
2. Create a new branch for your feature or bug fix.
3. Submit a pull request with a clear description of your changes.

Please open an issue to discuss major changes before starting work.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Credits

- Inspired by `cantools` and `python-can` for CAN bus parsing.
- Built using [pandas](https://pandas.pydata.org/), [NumPy](https://numpy.org/), [scikit-learn](https://scikit-learn.org/stable/), and [matplotlib](https://matplotlib.org/) for data manipulation, machine learning, and visualization.
- Special thanks to the Python community for their open-source contributions.

## Contact

For questions or support, please open an issue on the [GitHub repository](https://github.com/canml/canml).