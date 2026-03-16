# Quickstart

## Installation

Install kinematic-morphospace from the repository:

```bash
pip install git+https://github.com/LydiaFrance/kinematic-morphospace.git
```

For development with plotting support:

```bash
pip install "git+https://github.com/LydiaFrance/kinematic-morphospace.git#egg=kinematic-morphospace[plot]"
```

To build the documentation locally:

```bash
pip install "git+https://github.com/LydiaFrance/kinematic-morphospace.git#egg=kinematic-morphospace[docs]"
mkdocs serve
```

## Basic Usage

```python
import kinematic_morphospace

# Load motion-capture marker data
data = kinematic_morphospace.load_marker_data("path/to/data.csv")

# Run bilateral shape PCA
pca_result = kinematic_morphospace.run_bilateral_pca(data)

# Inspect variance explained by each mode
print(pca_result.explained_variance_ratio_)
```

Refer to the [API Reference](api/index.md) for detailed module documentation, or explore the example notebooks in the repository for complete workflows.
