# kinematic-morphospace

[![Actions Status][actions-badge]][actions-link]
[![PyPI version][pypi-version]][pypi-link]
[![PyPI platforms][pypi-platforms]][pypi-link]

Running PCA on bird wings and tails in flight.

## Installation

```bash
python -m pip install kinematic-morphospace
```

From source:
```bash
git clone https://github.com/LydiaFrance/kinematic-morphospace
cd kinematic-morphospace
python -m pip install .
```

### For tests...

```bash
pip install -e ."[test]"
pytest tests/
```

## Usage

```python
import kinematic_morphospace

# Load and preprocess motion-capture marker data
data = kinematic_morphospace.load_data("path/to/markers.csv")
markers, frame_info = kinematic_morphospace.prepare_marker_data(data)
scaled, scaler = kinematic_morphospace.scale_data(markers)

# Correct body rotation so wing/tail shape is in a common frame
rotated = kinematic_morphospace.undo_body_rotation(scaled)

# Run PCA
pca_model, scores = kinematic_morphospace.run_PCA(rotated)

# Reconstruct shapes from the first 4 principal components
reconstructed = kinematic_morphospace.reconstruct(pca_model, scores, n_components=4)

# Visualise explained variance (requires kinematic_morphospace[plot])
kinematic_morphospace.plot_explained(pca_model)
```

Install with plotting support:

```bash
pip install "kinematic-morphospace[plot]"
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for instructions on how to contribute.

## License

Distributed under the terms of the [MIT license](LICENSE).


<!-- prettier-ignore-start -->
[actions-badge]:            https://github.com/LydiaFrance/kinematic-morphospace/workflows/CI/badge.svg
[actions-link]:             https://github.com/LydiaFrance/kinematic-morphospace/actions
[pypi-link]:                https://pypi.org/project/kinematic-morphospace/
[pypi-platforms]:           https://img.shields.io/pypi/pyversions/kinematic-morphospace
[pypi-version]:             https://img.shields.io/pypi/v/kinematic-morphospace
<!-- prettier-ignore-end -->
