# API Reference

kinematic-morphospace is organised into focused modules covering the full analysis pipeline from data loading through to visualisation.

| Module | Description |
|--------|-------------|
| [`data_loading`](data_loading.md) | Load and parse motion-capture marker data |
| [`data_scaling`](data_scaling.md) | Scale and normalise marker coordinates |
| [`data_filtering`](data_filtering.md) | Filter and clean marker trajectories |
| [`pca_core`](pca_core.md) | Core bilateral shape PCA |
| [`pca_scores`](pca_scores.md) | Extract and manipulate PCA scores |
| [`pca_reconstruct`](pca_reconstruct.md) | Reconstruct shapes from PCA components |
| [`rotation`](rotation.md) | Whole-body rotation correction |
| [`validation`](validation.md) | Bootstrap, permutation, and robustness tests |
| [`species_transform`](species_transform.md) | Transform data between species morphologies |
| [`cross_species`](cross_species.md) | Cross-species generalisation analysis |
| [`labelling`](labelling.md) | Behavioural and flight-mode labelling |
| [`clustering`](clustering.md) | Clustering of shape modes and flight behaviours |
| [`null_testing`](null_testing.md) | Null-model and significance testing |
| [`plotting`](plotting.md) | Publication-ready visualisations |
