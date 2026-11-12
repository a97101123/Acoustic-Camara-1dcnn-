# Current pipeline

## Ownership

- feature_builder.py validates paired CSV inputs and publishes immutable, split-independent Feature Sets.
- trainer_batch.py splits whole tools, fits training-only label thresholds, selects features, trains models and writes evaluations.
- features/analysis_sampling.py builds synchronized short-window features and retains every complete chunk.
- features/reader.py reads current three-dimensional split views with feature_set_identity. It has no legacy fusion resolver.
- training.py and model.py own the CNN and training-only median/IQR scaling.
- evaluation.py and classification.py own chunk evaluation and shared decisions.
- inference.py accepts only current chunk deployments with explicit all_complete metadata.
- workflow.py runs build-features, train-batch and render-training-report in order and passes actual output directories between steps.
- gui.py exposes the same current CLI commands.

## Inputs and outputs

Configuration paths resolve relative to the YAML file. WAV paths resolve against input.data_root. Builder outputs are published under features/<process>/<unique run>; Trainer outputs go under results/<process>/<unique run>.

Only completed Feature Sets with matching file hashes can train. Old Feature Sets with missing retention metadata or block_size are rejected. Existing output directories are never upgraded in place. Resume requires the same pipeline revision, features and training settings.

The four-process launcher validates both inputs before extracting. Full runs cover four conditions, seven NFFTs and two methods, with three seeds per setting. R² selections without candidates fail explicitly.

## Source conventions

Turning uses one record per measurement, one segment per recording, and post-machining WearValue copied unchanged. Drilling source records retain S1/S2/S3 with five segments each. Neither process invents per-chunk wear interpolation.

The same numeric MIC/ACC crop bounds do not independently prove recording-device synchronization. Confirm the acquisition time origin when adopting new data.
