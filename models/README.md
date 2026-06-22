# Local model files

The backend never downloads models and never calls an inference API.

Copy the speech model into `iSpeak_v3/model_files/` and the contextual filler
classifier into `filler_classifier/`. The placeholder folders are committed;
the actual weights are not currently available in this workspace.

Large model files are configured for Git LFS so they can be shared with the
team without exceeding normal Git blob limits.
