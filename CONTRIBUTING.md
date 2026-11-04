# Contributing

## Before opening a pull request

1. Check for an existing issue or open pull request describing the change.
2. Create a branch with a short, descriptive name.
3. Keep changes focused and explain user-visible or model-output changes.
4. Run the application and exercise the affected path with the provided sample data.

When changing feature generation or preprocessing, preserve the feature order and shape
expected by the stored model artifacts, or update and retrain the models as part of the
change. Do not describe prototype predictions as validated clinical results.

Open a pull request with a summary of the change, how it was tested, and any known
limitations.
