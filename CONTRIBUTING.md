# Contributing to AMrit

First off, thank you for considering contributing to AMrit! It's people like you that make AMrit such a great tool for clinical decision support.

## 1. Where do I go from here?
If you've noticed a bug or have a feature request, make sure to check if there's already an open issue. If not, feel free to open a new one.

## 2. Fork & create a branch
If this is something you think you can fix, then fork AMrit and create a branch with a descriptive name.

## 3. Implementation Guidelines
- Ensure that your code adheres to PEP-8 standards.
- If you are updating the RDKit chemistry engine (`backend/chemistry_engine.py`), please ensure the 433-dimensional feature matrix remains consistent with the pre-trained XGBoost weights.
- Test your changes thoroughly on both the FastAPI backend and Streamlit frontend.

## 4. Make a Pull Request
At this point, you should switch back to your master branch, make sure it's up to date with AMrit's master branch. Then submit a Pull Request!
