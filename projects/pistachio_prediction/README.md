# Pistachio Classification (Voting Ensemble Model)

## Project Overview
This project classifies pistachio samples based on geometric and morphological features extracted from images.
The objective is to leverage shape-based attributes to distinguish between different pistachio classes effectively.

## Model Used
- Voting Ensemble Model
- Combines multiple base classifiers
- Reduces bias and variance compared to single models

## Features Used
- Area and perimeter
- Minor axis length and eccentricity
- Equivalent diameter
- Convex area
- Aspect ratio
- Compactness
- Shape factors

## Pipeline Flow
- Feature loading & validation
- Feature scaling and normalization
- Model training
- Shape-based prediction

## Output
- Predicted pistachio class
- Useful for quality inspection and automated sorting

## Why This Project?
- Demonstrates ML applied to image-derived features
- Emphasizes feature engineering over raw image processing
- Lightweight and easy to extend

## Tech Stack
- Python
- Pandas, NumPy
- Voting Ensemble Model
- Scikit-learn