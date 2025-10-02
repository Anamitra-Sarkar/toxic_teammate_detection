# Toxic Teammate Detection

A professional-grade web application to detect toxic teammates using machine learning. This project leverages a trained model to analyze team member behavior and provide predictions on toxicity, helping to foster healthier and more productive team environments.

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [How It Works](#how-it-works)
- [Getting Started](#getting-started)
- [Usage](#usage)
- [Model Details](#model-details)
- [Project Structure](#project-structure)
- [License](#license)

---

## Overview

Toxicity within teams can undermine productivity, morale, and project success. This repository provides a solution to proactively identify potentially toxic behaviors using a machine learning model, presented through an easy-to-use web interface built with Flask.

---

## Features

- **Web-Based Interface**: Intuitive web app powered by Flask.
- **Machine Learning Model**: Automated prediction of toxicity based on behavioral inputs.
- **Customizable Input Form**: Supports a wide range of behavioral metrics.
- **Instant Results**: Get prediction and probability scores in real-time.
- **Extensible Design**: Easily update model and feature columns for evolving needs.

---

## How It Works

1. **User Inputs**: The frontend form collects behavioral data (e.g., meeting attendance, deadline adherence, communication respect).
2. **Backend Processing**: 
    - Inputs are processed and encoded to match the trained features.
    - The `toxic_teammate_model.joblib` is loaded and used for prediction.
3. **Prediction Output**: The app returns a toxicity prediction and associated probabilities.

---

## Getting Started

### Prerequisites

- Python 3.7+
- [Flask](https://pypi.org/project/Flask/)
- [Flask-CORS](https://pypi.org/project/Flask-Cors/)
- [pandas](https://pypi.org/project/pandas/)
- [joblib](https://pypi.org/project/joblib/)

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Anamitra-Sarkar/toxic_teammate_detection.git
   cd toxic_teammate_detection
   ```
2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Model Placement:**
   - Ensure `model/toxic_teammate_model.joblib` exists. Place your trained model here.

4. **Templates:**
   - Place your HTML templates (e.g., `index.html`) in the `templates/` directory.

---

## Usage

1. **Run the Flask app:**
   ```bash
   python app.py
   ```
2. **Access the web app:**
   - Open your browser and go to [http://localhost:5000](http://localhost:5000).

3. **Submit behavioral data via the form.**
   - Receive instant toxicity prediction and probability.

---

## Model Details

- **Model File:** `model/toxic_teammate_model.joblib`
- **Feature Columns:** The model expects one-hot encoded features such as:
  - Missed Meetings (Frequency)
  - Deadline Adherence
  - Contribution Quality
  - Responsiveness
  - Communication Respect
  - Workload Fairness (Perception)
  - Discussion Participation
  - Credit Taking
  - Conflict/Negativity
  - Harsh Criticism
  - Rework Required

- **Prediction Output:** Returns whether a teammate is likely toxic or not, along with probability scores.

---

## Project Structure

```
toxic_teammate_detection/
├── app.py                      # Main Flask application
├── model/
│   └── toxic_teammate_model.joblib  # Trained ML model (Joblib format)
├── templates/
│   └── index.html              # HTML frontend (place your templates here)
├── LICENSE                     # License file
├── README.md                   # Project documentation
```

---

## License

This project is licensed under the terms of the [MIT License](LICENSE).

---

## Contributing

Feel free to fork the repository, open issues, and submit pull requests. For major changes, please open an issue first to discuss what you would like to change.

---

## Author

- [Anamitra Sarkar](https://github.com/Anamitra-Sarkar)

---

## Acknowledgments

- Flask web framework
- scikit-learn for model training
- pandas for data handling

---

*Empower your teams with data-driven insights for a healthier workplace!*
