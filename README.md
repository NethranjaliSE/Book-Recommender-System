# 📚 Book Recommendation System

A Machine Learning project that suggests books to users based on collaborative filtering. This system uses the **Nearest Neighbors** algorithm to find clusters of similar books based on user ratings. The project is structured with a modular MLOps pipeline including data ingestion, validation, transformation, and model training.

## 🚀 Overview

This project implements a complete end-to-end Machine Learning pipeline:
1.  **Ingests** raw data from a source URL.
2.  **Validates** and cleans the data (removing inactive users and unpopular books).
3.  **Transforms** data into a sparse matrix (Pivot Table).
4.  **Trains** a K-Nearest Neighbors (KNN) model.
5.  **Recommends** books similar to a given input.

## 🛠️ Tech Stack

* **Language:** Python 3.x
* **Libraries:** Scikit-learn, Pandas, NumPy, SciPy
* **Utilities:** Joblib/Pickle (Model Serialization), Python Logging
* **Architecture:** Modular OOP (Object Oriented Programming)

## How to Run Project 

### Step 01: Clone the Repository

### Step 02: Create and Activate Conda Environment
conda create -n books python=3.7.10 -y

conda activate books

### Step 03: Install Required Dependencies
pip install -r requirements.txt

###  04: Run the Streamlit Application
streamlit run app.py


## 📂 Project Structure

```text
├── books_recommender
│   ├── components          # Core logic (Ingestion, Validation, Transformation, Training)
│   ├── config              # Configuration manager
│   ├── constants           # Constant variables
│   ├── entity              # Data classes (Config Entities)
│   ├── exception           # Custom Exception Handling
│   ├── logger              # Custom Logging
│   ├── pipeline            # Training & Prediction Pipelines
│   └── utils               # Utility functions
├── config                  # YAML configuration files
├── logs                    # Log files storage
├── requirements.txt        # Python dependencies
├── main.py                 # Main execution script
└── README.md


