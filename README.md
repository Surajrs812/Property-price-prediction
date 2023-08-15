# Real Estate Price Prediction Project

This repository contains a Python project that focuses on predicting real estate property prices in Bengaluru. The project integrates data cleaning, feature engineering, and machine learning to create a predictive model.

## Table of Contents

- [Introduction](#introduction)
- [Data Cleaning](#data-cleaning)
- [Feature Engineering](#feature-engineering)
- [Machine Learning](#machine-learning)
- [Flask Web App](#flask-web-app)
- [How to Use](#how-to-use)
- [Contributing](#contributing)

## Introduction

Real estate prices can be influenced by numerous factors, making accurate price prediction challenging. This project aims to create a predictive model that takes into account various property attributes and location data to estimate property prices in Bengaluru.

## Data Cleaning

The project begins by cleaning the raw dataset, which includes handling missing values and converting data into a usable format. Outliers in property sizes and prices are also identified and managed.

## Feature Engineering

Feature engineering involves transforming and creating new features from the dataset to enhance the performance of the predictive model. This includes converting non-uniform property size data into numerical values and generating new features based on property attributes.

## Machine Learning

A linear regression model is trained using the cleaned and engineered dataset. The model is evaluated using cross-validation techniques to ensure its predictive performance.

## Flask Web App

The project includes a Flask-based web application that integrates the predictive model. Users can input property details through the web interface, and the app will provide an estimated property price based on the trained model.

## How to Use

1. Clone the repository to your local machine.
2. Set up a Python virtual environment and install the required dependencies using `pip install -r requirements.txt`.
3. Run the Flask web app using `python app.py`.
4. Access the web app through your browser and provide property details to get a predicted price.

## Contributing

Contributions are welcome! If you find any issues or have suggestions for improvements, feel free to create a pull request.


