![example workflow](https://github.com/wtaisner/atla-generator/actions/workflows/python-app.yml/badge.svg)
[![uses-badges](https://img.shields.io/badge/uses-badges-blue.svg)](https://knowyourmeme.com/photos/325428-imagination-spongebob)
# Deep question-answering based on speech style in *Avatar: The Last Airbender*

## Goal
The aim of this project was to create question-answering based on deep learning that answers in style of language used 
in *Avatar: The Legend of Aang* (officially known as *Avatar: The Last Airbender*).

## Data
To solve the problem we used data available in Kaggle datasets, under the name *Avatar: The Last Airbender* 
([link](https://www.kaggle.com/datasets/ekrembayar/avatar-the-last-air-bender))

## Reproducibility and Quality Assurance
To reproduce results you have to clone the repository and run indicated source files.

Quality is assured by a CI/CD pipeline consisting of the following stages:
- linting by `flake8`
- type checking by `mypy`
- testing using `pytest`

## Exploratory Data Analysis
Three of us, namely Ania, Konrad and 
Witek are familiar with the cartoon, but one person, Jacek, has never watched even a single episode. Therefore, to 
equalize expertise and to introduce you to the ATLA world we prepared EDA. In the jupyter notebook: 
`notebooks/exploratory_data_analysis.ipynb` you can find our discoveries, among others, most frequent lemmas and 
similarities between characters.