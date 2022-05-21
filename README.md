# MLDevOps: Clean Code Principles

- Project **Predict Customer Churn** of ML DevOps Engineer Nanodegree Udacity

## Project Description
This project puts to practice 5 key pillars towards MLOps or prod-ready machine learning projects. These pillars include;

1. *Error handling*: try-except blocks
2. *Writing unit tests*: checking that the outcome of your software matches the expected requirements
3. *Logging*: tracking your production code for informational, warning, and error catching purposes
4. *Model drift*: the change in inputs and outputs of a machine learning model over time
5. *Automated* vs. *non-automated* retraining of ML models

Pre-requisites to these pillars include fundamental considerations of *best coding practices*:


- Writing clean and modular code
- Refactoring code
- Optimizing code for efficiency
- Writing documentation
- Following PEP8 guidelines and linting


## Project Structure & Files Description

The project directory is structured as follows:


```
├── Guide.ipynb          # Getting started and troubleshooting tips
├── churn_notebook.ipynb # Contains the code to be refactored
├── churn_library.py     # Defines the functions
├── churn_script_logging_and_tests.py # Contains tests and logs
├── README.md            # Provides project overview, and instructions to use the code
├── data                 # Read this data
│   └── bank_data.csv
├── images               # Store EDA results 
│   ├── eda
│   └── results
├── logs                 # Store logs
└── models 
```

## Running Files
To run the programs on the command line:

```
$ python churn_library.py
```

For style checking & error spotting, run code as follows on the CLI:

```
$ pylint churn_library.py
$ pylint churn_script_logging_and_tests.py
```

NOTE: Make sure you have pylint installed.


To format refactored code:

```
$ autopep8 --in-place --aggressive --aggressive churn_script_logging_and_tests.py
$ autopep8 --in-place --aggressive --aggressive churn_library.py
```

NOTE: Make sure you have autopep8 installed.

References:
- Udacity ML DevOps Engineer ND
- https://github.com/anuraglahon16