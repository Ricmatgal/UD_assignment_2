# Disaster Response Pipeline Project

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`


# Summary 
A web app based on a ML pipeline trained and validated to classify public messages asking for aid during disasters and emergencies concerning whether and which relevant agencies to call upon. 

### Data
* The data (disaster_categories.csv and disaster_messages.csv) contains text messages gathered by FigureEight Inc. after major disasters and labelled into 36 different categories of victim needs. 

### ETL Pipeline
* The ETL Pipeline (process_data.py) etracts and clean data from the .csv files. The clean data is then loaded into an SQlite DB (data/DisasterResponse.db).

### ML Pipeline
* The ML pipeline (train_classifier.py), uses a  **Random Forest Classifier** as estimator, fitted on the presented data. It then returns a model (models/classifier.pkl) used from the web app and metrics of the model validation (models/model_metrics.csv).

### Web App 
* The web app allows a user to input a message. The trained model returns the disaster categories for which the inputed message is relevant.

# File Description 

### Structure
```
- app
| - template
| |- master.html  # main page of web app
| |- go.html  # classification result page of web app
|- run.py  # Flask file that runs app

- data
|- disaster_categories.csv  # data categories to process 
|- disaster_messages.csv  # data text messages to process
|- process_data.py  # ETL pipeline script
|- DisasterResponse.db  # database to save clean data to (generated from process_data.py)

- models
|- train_classifier.py  # ML pipeline script
|- classifier.pkl  # saved model (generated from train_classifier.py)
|- model_metrics.csv # metrics of the trained model, used also for visualization (generated from train_classifier.py)

- README.md


# Credits 
* Data provided by <a href = https://appen.com/> FigureEight, now appen </a>
* Materials provided by <a href = udacity.com> UDACITY </a> 

