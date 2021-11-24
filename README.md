# Disaster Response Pipeline Project

# Summary:
A web app based on a ML pipeline trained and validated to classify public messages asking for aid during disasters and emergencies concerning whether and which relevant agencies to call upon. 

### Data:
* The data (disaster_categories.csv and disaster_messages.csv) contains text messages gathered by FigureEight Inc. after major disasters and labelled into 36 different categories of victim needs. 

### ETL Pipeline:
* The ETL Pipeline (process_data.py) etracts and clean data from the .csv files. The clean data is then loaded into an SQlite DB (data/DisasterResponse.db).

### ML Pipeline:
* The ML pipeline (train_classifier.py), uses a  **Random Forest Classifier** as estimator, fitted on the presented data. It then returns a model (models/classifier.pkl) used from the web app and metrics of the model validation (models/model_metrics.csv).

### Web App:
* The web app allows a user to input a message. The trained model returns the disaster categories for which the inputed message is relevant.

# File Description:

### Structure:
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
```

# Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

# Outputs:
1. ETL pipeline:

```
Loading data...
    MESSAGES: data/disaster_messages.csv
    CATEGORIES: data/disaster_categories.csv
Cleaning data...
Saving data...
    DATABASE: data/DisasterResponse.db
Cleaned data saved to database!
```
2. ML pipeline:

```
Loading data...
    DATABASE: data/DisasterResponse.db
Building model...
Training model...
Elapsed time: 11.0 min.
Evaluating model...
                        precision    recall  f1-score   support

               related       0.85      0.95      0.90      4005
               request       0.83      0.50      0.63       901
                 offer       0.00      0.00      0.00        20
           aid_related       0.74      0.69      0.71      2183
          medical_help       0.59      0.08      0.15       403
      medical_products       0.75      0.09      0.17       287
     search_and_rescue       0.90      0.06      0.12       140
              security       0.00      0.00      0.00        94
              military       0.69      0.05      0.10       171
           child_alone       0.00      0.00      0.00         0
                 water       0.90      0.39      0.55       338
                  food       0.82      0.63      0.71       569
               shelter       0.81      0.39      0.53       476
              clothing       0.50      0.06      0.11        81
                 money       0.60      0.02      0.04       129
        missing_people       1.00      0.02      0.03        66
              refugees       0.67      0.05      0.09       161
                 death       0.77      0.21      0.33       241
             other_aid       0.52      0.03      0.06       698
infrastructure_related       0.50      0.01      0.01       350
             transport       0.90      0.08      0.14       233
             buildings       0.86      0.14      0.25       251
           electricity       0.57      0.04      0.07       104
                 tools       0.00      0.00      0.00        33
             hospitals       0.00      0.00      0.00        70
                 shops       0.00      0.00      0.00        23
           aid_centers       0.00      0.00      0.00        55
  other_infrastructure       0.33      0.00      0.01       236
       weather_related       0.85      0.73      0.79      1433
                floods       0.93      0.53      0.67       414
                 storm       0.75      0.55      0.64       478
                  fire       0.00      0.00      0.00        51
            earthquake       0.90      0.83      0.87       486
                  cold       0.81      0.13      0.22       100
         other_weather       0.47      0.03      0.05       274
         direct_report       0.76      0.37      0.50      1020

             micro avg       0.82      0.54      0.65     16574
             macro avg       0.57      0.21      0.26     16574
          weighted avg       0.76      0.54      0.58     16574
           samples avg       0.67      0.49      0.52     16574

Saving model...
    MODEL: models/classifier.pkl
Trained model saved!
```

3. Webapp:

```
* Serving Flask app 'run' (lazy loading)
 * Environment: production
   WARNING: This is a development server. Do not use it in a production deployment.
   Use a production WSGI server instead.
 * Debug mode: on
 * Running on all addresses.
   WARNING: This is a development server. Do not use it in a production deployment.
 * Running on http://192.168.1.143:3001/ (Press CTRL+C to quit)
 * Restarting with stat
 * Debugger is active!
 * Debugger PIN: 923-609-621
```



# Credits 
* Data provided by <a href = https://appen.com/> FigureEight, now appen </a>
* Materials provided by <a href = udacity.com> Udacity inc. </a> 

