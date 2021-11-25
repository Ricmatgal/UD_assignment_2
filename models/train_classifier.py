# general imports
import sys
import nltk
from sqlalchemy import create_engine
import re
import pandas as pd
import pickle
import os
import time

# nltk donwloads and import
nltk.download(['punkt', 'wordnet'], quiet = True)
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords

# scikit-learn imports
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.multioutput import MultiOutputClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

# suppress warnings
import warnings
warnings.filterwarnings("ignore")

# load data from database
def load_data(database_filepath):
    """This function loads the data from the process_data output database and returns X,Y and target categories
    for the ML pipeline
    Input:
            database_filepath: relative or absolute filepath of the database
    Output:
            X: Feature values
            Y: Feature labels
            category_names = target categories"""
    engine = create_engine('sqlite:///'+ database_filepath)
    #   extract table name from database name
    table_name = os.path.basename(database_filepath).split('.')[0]
    #   Load cleaned data into SQL engine, replacing data in database if specified
    #   name already exists.
    df = pd.read_sql_table(table_name, engine)
    X = df['message']
    Y = df.drop(columns = ['id','message','original','genre'])
    category_names = Y.columns
    return X,Y,category_names

# text items tokenization
def tokenize(text):
    """This function cleans the messages input into suitable form for the ML pipeline.
    Steps: eliminate non charachters values, lemmatize and remove stopwords
    Input:
            text: text values to tokenize
    Output:
           tokens: list of tokens from text """
    # clean text from non-needed characthers
    text = re.sub(r"[^a-zA-Z]"," ",text)
    
    # tokenize text
    tokens = word_tokenize(text)
    
    # initiate lemmatizer
    lemmatizer = WordNetLemmatizer()

    # iterate through each token
    clean_tokens = []
    for tok in tokens:
        
        # lemmatize, normalize case, and remove leading/trailing white space
        clean_tok = lemmatizer.lemmatize(tok.lower().strip())
        clean_tokens.append(clean_tok)
    
    # remove stopwords
    clean_tokens = [tok for tok in clean_tokens if tok not in stopwords.words('english')]

    return clean_tokens

# build the grid search ML pipeline
def build_model():
    """This function defines the pipeline for the ML model, consisting of a CountVectorizer, Tf-Idf and a multi_clf
    with RandomForestClassifiers (default n_estimators = 200)"""
    pipeline = Pipeline([

                ('vect', CountVectorizer(tokenizer=tokenize)),
                ('tf_idf', TfidfTransformer()),
                ('multi_clf', MultiOutputClassifier(RandomForestClassifier(n_estimators = 200)))
            ])

    # here you can try out different parameters for the grid search, in this case return grid insted of pipeline:

    parameters = {'multi_clf__estimator__n_estimators':[100,200]}
    grid = GridSearchCV(pipeline,param_grid=parameters)

    return grid
    #return pipeline

# evaluate the model using classification_report
def evaluate_model(model, X_test, Y_test, category_names):
    """This function evaluate the model with the test set and generates metrics of the classifier performance.
    Input:
            model: model created frol build_model()
            X_test: test set
            Y_test: set to use for validation
            category_name: list of the Y labels
    Output:
            metric_df: dataframe containing the metrics for each label"""


    # generate predictions from the model
    Y_pred = model.predict(X_test)
    
    # generate a dataframe containing the results from classification_report
    metric = classification_report(Y_test, Y_pred, target_names = category_names,output_dict = True)
    metric_df = pd.DataFrame(metric).transpose().reset_index(level = 0)
    metric_df.rename(columns = {'index':'category_names'},inplace = True)

    # print the report as well
    print(classification_report(Y_test, Y_pred, target_names = category_names))

    # return the metric dataframe
    return metric_df
        
# save the model
def save_model(model, model_filepath):
    """This functions saves the model in .pkl format"""
    pickle.dump(model,open(model_filepath, 'wb'))

# main routine
def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))

        # load the data
        X, Y, category_names = load_data(database_filepath)

        # split in test and train set
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()

         # training start time
        start_time = time.time()
        
        print('Training model...')
        model.fit(X_train, Y_train)

         # training time taken in min.
        print("Elapsed time: %s min." % ((time.time() - start_time)//60))
        
        # evaluate the model
        print('Evaluating model...')
        metric_df = evaluate_model(model, X_test, Y_test, category_names)
        metric_df.to_csv(os.path.dirname(model_filepath)+'/model_metrics.csv',index = False)

        # save the model
        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()