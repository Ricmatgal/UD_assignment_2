import json
import plotly
import pandas as pd
import re
import joblib

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar, Scatter
from sqlalchemy import create_engine


app = Flask(__name__)

def tokenize(text):
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

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('DisasterResponse', engine)
metrics = pd.read_csv("../models/model_metrics.csv")

# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    
    # create visuals
    # 2 graphs, the first one is the message genres overview, while the second is the classifier performance
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Categories',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                },
                'template': "plotly_dark"
            }

        },
        {
            'data': [
                Scatter(
                    name = 'Precision',
                    x = metrics['category_names'],
                    y = metrics['precision'],
                    mode = 'lines'
                ),
                Scatter(
                    name = 'Recall',
                    x = metrics['category_names'],
                    y = metrics['recall'],
                    mode = 'lines'
                ),
                Scatter(
                    name = 'F1 Score',
                    x = metrics['category_names'],
                    y = metrics['f1-score'],
                    mode = 'lines'
                )
            ],

            'layout':{
                'title': 'Random Forest Model Performance Metrics',
                "xaxis":{
                    'title': 'Categories',
                    'title_standoff': 100,
                    'tickangle': 45
                },
                "yaxis":{
                     'title': ""
                },
                "template": "plotly_dark"
            }
        }

    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()