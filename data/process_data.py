# imports needed
import sys
import pandas as pd
import os
from sqlalchemy import create_engine

# load data function
def load_data(messages_filepath, categories_filepath):
    """This function loads the messages and categories CSV files from figure 8 and merges them in a single df 
    Input: messages_filepath (absolute or relative path of messages CSV file), 
            categories filepath (absolute or relative path of categories CSV file)
    Output: Merge df 
    """
    messages = pd.read_csv(messages_filepath) # load messages
    categories = pd.read_csv(categories_filepath) # load categories
    df = messages.merge(categories, on='id') # merge in a df
    return df

# cleaning data function
def clean_data(df):
    """This function cleans the provided df in a suitable form for the classifier
    Input: df
    Output: df
    """
    categories = df.categories.str.split(pat = ';',expand = True) # split the categories in separate columns
    categories.columns = categories.iloc[0,:].apply(lambda x : x[:-2]) # create column names from the values in the first row
    
    for column in categories:

    # set each value to be the last character of the string
        categories[column] = categories[column].str[-1]
    # convert column from string to numeric
        categories[column] = pd.to_numeric(categories[column])
    # drop the previous categories column with the newly created columns
    df.drop(columns = ['categories'], inplace = True)
    df = pd.concat([df,categories], axis = 'columns')
    # remove duplicates
    df.drop_duplicates(subset=['message'],inplace=True)
    # remove rows with a non-binary related output
    df =  df[(df['related']!=2)]
    return df

# saving data function
def save_data(df, database_filename):
    """This function saves the dataframe/table in a sqlite database format
    Input: df,
            database filename (.db)
    Output: None"""
    engine = create_engine('sqlite:///'+ database_filename)
    #   extract table name from database name
    table_name = os.path.basename(database_filename).split('.')[0]
    #   Load cleaned data into SQL engine, replacing data in database if defined 
    #   name already exists.
    df.to_sql(table_name, engine, index=False,if_exists='replace')

# main routine
def main():

    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()