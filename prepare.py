import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import acquire

# Create a function named prep_iris that accepts the untransformed iris data, 
# and returns the data with the transformations above applied.
def prep_iris(iris_df):
    '''
    prep_iris will take in an iris dataframe, drop species_id and measurement_id columns,
    Rename the species_name column to just species,reate dummy variables of the species 
    name and concatenate onto the iris dataframe. Then return the cleaned dataframe.
    '''
    
    # Drop the species_id and measurement_id columns.
    iris_df.drop(columns=['species_id','measurement_id'], inplace=True)
    
    # Rename the species_name column to just species.
    iris_df.rename(columns={'species_name':'species'}, inplace=True)
    
    # Create dummy variables of the species name and concatenate onto the iris dataframe.
    iris_df = pd.concat([
        iris_df,
        pd.get_dummies(iris_df['species'], drop_first=True)],
        axis=1)
    
    return iris_df


# Create a function named prep_titanic that accepts the raw titanic_df data, 
# and returns the data with the transformations above applied.
def prep_titanic(titanic_df):
    '''
    prep_titanic will drop columns 'embarked', 'class', 'deck', 'passenger_id',
    it will fill in null values in the age and embark_town columns.
    Then it will encode the sex and embark_town columns and return the cleaned df.
    '''
    
    # Drop any unnecessary, unhelpful, or duplicated columns.
    # passenger_id is unnecessary
    # embarked and class are duplicate
    # deck is unhelpful
    titanic_df = titanic_df.drop(columns=['embarked', 'class', 'deck', 'passenger_id'])
    
    # fill in the null values
    titanic_df['age'].fillna(titanic_df.age.mean(), inplace=True)
    titanic_df.embark_town.fillna('Southampton', inplace=True)
    
    # Encode the categorical columns. Create dummy variables of the categorical 
    # columns and concatenate them onto the dataframe.
    titanic_df = pd.concat([
        titanic_df,
        pd.get_dummies(titanic_df[['sex', 'embark_town']], drop_first=True)],
        axis=1)
    
    return titanic_df


# Create a function named prep_telco that accepts the raw telco_df data,
# and returns the data with the transformations above applied.
def prep_telco(telco_df):
    '''
    prep_telco will drop columns 'payment_type_id', 'internet_service_type_id',
    'contract_type_id', 'customer_id' because they are duplicate or unnecessary.
    It will then convert 'total_charges' to float type.
    Then it will encode the categorical columns to dummies, remove the now duplicated
    columns and return the cleaned df.
    '''
    # Drop any unnecessary, unhelpful, or duplicated columns. 
    # 'payment_type_id', 'internet_service_type_id', 'contract_type_id' are redundant
    # since we joined tables in the sql query
    telco_df = telco_df.drop(columns=
                       ['payment_type_id',
                        'internet_service_type_id',
                        'contract_type_id'])
    # customer_id is unnecesary, since if we don't need to identify 
    # or contact any customers
    telco_df = telco_df.drop(columns='customer_id')
    
    # total_charges is showing up as an object type when it actually should be a float
    telco_df['total_charges'] = telco_df.total_charges.replace(
        ' ','').replace('','0').astype(float)
    
    # replace yes/no column values with 1/0
    telco_df['partner'] = telco_df.partner.map({'Yes': 1, 'No': 0})
    telco_df['dependents'] = telco_df.dependents.map({'Yes': 1, 'No': 0})
    telco_df['phone_service'] = telco_df.phone_service.map({'Yes': 1, 'No': 0})
    telco_df['paperless_billing'] = telco_df.paperless_billing.map({'Yes': 1, 'No': 0})
    telco_df['churn'] = telco_df.churn.map({'Yes': 1, 'No': 0})

    # convert the categorical data into dummies
    telco_df = pd.concat(
        [telco_df, pd.get_dummies(telco_df[[
         'gender',
         'multiple_lines',
         'online_security',
         'online_backup',
         'device_protection',
         'tech_support',
         'streaming_tv',
         'streaming_movies',
         'contract_type',
         'internet_service_type',
         'payment_type'
        ]], drop_first=True)],
                      axis=1)
    
    # now we have column names with spaces, which we don't want
    # let's rename the columns with spaces, remove ()s and lower case everything
    column_names = list(telco_df.columns)
    new_names = []
    for col in column_names:
        new_names.append(col.replace(' ', '_').replace('(','').replace(')', '').lower())
    telco_df.columns = new_names
    
    return telco_df


# Write a function to split your data into train, test and validate datasets.
def split_data(df, target=None, random_seed=4233):
    '''
    split_data will take in a DataFrame and a stratify target (default to None)
    random_seed is also asignable (default = 4233 for no reason).
    It will return the data split up for ML models. 
    The return values are: train, validate, test
    '''
    
    # if we are looking for a specific stratify target in the df, 
    # asign a variable with the target name else set variable as None
    if target != None:
        strat1 = df[target]
    else:
        strat1 = None
        
    # split our df into train_val and test:
    train_val, test = train_test_split(df,
                                       train_size=0.8,
                                       random_state=random_seed,
                                       stratify=strat1)
    
    # if we are looking for a specific stratify target in the train_val, 
    # asign a variable with the target name else set variable as None
    if target != None:
        strat2 = train_val[target]
    else:
        strat2 = None
        
    # split our train_val into train and validate:
    train, validate = train_test_split(train_val,
                                   train_size=0.7,
                                   random_state=random_seed,
                                   stratify=strat2)
    
    return train, validate, test

# encasulating the whole data wrangling pipeline into a single function
def wrangle_iris():
    iris =  prep_iris(acquire.get_iris_data())
    train_iris, validate_iris, test_iris = split_data(iris, target='species')
    return train_iris, validate_iris, test_iris
def wrangle_titanic():
    titanic = prep_titanic(acquire.get_titanic_data())
    train_titanic, validate_titanic, test_titanic = split_data(titanic, target='survived')
    return train_titanic, validate_titanic, test_titanic
def wrangle_telco():
    telco =  prep_telco(acquire.get_telco_data())
    train_telco, validate_telco, test_telco = split_data(telco, target='churn')
    return train_telco, validate_telco, test_telco

# example of calling the wrangle pipeline for multiple options of datasets
my_functs = {'iris': wrangle_iris,
            'titanic': wrangle_titanic,
            'telco': wrangle_telco}
def wrangle_data(command_string):
    return my_functs[command_string]()