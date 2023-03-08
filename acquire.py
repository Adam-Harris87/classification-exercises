import numpy as np
import pandas as pd
import os
import env

# Make a function named get_titanic_data that returns the titanic data
# from the codeup data science database as 
# a pandas data frame. Obtain your data from the Codeup Data Science Database.
def get_titanic_data():
    filename = "titanic.csv"

    #check if cached exists
    if os.path.isfile(filename):
        #return cached data
        # print('opening data from file')
        return pd.read_csv(filename)
    else:
        # read the SQL query into a dataframe
        # print('cached file not found, creating new file')
        connection = env.get_db_url('titanic_db')
        query = '''
SELECT * 
FROM passengers;
        '''
        df = pd.read_sql(query, connection)
        
        # Write that dataframe to disk for later. Called "caching" the data for later.
        df.to_csv(filename)
        # return the dataframe
        return df

# Make a function named get_iris_data that returns the data from the 
# iris_db on the codeup data science database as a pandas data frame. 
# The returned data frame should include the actual name of the species in 
# addition to the species_ids. Obtain your data from the Codeup Data Science Database.
def get_iris_data():
    filename = "iris.csv"

    #check if cached exists
    if os.path.isfile(filename):
        #return cached data
        # print('opening data from file')
        return pd.read_csv(filename)
    else:
        # read the SQL query into a dataframe
        # print('cached file not found, creating new file')
        connection = env.get_db_url('iris_db')
        query = '''
SELECT * 
FROM measurements
	JOIN species
		USING (species_id);
        '''
        df = pd.read_sql(query, connection)
        
        # Write that dataframe to disk for later. Called "caching" the data for later.
        df.to_csv(filename)
        # return the dataframe
        return df

# Make a function named get_telco_data that returns the data from the telco_churn 
# database in SQL. In your SQL, be sure to join contract_types, internet_service_types, 
# payment_types tables with the customers table, so that the resulting dataframe 
# contains all the contract, payment, and 
# internet service options. Obtain your data from the Codeup Data Science Database.
def get_telco_data():
    filename = "telco.csv"

    #check if cached exists
    if os.path.isfile(filename):
        #return cached data
        # print('opening data from file')
        return pd.read_csv(filename)
    else:
        # read the SQL query into a dataframe
        # print('cached file not found, creating new file')
        connection = env.get_db_url('telco_churn')
        query = '''
SELECT *
FROM customers
	JOIN contract_types
		USING (contract_type_id)
	JOIN internet_service_types
		USING (internet_service_type_id)
	JOIN payment_types
		USING (payment_type_id);
        '''
        df = pd.read_sql(query, connection)
    
        # Write that dataframe to disk for later. Called "caching" the data for later.
        df.to_csv(filename)
        # return the dataframe
        return df