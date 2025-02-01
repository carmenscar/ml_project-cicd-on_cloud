import pandas as pd
import logging

logging.basicConfig(filename='logs.log',
                    level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

def import_data(dataframe):
    try:
        df = pd.read_csv(dataframe)
        logging.info("SUCCESS: dataframe is loaded")
        return df
    except:
        logging.error("ERROR: dataframe not found")
        return None

def clean_data(dataframe):
    df = dataframe.copy()
    df.columns = df.columns.str.replace(' ', '')
    logging.info("SUCCESS: dataframe is cleaned")
    return df


if __name__ == "__main__":
    df_raw = import_data("/home/carmenscar/nd0821-c3-starter-code/starter/data/census.csv")
    df_cleaned = clean_data(df_raw)
    print(df_cleaned.head())