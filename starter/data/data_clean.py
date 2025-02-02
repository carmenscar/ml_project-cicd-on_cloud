import pandas as pd
import logging

logging.basicConfig(filename='logs.log',
                    level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

def import_data_and_clean(csv_path):
    try:
        df = pd.read_csv(csv_path)
        logging.info("SUCCESS: dataframe is loaded")
    except:
        logging.error("ERROR: dataframe not found")
        return None
    df.columns = df.columns.str.replace(' ', '')
    logging.info("SUCCESS: dataframe is cleaned")
    return df


if __name__ == "__main__":
    df_raw = import_data_and_clean("/home/carmenscar/nd0821-c3-starter-code/starter/data/census.csv")
    print(df_raw.head())