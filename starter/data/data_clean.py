import pandas as pd
import logging

logging.basicConfig(filename='logs.log',
                    level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

def import_data(csv_path):
    try:
        df = pd.read_csv(csv_path)
        logging.info("SUCCESS: dataframe is loaded")
    except:
        logging.error("ERROR: dataframe not found")
        return None
    logging.info("SUCCESS: dataframe is cleaned")
    return df
