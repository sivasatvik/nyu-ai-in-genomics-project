import requests
import pandas as pd
import io
import re


if __package__ is None or __package__ == "":
    from pathlib import Path
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[1]))

def fetch_uniprot_environmental_data(limit=100):
    """
    Fetches protein sequences from UniProt that have explicitly measured 
    temperature or pH dependencies.
    """
    url = "https://rest.uniprot.org/uniprotkb/search"
    query = '(reviewed:true) AND ("Optimum temperature" OR "Optimum pH")'
    fields = "accession,id,organism_name,temp_dependence,ph_dependence,sequence"
    
    params = {
        "query": query,
        "format": "tsv",
        "fields": fields,
        "size": limit
    }
    
    print(f"Querying UniProt API for {limit} sequences...")
    response = requests.get(url, params=params)
    response.raise_for_status() 
    
    # Load directly into pandas
    df = pd.read_csv(io.StringIO(response.text), sep='\t')
    return df

def extract_temperature(temp_string):
    """
    Extracts the temperature value from a UniProt biophysicochemical string.
    Returns the average as a float if a range is given.
    """
    if not isinstance(temp_string, str):
        return None
        
    match = re.search(r"([\d\.]+)(?:\s*-\s*([\d\.]+))?\s*degrees Celsius", temp_string)
    if match:
        val1 = float(match.group(1))
        if match.group(2):
            val2 = float(match.group(2))
            return (val1 + val2) / 2.0
        return val1
    return None

def extract_ph(ph_string):
    """
    Extracts the pH value from a UniProt biophysicochemical string.
    Returns the average as a float if a range is given.
    """
    if not isinstance(ph_string, str):
        return None
        
    # FIX: Changed [\d\.]+ to \d+(?:\.\d+)? to ignore trailing sentence periods
    match = re.search(r"Optimum pH is\s*(?:at least\s*)?(\d+(?:\.\d+)?)(?:\s*(?:-|to)\s*(\d+(?:\.\d+)?))?", ph_string)
    if match:
        val1 = float(match.group(1))
        if match.group(2):
            val2 = float(match.group(2))
            return (val1 + val2) / 2.0
        return val1
    return None

if __name__ == "__main__":
    # 1. Fetch the raw data
    df = fetch_uniprot_environmental_data(limit=50)
    
    # 2. Rename the columns coming from the UniProt TSV for clarity
    df = df.rename(columns={
        "Entry": "UniProt_ID", 
        "Organism": "Organism_Name",
        "Temperature dependence": "Raw_Temperature",
        "pH dependence": "Raw_pH"
    })

    # 3. Apply the regex extraction functions to create our float signals
    print("Extracting numerical signals via Regex...")
    df['Temperature_Signal'] = df['Raw_Temperature'].apply(extract_temperature)
    df['pH_Signal'] = df['Raw_pH'].apply(extract_ph)
    
    print(f"\nSuccessfully processed {len(df)} sequences.")
    print("-" * 60)
    
    # 4. Display the clean, machine-learning-ready data
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    
    display_df = df[['UniProt_ID', 'Organism_Name', 'Temperature_Signal', 'pH_Signal', 'Sequence']]
    print(display_df.head(10))
    
    # (Optional) Drop rows that don't have at least one signal, then save
    # clean_df = display_df.dropna(subset=['Temperature_Signal', 'pH_Signal'], how='all')
    # clean_df.to_csv("alien_model_training_data.csv", index=False)