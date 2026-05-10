import requests
import pandas as pd
import io, os
import re
import time


if __package__ is None or __package__ == "":
    from pathlib import Path
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[1]))

# Set to None to show all columns
pd.set_option('display.max_columns', None)

def fetch_uniprot_environmental_data(tax_id, limit=100):
    url = "https://rest.uniprot.org/uniprotkb/search"

    # We search for the specific taxonomy ID AND explicitly annotated temp/pH data
    # query = f'(taxonomy_id:{tax_id}) AND ("Optimum temperature" OR "Optimum pH")'
    # fields = "accession,id,organism_name,temp_dependence,ph_dependence,sequence"
    query = f'(taxonomy_id:{tax_id}) AND (reviewed:true)'
    fields = "accession,id,organism_name,sequence"
    
    params = {
        "query": query,
        "format": "tsv",
        "fields": fields,
        "size": limit
    }
    
    print(f"Querying UniProt API for up to {limit} sequences for Tax ID: {tax_id}...")
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
        
    # FIX: Updated to (\d+(?:\.\d+)?) to ignore trailing sentence periods safely
    match = re.search(r"(\d+(?:\.\d+)?)(?:\s*-\s*(\d+(?:\.\d+)?))?\s*degrees Celsius", temp_string)
    if match:
        val1 = float(match.group(1))
        if match.group(2):
            val2 = float(match.group(2))
            return (val1 + val2) / 2.0
        return val1
    else:
        print(f"Warning: Could not extract temperature from string: '{temp_string}'")
    return None

def extract_ph(ph_string):
    """
    Extracts the pH value from a UniProt biophysicochemical string.
    Returns the average as a float if a range is given.
    """
    if not isinstance(ph_string, str):
        return None
        
    match = re.search(r"Optimum pH is\s*(?:at least\s*)?(?:about\s*)?(?:around\s*)?(?:between\s*)?(\d+(?:\.\d+)?)(?:\s*(?:-|to|and)\s*(\d+(?:\.\d+)?))?", ph_string)
    if match:
        val1 = float(match.group(1))
        if match.group(2):
            val2 = float(match.group(2))
            return (val1 + val2) / 2.0
        return val1
    else:
        print(f"Warning: Could not extract pH from string: '{ph_string}'")
    return None

if __name__ == "__main__":
    # 1. Load the Excel file and specify the sheet name
    file_path = "goldData.xlsx"
    sheet_name = "Organism"
    fast_file_path = 'goldData_Organism.parquet'
    # Check if the fast-loading file already exists
    if os.path.exists(fast_file_path):
        print("Loading lightning-fast Parquet file...")
        gold_df = pd.read_parquet(fast_file_path)
    else:
        print(f"Loading '{sheet_name}' sheet from {file_path}...")
        try:
            gold_df = pd.read_excel(file_path, sheet_name=sheet_name)
            gold_df.to_parquet(fast_file_path)  # Save a Parquet version for next time
        except FileNotFoundError:
            print(f"Error: {file_path} not found. Please ensure it is in the same folder as this script.")
            exit()
        
    
    # 2. Extract the Taxonomy IDs
    # - Drop any blank rows (NaN)
    # - Grab only unique IDs so we don't query the same organism twice
    raw_tax_ids = gold_df['ORGANISM NCBI TAX ID'].dropna().unique()
    
    # Excel often reads numbers as floats (e.g., 515635.0). We need to convert them to clean strings (e.g., "515635")
    clean_tax_ids = [str(int(tid)) for tid in raw_tax_ids]
    
    print(f"Found {len(clean_tax_ids)} unique organisms to query.")
    
    # 3. Iterate through the Tax IDs and fetch sequences
    all_fetched_data = []

    # NOTE: We are slicing [:xx] to only test the first xx organisms. 
    # Remove the [:xx] when you are ready to run it on the entire dataset!
    for i, tax_id in enumerate(clean_tax_ids[:1000]):
        print(f"[{i+1}/1000] Fetching sequences for Tax ID: {tax_id}...")
        
        # Fetch the data from UniProt
        uniprot_df = fetch_uniprot_environmental_data(tax_id, limit=5) # getting 5 proteins per organism
        
        if not uniprot_df.empty:
            # Add a column so we know which Tax ID this came from
            uniprot_df['GOLD_NCBI_TAX_ID'] = tax_id
            all_fetched_data.append(uniprot_df)
        else:
            print(f"No sequences with explicit temp/pH data found for Tax ID {tax_id}.")
            
        # VERY IMPORTANT: Pause for 1 second between requests so UniProt doesn't block your IP address
        # time.sleep(1)
    
    # 4. Combine all the individual DataFrames into one massive dataset
    if all_fetched_data:
        final_dataset = pd.concat(all_fetched_data, ignore_index=True)
        
        print("\n" + "="*60)
        print(f"Successfully compiled {len(final_dataset)} sequences!")
        print("="*60)
        
        # Rename columns to be cleaner
        final_dataset = final_dataset.rename(columns={
            # "Entry": "Accession",
            "Organism": "Organism_Name",
            "Temperature dependence": "Raw_Temperature",
            "pH dependence": "Raw_pH"})
        
        # print("Extracting numerical signals via Regex...")
        final_dataset['Temperature_Signal'] = final_dataset['Raw_Temperature'].apply(extract_temperature)
        final_dataset['pH_Signal'] = final_dataset['Raw_pH'].apply(extract_ph)
        # Save this final dataset to parquet for lightning-fast loading in the future
        final_dataset.to_parquet('combined_uniprot_and_gold_environmental_data.parquet', index=False)
        
        # Show a preview
        display_final_dataset = final_dataset[['Entry', 'Entry Name', 'Organism_Name', 'Temperature_Signal', 'pH_Signal', 'Sequence']]
        print(display_final_dataset)
        
        # Save to a CSV for your AI model to use later
        # final_dataset.to_csv("alien_genomics_sequences.csv", index=False)
        # print("Saved data to alien_genomics_sequences.csv")
    else:
        print("\nNo sequences were successfully fetched.")