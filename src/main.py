if __package__ is None or __package__ == "":
    from pathlib import Path
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[1]))

import pandas as pd
from pathlib import Path
import os
import time


try:
    from .uniprot_and_gold_data import (
        fetch_uniprot_environmental_data,
        extract_temperature,
        extract_ph,
    )
    from .nasa_power_data import fetch_nasa_power_data, format_nasa_date
    from .soilgrids_data import fetch_soilgrids_data
    from .worldclim_data import fetch_worldclim_data
except ImportError:
    from uniprot_and_gold_data import (
        fetch_uniprot_environmental_data,
        extract_temperature,
        extract_ph,
    )
    from nasa_power_data import fetch_nasa_power_data, format_nasa_date
    from soilgrids_data import fetch_soilgrids_data
    from worldclim_data import fetch_worldclim_data


DATA_FILE = Path("goldData.xlsx")
SHEET_NAME = "Organism"
CACHE_FILE = Path("goldData_Organism.parquet")
OUTPUT_FILE = Path("combined_uniprot_and_gold_environmental_data.parquet")
MAX_TAX_IDS = 1000
PROTEINS_PER_TAX_ID = 5
API_PAUSE_SECONDS = 0.5


def load_gold_data(file_path, sheet_name, cache_path):
    """Load GOLD sample metadata from cache or from the source spreadsheet."""
    if cache_path.exists():
        print("Loading lightning-fast Parquet file...")
        return pd.read_parquet(cache_path)

    print(f"Loading '{sheet_name}' sheet from {file_path}...")
    try:
        gold_df = pd.read_excel(file_path, sheet_name=sheet_name)
    except FileNotFoundError:
        raise SystemExit(
            f"Error: {file_path} not found. Please ensure it is in the same folder as this script."
        )

    gold_df.to_parquet(cache_path)
    return gold_df


def keep_gold_rows_with_coordinates(gold_df):
    """Keep and sort only GOLD rows that have both latitude and longitude."""
    filtered_gold_df = gold_df.dropna(subset=["ORGANISM LATITUDE", "ORGANISM LONGITUDE"]).copy()
    filtered_gold_df = filtered_gold_df.sort_values(
        by=["ORGANISM LATITUDE", "ORGANISM LONGITUDE"],
        kind="stable",
    )

    print(
        f"Using {len(filtered_gold_df)} GOLD rows with coordinates "
        f"(filtered from {len(gold_df)} total rows)."
    )
    return filtered_gold_df


def clean_tax_id(value):
    """Convert a GOLD taxonomy value into the clean string form UniProt expects."""
    if pd.isna(value):
        return None

    try:
        return str(int(value))
    except (TypeError, ValueError):
        return str(value).strip()


def fetch_uniprot_sequences(gold_df):
    """Fetch and combine UniProt rows for the first batch of GOLD taxonomy IDs."""
    raw_tax_ids = gold_df["ORGANISM NCBI TAX ID"].dropna().unique()
    clean_tax_ids = [clean_tax_id(tax_id) for tax_id in raw_tax_ids]
    clean_tax_ids = [tax_id for tax_id in clean_tax_ids if tax_id]

    print(f"Found {len(clean_tax_ids)} unique organisms to query.")

    all_fetched_data = []
    for index, tax_id in enumerate(clean_tax_ids[:MAX_TAX_IDS]):
        print(f"[{index + 1}/{min(len(clean_tax_ids), MAX_TAX_IDS)}] Fetching sequences for Tax ID: {tax_id}...")

        uniprot_df = fetch_uniprot_environmental_data(tax_id, limit=PROTEINS_PER_TAX_ID)

        if not uniprot_df.empty:
            uniprot_df["GOLD_NCBI_TAX_ID"] = tax_id
            all_fetched_data.append(uniprot_df)
        else:
            print(f"No sequences with explicit temp/pH data found for Tax ID {tax_id}.")

    if not all_fetched_data:
        return None

    final_dataset = pd.concat(all_fetched_data, ignore_index=True)
    print("\n" + "=" * 60)
    print(f"Successfully compiled {len(final_dataset)} sequences!")
    print("=" * 60)

    final_dataset = final_dataset.rename(
        columns={
            "Organism": "Organism_Name",
            # "Temperature dependence": "Raw_Temperature",
            # "pH dependence": "Raw_pH",
        }
    )
    # final_dataset["Temperature_Signal"] = final_dataset["Raw_Temperature"].apply(extract_temperature)
    # final_dataset["pH_Signal"] = final_dataset["Raw_pH"].apply(extract_ph)
    return final_dataset


def attach_gold_locations(final_dataset, gold_df):
    """Join GOLD location metadata onto the UniProt rows."""
    print("Merging UniProt sequences with GOLD geographic locations...")

    gold_subset = gold_df[
        [
            "ORGANISM NCBI TAX ID",
            "ORGANISM LATITUDE",
            "ORGANISM LONGITUDE",
            "ORGANISM SAMPLE COLLECTION DATE",
        ]
    ].copy()

    gold_subset = gold_subset.drop_duplicates(subset=["ORGANISM NCBI TAX ID"])
    gold_subset["ORGANISM NCBI TAX ID"] = gold_subset["ORGANISM NCBI TAX ID"].apply(clean_tax_id)

    return pd.merge(
        final_dataset,
        gold_subset,
        left_on="GOLD_NCBI_TAX_ID",
        right_on="ORGANISM NCBI TAX ID",
        how="left",
    )


def fetch_environmental_api_data(final_dataset):
    """Fetch NASA POWER, SoilGrids, and WorldClim values for each unique location/date pair."""
    unique_locations = (
        final_dataset[
            ["ORGANISM LATITUDE", "ORGANISM LONGITUDE", "ORGANISM SAMPLE COLLECTION DATE"]
        ]
        .drop_duplicates()
        .dropna(subset=["ORGANISM LATITUDE", "ORGANISM LONGITUDE"])
    )

    print(f"Found {len(unique_locations)} unique geographic locations. Fetching NASA, SoilGrids & WorldClim data...")

    env_api_results = []
    for _, row in unique_locations.iterrows():
        lat = row["ORGANISM LATITUDE"]
        lon = row["ORGANISM LONGITUDE"]
        raw_date = row["ORGANISM SAMPLE COLLECTION DATE"]

        print(f"  -> Fetching external API data for Location ({lat}, {lon})")

        nasa_date = format_nasa_date(raw_date)
        nasa_data = fetch_nasa_power_data(lat, lon, nasa_date)
        time.sleep(API_PAUSE_SECONDS)

        soil_data = fetch_soilgrids_data(lat, lon)
        time.sleep(API_PAUSE_SECONDS)

        worldclim_data = fetch_worldclim_data(lat, lon)
        time.sleep(API_PAUSE_SECONDS)

        # Build result row with NASA, SoilGrids, and WorldClim data
        result_row = {
            "ORGANISM LATITUDE": lat,
            "ORGANISM LONGITUDE": lon,
            "ORGANISM SAMPLE COLLECTION DATE": raw_date,
            "NASA_Temp_C": nasa_data.get("Temperature_C") if nasa_data else None,
            "NASA_Radiation": nasa_data.get("Radiation_W_m2") if nasa_data else None,
            # "SoilGrids_pH": soil_data.get("phh2o") if soil_data else None,
            # "SoilGrids_Carbon": soil_data.get("soc") if soil_data else None,
        }
        
        # Add WorldClim monthly data
        if worldclim_data:
            result_row.update(worldclim_data)
        
        env_api_results.append(result_row)

    env_api_df = pd.DataFrame(env_api_results)
    if env_api_df.empty:
        return final_dataset

    final_dataset["ORGANISM SAMPLE COLLECTION DATE"] = final_dataset["ORGANISM SAMPLE COLLECTION DATE"].astype(str)
    env_api_df["ORGANISM SAMPLE COLLECTION DATE"] = env_api_df["ORGANISM SAMPLE COLLECTION DATE"].astype(str)

    return pd.merge(
        final_dataset,
        env_api_df,
        on=["ORGANISM LATITUDE", "ORGANISM LONGITUDE", "ORGANISM SAMPLE COLLECTION DATE"],
        how="left",
    )


def keep_only_complete_environment_rows(final_dataset):
    """Drop rows that are missing location metadata or either environmental source."""
    required_columns = [
        "ORGANISM LATITUDE",
        "ORGANISM LONGITUDE",
    ]
    return final_dataset.dropna(subset=required_columns)


def main():
    gold_df = load_gold_data(DATA_FILE, SHEET_NAME, CACHE_FILE)
    gold_df = keep_gold_rows_with_coordinates(gold_df)
    print("GOLD Columns Available:", gold_df.columns.tolist())

    final_dataset = fetch_uniprot_sequences(gold_df)
    if final_dataset is None:
        print("\nNo sequences were successfully fetched.")
        return

    final_dataset = attach_gold_locations(final_dataset, gold_df)
    final_dataset = fetch_environmental_api_data(final_dataset)
    final_dataset = keep_only_complete_environment_rows(final_dataset)

    final_dataset.to_parquet(OUTPUT_FILE, index=False)

    # preview_columns = ["Entry", "Entry Name", "Organism_Name", "Temperature_Signal", "pH_Signal", "Sequence"]
    print(final_dataset)

if __name__ == "__main__":
    main()