import pandas as pd
import requests
import io

def fetch_exoplanet_data():
    """
    Fetches confirmed exoplanets and their environmental parameters 
    from the NASA Exoplanet Archive.
    """
    url = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"
    
    # We write an ADQL (Astronomical Data Query Language) query to get specific columns
    # pl_name = Planet Name
    # pl_eqt = Equilibrium Temperature (K)
    # pl_insol = Insolation Flux (Earth Flux)
    # pl_rade = Planet Radius (Earth Radii)
    # st_spectype = Star Spectral Type (e.g., G2V for our Sun)
    
    query = """
        SELECT pl_name, pl_eqt, pl_insol, pl_rade, st_spectype
        FROM ps
        WHERE default_flag = 1 
        AND pl_eqt IS NOT NULL
    """
    
    params = {
        "request": "doQuery",
        "lang": "ADQL",
        "format": "csv",
        "query": query
    }
    
    print("Querying NASA Exoplanet Archive...")
    response = requests.get(url, params=params)
    response.raise_for_status()
    
    # Load into Pandas
    df = pd.read_csv(io.StringIO(response.text))
    
    # Convert Kelvin to Celsius so it matches your UniProt/GOLD data
    df['pl_temp_C'] = df['pl_eqt'] - 273.15
    
    return df

if __name__ == "__main__":
    exo_df = fetch_exoplanet_data()
    print(f"Fetched {len(exo_df)} exoplanets with temperature data.")
    print(exo_df.head())