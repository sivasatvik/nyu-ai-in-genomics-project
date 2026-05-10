import requests
import pandas as pd


if __package__ is None or __package__ == "":
    from pathlib import Path
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[1]))


def fetch_nasa_power_data(lat, lon, date_str):
    """
    Fetches weather/radiation data from NASA POWER for a specific date.
    date_str must be in YYYYMMDD format (e.g., '20220515')
    """
    # T2M = Temperature at 2 meters, RH2M = Relative Humidity, 
    # ALLSKY_SFC_SW_DWN = Solar Radiation
    parameters = "T2M,RH2M,ALLSKY_SFC_SW_DWN"
    
    url = f"https://power.larc.nasa.gov/api/temporal/daily/point"
    params = {
        "parameters": parameters,
        "community": "RE", # Renewable Energy community has good surface metrics
        "longitude": lon,
        "latitude": lat,
        "start": date_str,
        "end": date_str,
        "format": "JSON"
    }
    
    response = requests.get(url, params=params)
    if response.status_code == 200:
        data = response.json()
        
        # Extract the metrics (they are nested under properties -> parameter)
        metrics = data['properties']['parameter']
        
        return {
            "Temperature_C": metrics['T2M'][date_str],
            "Humidity_Percent": metrics['RH2M'][date_str],
            "Radiation_W_m2": metrics['ALLSKY_SFC_SW_DWN'][date_str]
        }
    else:
        print(f"Error fetching NASA data: {response.status_code}")
        return None

def format_nasa_date(date_val):
    """Formats GOLD dates to YYYYMMDD required by NASA POWER."""
    if pd.isna(date_val):
        return "20150101" # Default fallback if no date is provided
    try:
        dt = pd.to_datetime(str(date_val), errors='coerce')
        if pd.isna(dt):
            return "20150101"
        return dt.strftime("%Y%m%d")
    except:
        return "20150101"

if __name__ == "__main__":
    # Test it
    lat, lon = 54.517335, 159.973068  # From your Dictyoglomus turgidum sample
    print(fetch_nasa_power_data(lat, lon, "20200101"))