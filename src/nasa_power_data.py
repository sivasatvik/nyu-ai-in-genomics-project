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
    
    try:
        response = requests.get(url, params=params, timeout=10)
        if response.status_code == 200:
            data = response.json()

            # Check if response contains properties and parameters
            if 'properties' not in data or 'parameter' not in data['properties']:
                print(f"NASA API error for ({lat}, {lon}, {date_str}): Missing properties in response")
                return None

            metrics = data['properties']['parameter']
            result = {}

            # Safely extract each parameter
            if 'T2M' in metrics and date_str in metrics['T2M']:
                result["Temperature_C"] = metrics['T2M'][date_str]
            else:
                print(f"Warning: T2M data missing for ({lat}, {lon}, {date_str})")

            if 'RH2M' in metrics and date_str in metrics['RH2M']:
                result["Humidity_Percent"] = metrics['RH2M'][date_str]
            else:
                print(f"Warning: RH2M data missing for ({lat}, {lon}, {date_str})")

            if 'ALLSKY_SFC_SW_DWN' in metrics and date_str in metrics['ALLSKY_SFC_SW_DWN']:
                result["Radiation_W_m2"] = metrics['ALLSKY_SFC_SW_DWN'][date_str]
            else:
                print(f"Warning: ALLSKY_SFC_SW_DWN data missing for ({lat}, {lon}, {date_str})")

            return result if result else None
        else:
            print(f"Error fetching NASA data: Status {response.status_code} for ({lat}, {lon}, {date_str})")
            return None
    except Exception as e:
        print(f"Exception fetching NASA data for ({lat}, {lon}, {date_str}): {e}")
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