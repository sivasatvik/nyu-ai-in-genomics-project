import requests
import pandas as pd


if __package__ is None or __package__ == "":
    from pathlib import Path
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[1]))

def fetch_soilgrids_data(lat, lon):
    """
    Fetches soil pH and Soil Organic Carbon (SOC) from SoilGrids v2.0 REST API.
    """
    url = "https://rest.isric.org/soilgrids/v2.0/properties/query"
    params = {
        "lon": lon,
        "lat": lat,
        "property": ["phh2o", "soc"], 
        "depth": "0-5cm",             
        "value": "mean"               
    }
    try:
        response = requests.get(url, params=params, timeout=10)
        if response.status_code == 200:
            data = response.json()
            results = {}
            for prop in data['properties']['layers']:
                name = prop['name']
                val = prop['depths'][0]['values']['mean']
                if name == 'phh2o' and val is not None:
                    val = val / 10.0 # SoilGrids stores pH multiplied by 10
                results[name] = val
            return results
    except Exception as e:
        print(f"SoilGrids API error for ({lat}, {lon}): {e}")
    return None