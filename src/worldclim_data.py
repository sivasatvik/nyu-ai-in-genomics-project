import rasterio
from pathlib import Path
from rasterio.transform import from_bounds


if __package__ is None or __package__ == "":
    from pathlib import Path
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[1]))


WORLDCLIM_BASE = Path("data/worldclim")
WORLDCLIM_VARS = {
    "prec": "wc2.1_10m_prec",
    "srad": "wc2.1_10m_srad",
    "tavg": "wc2.1_10m_tavg",
    "vapr": "wc2.1_10m_vapr",
}


def get_value_from_tif(tif_path, lat, lon):
    """
    Extract a single pixel value from a GeoTIFF file at the given lat/lon.
    
    Parameters:
        tif_path: Path to the TIF file
        lat: Latitude (degrees)
        lon: Longitude (degrees)
    
    Returns:
        The pixel value, or None if extraction fails
    """
    try:
        with rasterio.open(tif_path) as src:
            row, col = src.index(lon, lat)
            value = src.read(1, window=((row, row + 1), (col, col + 1)))
            return value[0, 0] if value.size > 0 else None
    except Exception as e:
        print(f"Error extracting from {tif_path}: {e}")
        return None


def fetch_worldclim_data(lat, lon):
    """
    Fetch WorldClim monthly data for all four variables at the given coordinates.
    
    Returns a dict with keys like:
        - prec_01, prec_02, ..., prec_12 (precipitation in mm)
        - srad_01, srad_02, ..., srad_12 (solar radiation in kJ/m²/day)
        - tavg_01, tavg_02, ..., tavg_12 (average temperature in °C × 10)
        - vapr_01, vapr_02, ..., vapr_12 (vapour pressure in kPa × 10)
    """
    result = {}
    
    for var_short, var_dir in WORLDCLIM_VARS.items():
        var_path = WORLDCLIM_BASE / var_dir
        
        if not var_path.exists():
            print(f"Warning: {var_path} does not exist.")
            continue
        
        for month in range(1, 13):
            tif_file = var_path / f"wc2.1_10m_{var_short}_{month:02d}.tif"
            
            if not tif_file.exists():
                print(f"Warning: {tif_file} does not exist.")
                result[f"{var_short}_{month:02d}"] = None
                continue
            
            value = get_value_from_tif(tif_file, lat, lon)
            result[f"{var_short}_{month:02d}"] = value
    
    return result


if __name__ == "__main__":
    # Test the function with a sample coordinate
    lat, lon = 54.517335, 159.973068
    print(f"Fetching WorldClim data for ({lat}, {lon})...")
    data = fetch_worldclim_data(lat, lon)
    for key, value in sorted(data.items()):
        print(f"  {key}: {value}")
