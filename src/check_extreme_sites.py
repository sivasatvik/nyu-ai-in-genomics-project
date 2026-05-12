import pandas as pd
import numpy as np

if __package__ is None or __package__ == "":
    from pathlib import Path
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[1]))


def is_extreme_row(row):
    """
    Check if a row from the CSV has extreme environmental conditions.
    
    Returns: (is_extreme, reasons_list, avg_temp, total_precip, mean_srad)
    """
    # Extract monthly values from the row
    tavg_cols = [f'tavg_{str(i).zfill(2)}' for i in range(1, 13)]
    prec_cols = [f'prec_{str(i).zfill(2)}' for i in range(1, 13)]
    srad_cols = [f'srad_{str(i).zfill(2)}' for i in range(1, 13)]
    
    # Get values; use NaN if column doesn't exist
    tavg_values = [row.get(col, np.nan) for col in tavg_cols]
    prec_values = [row.get(col, np.nan) for col in prec_cols]
    srad_values = [row.get(col, np.nan) for col in srad_cols]
    
    # Filter out NaN values
    tavg_values = [v for v in tavg_values if not pd.isna(v)]
    prec_values = [v for v in prec_values if not pd.isna(v)]
    srad_values = [v for v in srad_values if not pd.isna(v)]
    
    # Calculate annual aggregates
    avg_temp = np.mean(tavg_values) if tavg_values else 0
    total_precip = np.sum(prec_values) if prec_values else 0
    mean_srad = np.mean(srad_values) if srad_values else 0
    
    reasons = []
    
    # 1. Temperature Check (Extreme Cold or Extreme Heat)
    if avg_temp < -10:
        reasons.append(f"Extreme Cold ({avg_temp:.1f}°C)")
    elif avg_temp > 50:
        reasons.append(f"Extreme Heat ({avg_temp:.1f}°C)")
    
    # 2. Aridity Check (Hyper-Arid or Hyper-Humid)
    if total_precip < 50:
        reasons.append(f"Hyper-Arid ({total_precip:.1f}mm/yr)")
    elif total_precip > 4000:
        reasons.append(f"Hyper-Humid ({total_precip:.1f}mm/yr)")
    
    # 3. Solar Radiation Check (High UV/Energy Stress)
    # Threshold: > 18,000 kJ/m2/day is very high for annual mean
    if mean_srad > 18000:
        reasons.append(f"High Radiation ({mean_srad:.0f} kJ/m2/day)")
    
    is_extreme_site = len(reasons) > 0
    return is_extreme_site, reasons, avg_temp, total_precip, mean_srad


def main():
    """Load CSV and check extreme conditions for each row."""
    csv_path = 'data/cleaned_environmental_data_10K.csv'
    
    print(f"Loading data from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    total_checked = len(df)
    extreme_count = 0
    extreme_details = []
    
    print(f"Checking {total_checked} locations for extreme conditions...\n")
    
    for idx, row in df.iterrows():
        is_extreme, reasons, avg_temp, total_precip, mean_srad = is_extreme_row(row)
        
        if is_extreme:
            extreme_count += 1
            extreme_details.append({
                'index': idx,
                'reasons': reasons,
                'avg_temp': avg_temp,
                'total_precip': total_precip,
                'mean_srad': mean_srad
            })
    
    # Print summary
    print("=" * 70)
    print("EXTREME SITE SUMMARY")
    print("=" * 70)
    print(f"Total locations checked: {total_checked}")
    print(f"Extreme locations found: {extreme_count}")
    print(f"Percentage extreme: {(extreme_count / total_checked * 100):.2f}%")
    print("=" * 70)
    
    # Print details of extreme sites
    if extreme_details:
        print("\nDetails of Extreme Sites (first 20):")
        print("-" * 70)
        for detail in extreme_details[:20]:
            print(f"\nRow {detail['index']}:")
            print(f"  Avg Temperature: {detail['avg_temp']:.1f}°C")
            print(f"  Total Annual Precip: {detail['total_precip']:.1f} mm")
            print(f"  Mean Solar Rad: {detail['mean_srad']:.0f} kJ/m2/day")
            print(f"  Extreme Reasons:")
            for reason in detail['reasons']:
                print(f"    - {reason}")
        
        if len(extreme_details) > 20:
            print(f"\n... and {len(extreme_details) - 20} more extreme sites")
    else:
        print("\nNo extreme sites found.")




if __name__ == "__main__":
    main()
