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

    # Prepare per-condition buckets to collect up to N examples each
    N_EXAMPLES = 5
    buckets = {
        'Extreme Cold': [],
        'Extreme Heat': [],
        'Hyper-Arid': [],
        'Hyper-Humid': [],
        'High Radiation': []
    }
    
    print(f"Checking {total_checked} locations for extreme conditions...\n")
    
    for idx, row in df.iterrows():
        is_extreme, reasons, avg_temp, total_precip, mean_srad = is_extreme_row(row)
        
        if is_extreme:
            extreme_count += 1
            detail = {
                'index': idx,
                'reasons': reasons,
                'avg_temp': avg_temp,
                'total_precip': total_precip,
                'mean_srad': mean_srad
            }
            extreme_details.append(detail)

            # Add to per-condition buckets (limit to N_EXAMPLES each)
            for reason in reasons:
                if 'Extreme Cold' in reason and len(buckets['Extreme Cold']) < N_EXAMPLES:
                    buckets['Extreme Cold'].append(detail)
                if 'Extreme Heat' in reason and len(buckets['Extreme Heat']) < N_EXAMPLES:
                    buckets['Extreme Heat'].append(detail)
                if 'Hyper-Arid' in reason and len(buckets['Hyper-Arid']) < N_EXAMPLES:
                    buckets['Hyper-Arid'].append(detail)
                if 'Hyper-Humid' in reason and len(buckets['Hyper-Humid']) < N_EXAMPLES:
                    buckets['Hyper-Humid'].append(detail)
                if 'High Radiation' in reason and len(buckets['High Radiation']) < N_EXAMPLES:
                    buckets['High Radiation'].append(detail)
    
    # Print summary
    print("=" * 70)
    print("EXTREME SITE SUMMARY")
    print("=" * 70)
    print(f"Total locations checked: {total_checked}")
    print(f"Extreme locations found: {extreme_count}")
    print(f"Percentage extreme: {(extreme_count / total_checked * 100):.2f}%")
    print("=" * 70)
    
    # Print details of extreme sites: show up to N_EXAMPLES per condition
    if extreme_details:
        print("\nPer-condition examples (up to 5 each):")
        print("-" * 70)
        for cond, examples in buckets.items():
            print(f"\n{cond} (showing {len(examples)} example(s)):")
            if not examples:
                print("  - None found")
                continue
            for detail in examples:
                print(f"\n  Row {detail['index']}:")
                print(f"    Avg Temperature: {detail['avg_temp']:.1f}°C")
                print(f"    Total Annual Precip: {detail['total_precip']:.1f} mm")
                print(f"    Mean Solar Rad: {detail['mean_srad']:.0f} kJ/m2/day")
                print(f"    Reasons:")
                for reason in detail['reasons']:
                    print(f"      - {reason}")

        if len(extreme_details) > sum(len(v) for v in buckets.values()):
            remaining = len(extreme_details) - sum(len(v) for v in buckets.values())
            if remaining > 0:
                print(f"\n... plus {remaining} additional extreme sites not shown in per-condition examples.")

        # Also write a compact results file listing the per-condition examples
        out_path = 'results/extreme_conditions.txt'
        from pathlib import Path
        Path('results').mkdir(parents=True, exist_ok=True)
        with open(out_path, 'w') as fh:
            fh.write(f"Total locations checked: {total_checked}\n")
            fh.write(f"Extreme locations found: {extreme_count}\n")
            fh.write(f"Percentage extreme: {(extreme_count / total_checked * 100):.2f}%\n\n")
            for cond, examples in buckets.items():
                fh.write(f"{cond} (showing {len(examples)} example(s)):\n")
                if not examples:
                    fh.write("  - None found\n\n")
                    continue
                for detail in examples:
                    fh.write(f"  Row {detail['index']}: AvgTemp={detail['avg_temp']:.1f}C, Precip={detail['total_precip']:.1f}mm, SRad={detail['mean_srad']:.0f}kJ/m2/day, Reasons={';'.join(detail['reasons'])}\n")
                fh.write("\n")

        print(f"\nWrote per-condition extreme examples to {out_path}")
    else:
        print("\nNo extreme sites found.")




if __name__ == "__main__":
    main()
