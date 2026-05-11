import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def main(input_parquet, output_csv, cleaned_output_csv):
    # 1. Load parquet and save as csv the dataset
    df = pd.read_parquet(input_parquet)

    output_csv_path = Path(output_csv)
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv_path, index=False)

    print(f"Original rows: {len(df)}")

    # 2. Select only the numerical columns for checking
    numeric_cols = df.select_dtypes(include=[np.number]).columns

    # 3. Create a mask for extreme Float/Int32 boundaries
    # (Captures the -3.4e38 float nodata markers and any rogue int32 max/mins)
    mask_extreme_bounds = (df[numeric_cols] < -1e30) | (df[numeric_cols] > 1e30)

    # 4. Create a mask specifically for Int16 and UInt16 overflow 'NoData' flags
    # (Note: We exclude legitimate values like 15 or 127 which are valid temperatures/precipitation)
    overflow_flags = [
        65535,
        65535.0,  # uint16 max
        32767,
        32767.0,  # int16 max
        -32768,
        -32768.0,  # int16 min
        -9999,
        -9999.0,  # standard custom nodata flag
    ]
    mask_specific_flags = df[numeric_cols].isin(overflow_flags)

    # 5. Combine the masks: True if ANY column in a row has an overflow flag
    combined_mask = mask_extreme_bounds | mask_specific_flags

    # 6. Drop the rows containing these flags
    rows_to_drop = combined_mask.any(axis=1)
    cleaned_df = df[~rows_to_drop]

    print(f"Cleaned rows: {len(cleaned_df)}")
    print(f"Total corrupted rows removed: {len(df) - len(cleaned_df)}")

    # 7. Save the fully sanitized dataset
    cleaned_output_path = Path(cleaned_output_csv)
    cleaned_output_path.parent.mkdir(parents=True, exist_ok=True)
    cleaned_df.to_csv(cleaned_output_path, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clean environmental dataset from parquet input")
    parser.add_argument(
        "-i",
        "--input-parquet",
        default="data/combined_uniprot_and_gold_environmental_data_10K.parquet",
        help="Input parquet path",
    )
    parser.add_argument(
        "-o",
        "--output-csv",
        default="data/combined_uniprot_and_gold_environmental_data_10K.csv",
        help="Output CSV path for raw converted data",
    )
    parser.add_argument(
        "-c",
        "--cleaned-output-csv",
        default="data/cleaned_environmental_data_10K.csv",
        help="Output CSV path for cleaned data",
    )
    args = parser.parse_args()
    main(args.input_parquet, args.output_csv, args.cleaned_output_csv)