import pandas as pd
import os

input_file = "/home/jzyh/xjp_projects/SIC/refer/DMN.csv"
output_file = "/home/jzyh/xjp_projects/SIC/refer/DMN_normalized.csv"

try:
    # Read the CSV file
    df = pd.read_csv(input_file)
    
    # Perform normalization: Pandas automatically detects min/max for each column
    # Cells that were originally NaN will remain NaN after the operation
    normalized_df = (df - df.min()) / (df.max() - df.min())
    
    # Save the normalized DataFrame without filling NaN
    # float_format='%.2f' formats numbers to 2 decimal places
    # NaN will be saved as empty cells in the CSV
    normalized_df.to_csv(output_file, index=False, float_format='%.2f', na_rep='')
    
    print(f"✅ Normalization completed and saved to: {output_file}")
    print("Original empty values are preserved, numeric values kept to two decimal places.")

except Exception as e:
    print(f"❌ Error: {e}")
