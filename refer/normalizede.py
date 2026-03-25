import pandas as pd
import os

input_file = "/home/jzyh/xjp_projects/SIC/refer/DMN.csv"
output_file = "/home/jzyh/xjp_projects/SIC/refer/DMN_normalized.csv"

try:
    df = pd.read_csv(input_file)
    
    # 执行归一化：Pandas 会自动识别每一列的 min/max
    # 对于原本是 NaN 的单元格，运算后依然会保持为 NaN
    normalized_df = (df - df.min()) / (df.max() - df.min())
    
    # 不使用 fillna(0)，直接保存
    # float_format='%.2f' 会处理数字，而 NaN 在 to_csv 中默认会保存为空字符串（也就是空的单元格）
    normalized_df.to_csv(output_file, index=False, float_format='%.2f', na_rep='')
    
    print(f"✅ 归一化完成，已保存至: {output_file}")
    print("原始空值已保留，数值已保留两位小数。")

except Exception as e:
    print(f"❌ 错误: {e}")