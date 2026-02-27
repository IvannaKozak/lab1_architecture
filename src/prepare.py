import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare Telco dataset for training")
    parser.add_argument("input_file", type=str, help="Path to raw CSV, e.g. data/raw/dataset.csv")
    parser.add_argument("output_dir", type=str, help="Output directory, e.g. data/prepared")
    parser.add_argument("--target-col", type=str, default="Churn", help="Target column name")
    parser.add_argument("--test-size", type=float, default=0.2, help="Test size fraction")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed")
    return parser.parse_args()


def load_and_clean_telco(data_path: str) -> pd.DataFrame:
    df = pd.read_csv(data_path)

    # 1) Прибираємо пробіли в назвах колонок
    df.columns = [c.strip() for c in df.columns]

    # 2) Обрізаємо пробіли в текстових значеннях
    obj_cols = df.select_dtypes(include="object").columns.tolist()
    for col in obj_cols:
        df[col] = df[col].astype(str).str.strip()

    # 3) Замінюємо порожні рядки на NaN
    df = df.replace(r"^\s*$", np.nan, regex=True)

    # 4) Конвертуємо TotalCharges у numeric (типова проблема Telco)
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    return df


def main():
    args = parse_args()

    input_path = Path(args.input_file)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path.resolve()}")

    df = load_and_clean_telco(str(input_path))

    if args.target_col not in df.columns:
        raise ValueError(f"Target column '{args.target_col}' not found in dataset.")

    # Stratified split по цільовій колонці (Yes/No)
    train_df, test_df = train_test_split(
        df,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=df[args.target_col],
    )

    train_path = output_dir / "train.csv"
    test_path = output_dir / "test.csv"

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    # Корисний summary для перевірки
    summary = {
        "input_file": str(input_path),
        "rows_total": int(df.shape[0]),
        "cols_total": int(df.shape[1]),
        "rows_train": int(train_df.shape[0]),
        "rows_test": int(test_df.shape[0]),
        "target_col": args.target_col,
        "test_size": args.test_size,
        "random_state": args.random_state,
    }
    with open(output_dir / "prepare_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("✅ Prepare completed")
    print(f"Saved: {train_path}")
    print(f"Saved: {test_path}")


if __name__ == "__main__":
    main()