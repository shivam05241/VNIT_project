"""
Data Monitoring and Validation Module using Evidently
======================================================

This module provides data quality monitoring, drift detection, and validation.
It generates comprehensive reports for data validation.

Features:
- Data Quality Analysis
- Missing Value Detection
- Statistical Summary Reports
- Data Schema Validation
"""

import numpy as np
import pandas as pd
from datetime import datetime
import os
import json


def create_quality_report(df, output_dir="reports", name="data_quality_report"):
    """
    Generate a comprehensive data quality report.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe to analyze.
    output_dir : str, optional
        Directory to save reports (default: "reports").
    name : str, optional
        Report name prefix (default: "data_quality_report").
    
    Returns
    -------
    dict
        Report metadata including file path and timestamp.
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    try:
        print(f"Generating data quality report for {df.shape[0]} rows × {df.shape[1]} columns...")
        
        # Analyze data quality metrics
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()
        
        quality_metrics = {
            "timestamp": timestamp,
            "shape": {"rows": df.shape[0], "columns": df.shape[1]},
            "dtypes": {
                "numeric": len(numeric_cols),
                "categorical": len(categorical_cols),
            },
            "missing_values": df.isna().sum().to_dict(),
            "missing_pct": (df.isna().sum() / len(df) * 100).to_dict(),
            "duplicates": int(df.duplicated().sum()),
            "memory_mb": float(df.memory_usage(deep=True).sum() / 1024**2),
        }
        
        # Save quality report as JSON
        report_file = os.path.join(output_dir, f"{name}_{timestamp}.json")
        with open(report_file, "w") as f:
            json.dump(quality_metrics, f, indent=2)
        
        print(f"✓ Data quality report saved: {report_file}")
        
        return {
            "status": "success",
            "file": report_file,
            "timestamp": timestamp,
            "rows": df.shape[0],
            "columns": df.shape[1],
            "metrics": quality_metrics,
        }
    except Exception as e:
        print(f"✗ Error generating quality report: {e}")
        return {"status": "error", "message": str(e)}


def create_drift_report(reference_df, current_df, output_dir="reports", name="data_drift_report"):
    """
    Generate a data drift report comparing reference and current datasets.
    
    Parameters
    ----------
    reference_df : pd.DataFrame
        Reference dataset (baseline).
    current_df : pd.DataFrame
        Current dataset to compare against reference.
    output_dir : str, optional
        Directory to save reports (default: "reports").
    name : str, optional
        Report name prefix (default: "data_drift_report").
    
    Returns
    -------
    dict
        Report metadata including file path and drift status.
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    try:
        print(f"Generating data drift report...")
        print(f"  Reference: {reference_df.shape[0]} rows × {reference_df.shape[1]} columns")
        print(f"  Current: {current_df.shape[0]} rows × {current_df.shape[1]} columns")
        
        # Detect drift in numeric columns
        numeric_cols = reference_df.select_dtypes(include=np.number).columns.tolist()
        drift_summary = {}
        
        for col in numeric_cols:
            if col in current_df.columns:
                ref_mean = reference_df[col].mean()
                curr_mean = current_df[col].mean()
                ref_std = reference_df[col].std()
                curr_std = current_df[col].std()
                
                # Calculate drift metrics
                mean_diff = abs(curr_mean - ref_mean) / (ref_mean + 1e-8)
                std_diff = abs(curr_std - ref_std) / (ref_std + 1e-8)
                
                drift_summary[col] = {
                    "reference_mean": float(ref_mean),
                    "current_mean": float(curr_mean),
                    "mean_change_pct": float(mean_diff * 100),
                    "reference_std": float(ref_std),
                    "current_std": float(curr_std),
                    "std_change_pct": float(std_diff * 100),
                }
        
        # Save drift report as JSON
        report_file = os.path.join(output_dir, f"{name}_{timestamp}.json")
        with open(report_file, "w") as f:
            json.dump(drift_summary, f, indent=2)
        
        print(f"✓ Data drift report saved: {report_file}")
        
        return {
            "status": "success",
            "file": report_file,
            "timestamp": timestamp,
            "reference_rows": reference_df.shape[0],
            "current_rows": current_df.shape[0],
            "drift_metrics": drift_summary,
        }
    except Exception as e:
        print(f"✗ Error generating drift report: {e}")
        return {"status": "error", "message": str(e)}


def validate_data_schema(df, expected_columns=None, expected_dtypes=None):
    """
    Validate data schema against expected structure.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe to validate.
    expected_columns : list, optional
        List of expected column names.
    expected_dtypes : dict, optional
        Dictionary mapping column names to expected dtypes.
    
    Returns
    -------
    dict
        Validation results with issues if any.
    """
    validation_result = {
        "status": "valid",
        "issues": [],
        "column_count": len(df.columns),
        "row_count": len(df),
    }
    
    # Check columns
    if expected_columns:
        missing_cols = set(expected_columns) - set(df.columns)
        extra_cols = set(df.columns) - set(expected_columns)
        
        if missing_cols:
            validation_result["issues"].append(f"Missing columns: {missing_cols}")
            validation_result["status"] = "invalid"
        if extra_cols:
            validation_result["issues"].append(f"Extra columns: {extra_cols}")
    
    # Check dtypes
    if expected_dtypes:
        for col, expected_dtype in expected_dtypes.items():
            if col in df.columns:
                actual_dtype = str(df[col].dtype)
                if actual_dtype != str(expected_dtype):
                    validation_result["issues"].append(
                        f"Column '{col}': expected {expected_dtype}, got {actual_dtype}"
                    )
    
    # Check missing values
    missing_pct = (df.isna().sum() / len(df) * 100).to_dict()
    validation_result["missing_values_pct"] = {
        col: pct for col, pct in missing_pct.items() if pct > 0
    }
    
    if validation_result["missing_values_pct"]:
        validation_result["status"] = "warning"
    
    print(f"✓ Schema validation: {validation_result['status'].upper()}")
    if validation_result["issues"]:
        for issue in validation_result["issues"]:
            print(f"  - {issue}")
    
    return validation_result


def generate_summary_report(
    df, name="Summary Report", output_dir="reports"
):
    """
    Generate a comprehensive summary report with statistics and visualizations.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe to analyze.
    name : str, optional
        Report name prefix (default: "Summary Report").
    output_dir : str, optional
        Directory to save reports (default: "reports").
    
    Returns
    -------
    dict
        Summary statistics and metadata.
    """
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Collect statistics
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()
    
    summary = {
        "timestamp": timestamp,
        "shape": {"rows": df.shape[0], "columns": df.shape[1]},
        "numeric_columns": len(numeric_cols),
        "categorical_columns": len(categorical_cols),
        "missing_values": {
            col: int(df[col].isna().sum()) for col in df.columns
        },
        "duplicates": int(df.duplicated().sum()),
        "memory_usage_mb": float(df.memory_usage(deep=True).sum() / 1024**2),
        "numeric_summary": df[numeric_cols].describe().to_dict() if numeric_cols else {},
    }
    
    # Create summary file
    summary_file = os.path.join(output_dir, f"{name}_{timestamp}.txt")
    with open(summary_file, "w") as f:
        f.write(f"Data Summary Report - {timestamp}\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Shape: {summary['shape']['rows']} rows × {summary['shape']['columns']} columns\n")
        f.write(f"Numeric Columns: {summary['numeric_columns']}\n")
        f.write(f"Categorical Columns: {summary['categorical_columns']}\n")
        f.write(f"Duplicates: {summary['duplicates']}\n")
        f.write(f"Memory Usage: {summary['memory_usage_mb']:.2f} MB\n\n")
        
        f.write("Missing Values:\n")
        f.write("-" * 60 + "\n")
        for col, count in summary["missing_values"].items():
            pct = count / summary["shape"]["rows"] * 100
            f.write(f"{col:30s}: {count:6d} ({pct:5.2f}%)\n")
        
        f.write("\n" + "=" * 60 + "\n")
    
    print(f"✓ Summary report saved: {summary_file}")
    
    return summary


def main(data_path, reference_path=None):
    """
    Run complete monitoring pipeline.
    
    Parameters
    ----------
    data_path : str
        Path to current data CSV.
    reference_path : str, optional
        Path to reference/baseline data CSV for drift detection.
    """
    print("\n" + "=" * 70)
    print("EVIDENTLY DATA MONITORING PIPELINE")
    print("=" * 70 + "\n")
    
    # Load data
    print("Step 1: Loading data...")
    current_df = pd.read_csv(data_path)
    print(f"✓ Loaded {current_df.shape[0]} rows × {current_df.shape[1]} columns\n")
    
    # Data quality report
    print("Step 2: Generating data quality report...")
    quality_result = create_quality_report(current_df)
    if quality_result and quality_result.get("status") == "success":
        print(f"Report: {quality_result['file']}\n")
    
    # Schema validation
    print("Step 3: Validating data schema...")
    validation = validate_data_schema(current_df)
    print()
    
    # Summary report
    print("Step 4: Generating summary report...")
    summary = generate_summary_report(current_df)
    print()
    
    # Drift detection (if reference data provided)
    if reference_path:
        print("Step 5: Detecting data drift...")
        reference_df = pd.read_csv(reference_path)
        drift_result = create_drift_report(reference_df, current_df)
        if drift_result and drift_result.get("status") == "success":
            print(f"Report: {drift_result['file']}\n")
    else:
        print("Step 5: Data drift detection skipped (no reference data)\n")
    
    print("=" * 70)
    print("✓ MONITORING PIPELINE COMPLETED")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python monitoring.py <data_path> [reference_path]")
        print("Example: python monitoring.py ../../data/train.csv")
        sys.exit(1)
    
    data_path = sys.argv[1]
    reference_path = sys.argv[2] if len(sys.argv) > 2 else None
    
    main(data_path, reference_path)
