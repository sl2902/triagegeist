
import pandas as pd

def read_datasets(filepath: str) -> pd.DataFrame:
    """Read source dataset"""

    try:
        return pd.read_csv(filepath)
    except Exception as err:
        print(f"Failed to read ddataset {filepath}: {err}")
        raise