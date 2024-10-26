import pandas as pd

def categorize_products(activities: pd.DataFrame) -> pd.DataFrame:
    results = activities.groupby('sell_date').agg(num_sold=('product', 'nunique'), products=('product', lambda x: ','.join(sorted(x.unique())))).reset_index().sort_values(by='sell_date')
    return results