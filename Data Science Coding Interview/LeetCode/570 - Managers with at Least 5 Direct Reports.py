import pandas as pd

def find_managers(employee: pd.DataFrame) -> pd.DataFrame:
    results = employee.groupby('managerId').agg({'id': 'count'}).reset_index()
    results = results[results['id'] >= 5]
    results = employee[employee['id'].isin(results['managerId'].tolist())][['name']]
    return results