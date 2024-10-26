import pandas as pd

def total_time(employees: pd.DataFrame) -> pd.DataFrame:
    employees['total_time'] = employees['out_time'] - employees['in_time']
    results = employees.groupby(['emp_id', 'event_day']).agg({'total_time': 'sum'}).reset_index()
    results.columns = ['emp_id', 'day', 'total_time']
    return results