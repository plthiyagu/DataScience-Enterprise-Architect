import pandas as pd

def find_classes(courses: pd.DataFrame) -> pd.DataFrame:
    results = courses.groupby('class').agg({'student': 'nunique'}).reset_index()
    results = results[results['student'] >= 5]
    return results.drop(columns=['student'])