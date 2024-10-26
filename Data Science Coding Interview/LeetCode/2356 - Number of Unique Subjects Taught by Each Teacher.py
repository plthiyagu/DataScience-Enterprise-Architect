import pandas as pd

def count_unique_subjects(teacher: pd.DataFrame) -> pd.DataFrame:
    results = teacher.groupby('teacher_id').agg({'subject_id': 'nunique'}).reset_index()
    results.columns = ['teacher_id', 'cnt']
    return results