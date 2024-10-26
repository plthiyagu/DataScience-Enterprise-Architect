import pandas as pd

def department_highest_salary(employee: pd.DataFrame, department: pd.DataFrame) -> pd.DataFrame:
    highest_salary = employee.groupby('departmentId').agg({'salary': 'max'}).reset_index()
    highest_salary.columns = ['id', 'salary']

    highest_salary = department.merge(highest_salary, on='id', how='inner')
    highest_salary.columns = ['departmentId', 'departmentName', 'salary']

    highest_salary = employee.merge(highest_salary, on=['departmentId', 'salary'], how='inner')
    highest_salary = highest_salary.drop(columns=['id', 'departmentId'])
    highest_salary = highest_salary.rename(columns={'departmentName': 'Department', 'name': 'Employee', 'salary': 'Salary'})
    return highest_salary