import pandas as pd

def second_highest_salary(employee: pd.DataFrame) -> pd.DataFrame:
    second_highest = employee.sort_values(by='salary', ascending=False)['salary'].unique()
    
    if len(second_highest) < 2:
        second_highest = None
    else:
        second_highest = second_highest[1]

    return pd.DataFrame({'SecondHighestSalary': [second_highest]})