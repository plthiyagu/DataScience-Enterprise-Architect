import pandas as pd

def nth_highest_salary(employee: pd.DataFrame, N: int) -> pd.DataFrame:
    nth_highest = employee.sort_values(by='salary', ascending=False)['salary'].unique()

    if len(nth_highest) < N or N <= 0:
        nth_highest = None
    else:
        nth_highest = nth_highest[N-1]

    return pd.DataFrame({f'getNthHighestSalary({N})': [nth_highest]})