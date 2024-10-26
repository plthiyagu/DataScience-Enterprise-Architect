import pandas as pd

def count_salary_categories(accounts: pd.DataFrame) -> pd.DataFrame:
    low_salary_count = accounts[accounts['income'] < 20000].shape[0]
    average_salary_count = accounts[(accounts['income'] >= 20000) & (accounts['income'] <= 50000)].shape[0]
    high_salary_count = accounts[accounts['income'] > 50000].shape[0]

    results = pd.DataFrame({
        'category': ['High Salary', 'Low Salary', 'Average Salary'],
        'accounts_count': [high_salary_count, low_salary_count, average_salary_count]
    })
    return results