import pandas as pd

def find_customers(customers: pd.DataFrame, orders: pd.DataFrame) -> pd.DataFrame:
    customers_who_bought_something = set(orders['customerId'].tolist())
    all_customers = set(customers['id'].tolist())
    customers_who_never_order = list(all_customers - customers_who_bought_something)

    customers_name = customers[customers['id'].isin(customers_who_never_order)].drop(columns='id')
    customers_name.columns = ['Customers']
    return customers_name