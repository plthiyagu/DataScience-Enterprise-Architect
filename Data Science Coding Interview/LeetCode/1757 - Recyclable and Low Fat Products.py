import pandas as pd

def find_products(products: pd.DataFrame) -> pd.DataFrame:
    return products.query('low_fats == "Y" & recyclable == "Y"')[['product_id']]