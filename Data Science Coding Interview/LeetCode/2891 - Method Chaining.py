import pandas as pd

def findHeavyAnimals(animals: pd.DataFrame) -> pd.DataFrame:
    return animals.query('weight > 100').sort_values('weight', ascending=False).drop(columns=['species', 'age', 'weight'])