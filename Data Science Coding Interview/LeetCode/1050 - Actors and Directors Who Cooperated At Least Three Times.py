import pandas as pd

def actors_and_directors(actor_director: pd.DataFrame) -> pd.DataFrame:
    results = actor_director.groupby(['actor_id', 'director_id']).agg({'timestamp': 'count'}).reset_index()
    results = results[results['timestamp'] >= 3].drop(columns=['timestamp'])
    return results