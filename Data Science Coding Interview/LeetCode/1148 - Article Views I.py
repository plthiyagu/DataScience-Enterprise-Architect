import pandas as pd

def article_views(views: pd.DataFrame) -> pd.DataFrame:
    unique_author_ids = set(views[views['author_id'] == views['viewer_id']]['author_id'].tolist())
    return pd.DataFrame({'id': list(unique_author_ids)}).sort_values(by='id')