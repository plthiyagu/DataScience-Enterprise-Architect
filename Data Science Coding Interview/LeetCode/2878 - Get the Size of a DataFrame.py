import pandas as pd

def getDataframeSize(players: pd.DataFrame) -> List[int]:
    rows, cols = players.shape
    return [rows, cols]