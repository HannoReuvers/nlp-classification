import pandas as pd
import numpy as np
import os
from pathlib import Path


class unlabelledTxtToDataFrame:
    """
    Load text files organized as:

    root/
        pos/
        neg/

    Each subfolder = label
    """

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        np.random.seed(self.random_state)
    
    def loadFromFolder(self, root: Path, dataset_type: str, label_map: Dict[str, int], threshold: Optional[int] = 0, subsample: Optional[int] = 0, frac: Optional[float]= None) -> pd.DataFrame:
        '''
        Reads all .txt files from a folder and creates a DataFrame with text and target label.

        Parameters:
            folder (Path): Path to the folder containing .txt files.
            target (int): Target label for all texts.
            thereshold (int): Threshold to decide if subsampling is needed.
            subsample (int): Number of samples to return. If 0, return all samples.
            frac (float): Fraction of samples to return. If None, use subsample.

        Returns:
            pd.DataFrame: DataFrame with columns 'text' and 'target'.
        '''
        rows = []

        for folder_name, label in label_map.items():
            folder = root / dataset_type / folder_name

            print(f"Processing folder: {folder}")

            for file in folder.glob("*.txt"):
                text = file.read_text(encoding="utf-8")

                rows.append({
                    "text": text,
                    "target": label
                })

        df = pd.DataFrame(rows).reset_index(drop=True)
        #df = df[0:400]

        if subsample != 0 and df.shape[0] > threshold:
            print("entrato primo if")
            df_1 = df[df["target"] == 1].sample(n=subsample, random_state=self.random_state).reset_index(drop=True)
            df_0 = df[df["target"] == 0].sample(n=subsample, random_state=self.random_state).reset_index(drop=True)
            df = pd.concat([df_1, df_0], axis=0).reset_index(drop=True)
        elif frac != None and df.shape[0] > threshold:
            print("entrato secondo if")
            df_1 = df[df["target"] == 1].sample(frac=frac, random_state=self.random_state).reset_index(drop=True)
            df_0 = df[df["target"] == 0].sample(frac=frac, random_state=self.random_state).reset_index(drop=True)
            df = pd.concat([df_1, df_0], axis=0).reset_index(drop=True)
        else:
            print("entrato else")
        

        return df

        