import json
from pathlib import Path


class RAGDataset():
    def __init__(self, data_path: Path) -> None:
        """
        Initializes the RAGDataset object with the path to the dataset."
        """
        self.data_path = data_path
        self.data = self.load_data()
    
    def load_data(self) -> list[dict]:
        """
        Loads the data from the dataset.
        
        Returns:
            list: List of dictionaries with the data.
        """
        with open(self.data_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        return data
    
    def __len__(self) -> int:
        """
        Returns the length of the dataset.
        
        Returns:
            int: Length of the dataset.
        """
        return len(self.data)
    
    def __getitem__(self, idx: int) -> dict:
        """
        Returns the item at the specified index.
        
        Args:
            idx (int): Index of the item.
        
        Returns:
            dict: Item at the specified index.
        """
        return self.data[idx]["question"], set(self.data[idx]["files"])
    