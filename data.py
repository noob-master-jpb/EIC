import pandas as pd
import datasets
def load_dataset(path:str):
    """
    Supported Formats:
    - Parquet (Recommended)
    - CSV
    
    Args:
        path (str): Path to the dataset file
        
    Returns:
        pd.DataFrame: Loaded dataset
    """

    if path.endswith(".parquet"):
        df = pd.read_parquet(path)
    elif path.endswith(".csv"):
        df = pd.read_csv(path)
    else:
        raise ValueError("Unsupported file format")
    return df

def format_dataset(df:pd.DataFrame,user_prompt:str,agent_response:str,dataset:bool=True):
    messages = []
    for _, val in df.iterrows():
        
        conv = []
        conv.append(
            { "role": "user", "content": [{"type": "text", "text": str(val[user_prompt])}] }
        )
        conv.append(
            { "role": "model", "content": [{"type": "text", "text": str(val[agent_response])}] }
        )
        messages.append(conv)
    dataset = {'messages' : messages}
    if dataset:
        return datasets.Dataset.from_dict(dataset)
    return dataset
    
def load_format_dataset(path:str,user_prompt:str,agent_response:str,dataset:bool=True):
    return format_dataset(load_dataset(path),user_prompt,agent_response,dataset)

