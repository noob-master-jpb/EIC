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

def format_dataset(df:pd.DataFrame,user_prompt:str,agent_response:str,as_hf_dataset:bool=True):
    """
    Args:
        df (pd.DataFrame): DataFrame containing the dataset
        user_prompt (str): Column name for the user prompt
        agent_response (str): Column name for the agent response
        as_hf_dataset (bool): Whether to return a Hugging Face Dataset    
    """

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
    if as_hf_dataset:
        return datasets.Dataset.from_dict(dataset)
    return dataset

def process_dataset(path:str,user_prompt:str,agent_response:str,processor=None,num_proc=4):
    """
    Args:
        path (str): Path to the dataset file
        user_prompt (str): Column name for the user prompt
        agent_response (str): Column name for the agent response
        processor (AutoProcessor): Processor to use for formatting

    Returns:
        datasets.Dataset: Processed dataset

    Note:
        It will always return hf dataset
    """
    if processor is None:
        raise ValueError("Processor is required")
    df = load_dataset(path)
    dataset=format_dataset(df,user_prompt,agent_response,as_hf_dataset=True)
    final_ds = dataset.map(
        lambda x: {"text": processor.apply_chat_template(x["messages"], tokenize=False)},
        remove_columns=["messages"],
        num_proc=num_proc
    )
    return final_ds