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

def split_dataset(df, split=0.8, random_seed=42):
    """
    Splits a pandas DataFrame into training and validation sets.

    Args:
        df (pandas.DataFrame): The dataset to be split.
        split (float): The percentage of data for the training set. Default is 0.8.
        random_seed (int): The seed for reproducibility. Default is 42.

    Returns:
        tuple: Two DataFrames containing the training set and validation set.
    """
    train_df = df.sample(frac=split, random_state=random_seed)
    val_df = df.drop(train_df.index)
    
    return train_df, val_df

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

def process_dataset(df: str | pd.DataFrame | None = None, user_prompt: str = None, agent_response: str = None, processor=None, num_proc=4):
    """
    Args:
        df (str | pd.DataFrame | None): Path to the dataset file (str) or a
            pre-loaded DataFrame. If str, the dataset is loaded from that path.
            If pd.DataFrame, it is used directly. Defaults to None.
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
    if df is None:
        raise ValueError("'df' is required: provide a file path (str) or a pd.DataFrame.")
    if isinstance(df, str):
        df = load_dataset(df)
    elif not isinstance(df, pd.DataFrame):
        raise ValueError(f"'df' must be a file path (str) or a pd.DataFrame, got {type(df).__name__}.")
    dataset = format_dataset(df, user_prompt, agent_response, as_hf_dataset=True)
    final_ds = dataset.map(
        lambda x: {"text": processor.apply_chat_template(x["messages"], tokenize=False)},
        remove_columns=["messages"],
        num_proc=num_proc,
        load_from_cache_file = True,
    )
    return final_ds



def combine_dataset(
    paths: list[str],
    column_mapping: dict | None = None,
    output_mapping: dict | list | None = None,
    output_path: str | None = None,
) -> pd.DataFrame:
    """
    Load and combine multiple datasets into a single DataFrame with two columns:
    one for user prompts and one for agent responses.

    Args:
        paths (list[str]):
            List of file paths to load. Each file must be a supported format
            (Parquet or CSV).

        column_mapping (dict | None):
            Maps each dataset path to its [user_prompt_col, agent_output_col].

            - If None: all datasets are assumed to share **identical** column
              names. The function validates this assumption and raises a
              ValueError if any dataset has different column names.
            - If provided: must be a dict of the form
                {
                    "path/to/dataset1": ["user_col", "agent_col"],
                    "path/to/dataset2": ["question", "answer"],
                    ...
                }
              Every path in `paths` must have an entry.

        output_mapping (dict | list | None):
            **Required** when `column_mapping` is provided. Specifies the
            final column names of the combined DataFrame.

            Accepted forms:
                dict  → {"user": "col_name", "agent": "col_name"}
                list  → ["user_col_name", "agent_col_name"]

            Raises a ValueError if `column_mapping` is provided but
            `output_mapping` is None.

    Returns:
        pd.DataFrame: Combined DataFrame with exactly two columns
            [user_prompt_col, agent_response_col] as defined by
            `output_mapping` (or the shared column names when
            `column_mapping` is None).

        output_path (str | None):
            Optional path to save the combined DataFrame as a Parquet file.
            If None the result is only returned and not written to disk.
            Must end with '.parquet'.

    Returns:
        pd.DataFrame: Combined DataFrame with exactly two columns.

    Raises:
        ValueError: On any of the following conditions:
            - A path is not found or has an unsupported format.
            - `column_mapping` is None but datasets have inconsistent columns.
            - `column_mapping` is provided but `output_mapping` is None.
            - `column_mapping` is provided but a path has no mapping entry.
            - `output_mapping` has an invalid format.
            - `output_path` is provided but does not end with '.parquet'.
    """


    dfs: list[pd.DataFrame] = []
    for path in paths:
        dfs.append(load_dataset(path))


    if column_mapping is None:
        reference_cols = list(dfs[0].columns)

        if len(reference_cols) != 2:
            raise ValueError(
                f"When 'column_mapping' is None every dataset must have exactly "
                f"2 columns. '{paths[0]}' has {len(reference_cols)} column(s): "
                f"{reference_cols}"
            )

        for path, df in zip(paths[1:], dfs[1:]):
            if list(df.columns) != reference_cols:
                raise ValueError(
                    f"Column mismatch detected while combining datasets without "
                    f"a 'column_mapping'.\n"
                    f"  Reference columns (from '{paths[0]}'): {reference_cols}\n"
                    f"  Columns in '{path}': {list(df.columns)}\n"
                    f"Provide a 'column_mapping' and an 'output_mapping' to "
                    f"handle datasets with different column names."
                )

        combined = pd.concat(dfs, ignore_index=True)
        if output_path is not None:
            if not output_path.endswith(".parquet"):
                raise ValueError(f"'output_path' must end with '.parquet'. Got: '{output_path}'")
            combined.to_parquet(output_path, index=False)
        return combined

    if output_mapping is None:
        raise ValueError(
            "'output_mapping' is required when 'column_mapping' is provided. "
            "Supply it as a dict {'user': col_name, 'agent': col_name} "
            "or a list [user_col_name, agent_col_name]."
        )


    if isinstance(output_mapping, dict):
        if "user" not in output_mapping or "agent" not in output_mapping:
            raise ValueError(
                "'output_mapping' dict must contain both 'user' and 'agent' keys. "
                f"Got: {list(output_mapping.keys())}"
            )
        out_user_col = output_mapping["user"]
        out_agent_col = output_mapping["agent"]
    elif isinstance(output_mapping, list):
        if len(output_mapping) != 2:
            raise ValueError(
                "'output_mapping' list must have exactly 2 elements "
                f"[user_col_name, agent_col_name]. Got {len(output_mapping)} element(s)."
            )
        out_user_col, out_agent_col = output_mapping
    else:
        raise ValueError(
            f"'output_mapping' must be a dict or list, got {type(output_mapping).__name__}."
        )

 
    renamed_dfs: list[pd.DataFrame] = []
    for path, df in zip(paths, dfs):
        if path not in column_mapping:
            raise ValueError(
                f"No entry for '{path}' in 'column_mapping'. "
                f"Every path in 'paths' must have a corresponding mapping."
            )
        mapping_entry = column_mapping[path]
        if not (isinstance(mapping_entry, (list, tuple)) and len(mapping_entry) == 2):
            raise ValueError(
                f"column_mapping['{path}'] must be a list/tuple of 2 elements "
                f"[user_prompt_col, agent_output_col]. Got: {mapping_entry}"
            )
        src_user_col, src_agent_col = mapping_entry

        for col, label in [(src_user_col, "user"), (src_agent_col, "agent")]:
            if col not in df.columns:
                raise ValueError(
                    f"Column '{col}' (mapped as {label} column) not found in "
                    f"'{path}'. Available columns: {list(df.columns)}"
                )

        renamed_dfs.append(
            df[[src_user_col, src_agent_col]].rename(
                columns={src_user_col: out_user_col, src_agent_col: out_agent_col}
            )
        )

    combined = pd.concat(renamed_dfs, ignore_index=True)
    if output_path is not None:
        if not output_path.endswith(".parquet"):
            raise ValueError(f"'output_path' must end with '.parquet'. Got: '{output_path}'")
        combined.to_parquet(output_path, index=False)
    return combined


if __name__ == "__main__":
    pass
