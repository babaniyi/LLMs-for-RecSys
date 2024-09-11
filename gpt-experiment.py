import json
from functools import partial
from importlib.metadata import version
import math
import os
import re
import time
import urllib
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

import pandas as pd
import tiktoken
from functools import partial
import torch
from torch.utils.data import Dataset, DataLoader
import time
from tqdm import tqdm
import psutil

import random
from datetime import datetime
import numpy as np


from gpt_reviews_finetuning import (
    generate,
    text_to_token_ids,
    token_ids_to_text,
    calc_loss_loader,
    train_model_simple,
    #plot_losses,

    GPTModel, 
    load_weights_into_gpt
)

from gpt_download import download_and_load_gpt2

#_____________________________________________________________________________________
# FUNCTIONS
#_____________________________________________________________________________________
def load_json_data(file_path):
    """
    The function gets a file path, opens it and organizes it in a list of dictionaries.
    :param file_path: path to a Json file with multiple Json objects.
    :return: list of dictionaries.
    """
    json_list = []
    with open(file_path, 'r') as file:
        for json_object in file:
            json_dict = json.loads(json_object.strip())
            json_list.append(json_dict)

    return json_list


def cast_unix_to_date(unix_time_str: pd.Series) -> pd.Series:
    """
    Converts a Series of Unix timestamps (milliseconds) to datetime strings.

    Args:
        unix_time_str (pd.Series): Series of Unix timestamps as strings.

    Returns:
        pd.Series: Series of datetime strings in the format 'YYYY-MM-DD'.
    """

    # Convert to numeric directly
    unix_timestamps = pd.to_numeric(unix_time_str)

    # Convert to datetime using NumPy for efficiency
    datetime_values = pd.to_datetime(unix_timestamps / 1000, unit='s')  # Convert milliseconds to seconds

    # Format as strings
    return datetime_values.dt.strftime('%Y-%m-%d')


def format_input_alpaca(entry):
    """ 
    Format entry according to the Alpaca-style prompt template. Example of data entry is as follows:

    Below is an instruction that describes a task. Write a response that appropriately completes the request.

    ### Instruction:
    Identify the correct spelling of the following word.

    ### Input:
    Occassion

    ### Response:
    The correct spelling is 'Occasion.'
    """
    
    instruction_text = (
        f"Below is an instruction that describes a task. "
        f"Write a response that appropriately completes the request."
        f"\n\n### Instruction:\n{entry['instruction']}"
    )

    input_text = f"\n\n### Input:\n{entry['input']}" if entry["input"] else ""

    return instruction_text + input_text
    

def format_input_phi(entry):
    """ 
    Phi-3 prompt template
    This prompt template is substantially shorter, which reduces the runtime and hardware requirements for 
    finetuning the LLM and generating text since the input prompts are shorter. 
    It formats the data entry as follows:

    <user>
    Identify the correct spelling of the following word: 'Occasion'

    <assistant>
    The correct spelling is 'Occasion'.
    """ 

    instruction_text = (
        f"<|user|>\n{entry['instruction']}"
    )

    input_text = f"\n{entry['input']}" if entry["input"] else ""

    return instruction_text + input_text


class AmazonDatasetAlpaca(Dataset):
    # Adjust Dataset to Alpaca template
    def __init__(self, data, tokenizer, special_chars):
        self.data = data

        # Pre-tokenize texts
        self.encoded_texts = []
        for entry in data:
            instruction_plus_input = format_input_alpaca(entry)
            response_text = f"\n\n### Response:\n{entry['output']}"
            full_text = instruction_plus_input + response_text
            self.encoded_texts.append(
                tokenizer.encode(full_text, allowed_special=special_chars)
            )

    def __getitem__(self, index):
        return self.encoded_texts[index]

    def __len__(self):
        return len(self.data)


class AmazonDatasetPhi(Dataset):
    # Adjust Dataset to Phi3 format
    def __init__(self, data, tokenizer, special_chars):
        self.data = data

        # Pre-tokenize texts
        self.encoded_texts = []
        for entry in data:
            instruction_plus_input = format_input_phi(entry)
            response_text = f"\n<|assistant|>:\n{entry['output']}"
            full_text = instruction_plus_input + response_text
            self.encoded_texts.append(
                tokenizer.encode(full_text, allowed_special=special_chars)
            )

    def __getitem__(self, index):
        return self.encoded_texts[index]

    def __len__(self):
        return len(self.data)


def details_to_sentence(details_dict):
    sentences = [f"{key.lower()} is: {value}" for key, value in details_dict.items()]
    return '. '.join(sentences) + '.'


def split_data_temporal(df, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    """
    Splits the data into train, validation, and test sets while preserving the temporal order of each user's interactions.

    Args:
    - df (pd.DataFrame): DataFrame containing user interactions with 'user_id' and 'date' columns.
    - train_ratio (float): Proportion of data to be used for training.
    - val_ratio (float): Proportion of data to be used for validation.
    - test_ratio (float): Proportion of data to be used for testing.

    Returns:
    - train_df (pd.DataFrame): Training set.
    - val_df (pd.DataFrame): Validation set.
    - test_df (pd.DataFrame): Test set.
    """
    assert train_ratio + val_ratio + test_ratio == 1.0, "Ratios must sum up to 1."

    # Sort data by user_id and date to maintain chronological order
    df_sorted = df.sort_values(by=['input', 'date']).reset_index(drop=True)

    # Create empty DataFrames for train, validation, and test
    train_df = pd.DataFrame()
    val_df = pd.DataFrame()
    test_df = pd.DataFrame()

    # Split each user's data chronologically
    for user_id, group in df_sorted.groupby('input'):
        num_interactions = len(group)

        if num_interactions < 3:
            # If a user has fewer than 3 interactions, skip them
            continue

        # Calculate indices for train, validation, and test splits
        train_end = int(train_ratio * num_interactions)
        val_end = train_end + int(val_ratio * num_interactions)

        # Ensure there's at least one interaction in each split if possible
        if train_end == 0:
            train_end = 1
        if val_end == train_end:
            val_end = train_end + 1

        # Concatenate each user's splits to the respective DataFrame
        train_df = pd.concat([train_df, group.iloc[:train_end]])
        val_df = pd.concat([val_df, group.iloc[train_end:val_end]])
        test_df = pd.concat([test_df, group.iloc[val_end:]])

    # Reset index after concatenation
    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    return train_df, val_df, test_df



def get_next_10_items(df, padding_strategy='repeat', pad_token="<|endoftext|>"):
    """
    For each row in the dataframe, get the next 10 items the user interacted with, 
    while handling cases where less than 10 next items are available.

    Args:
    - df (pd.DataFrame): DataFrame containing user interactions, sorted by 'user_id' and 'date'.
    - padding_strategy (str): How to pad the sequence if less than 10 items are found ('none', 'repeat', 'pad_token').
    - pad_token (int): The padding token to use if padding_strategy is 'pad_token'.

    Returns:
    - pd.DataFrame: DataFrame with additional 'next_10_items' column.
    """

    # Ensure the DataFrame is sorted by user_id and date for correct sequential order
    df_sorted = df.sort_values(by=['user_id', 'date']).reset_index(drop=True)

    # Initialize a dictionary to collect the new column data
    next_10_items_data = []

    # Group by each user
    for user_id, user_data in df_sorted.groupby('user_id'):
        user_data = user_data.reset_index(drop=True)
        num_interactions = len(user_data)

        # Skip users with fewer than 11 interactions
        if num_interactions < 11:
            continue

        # Iterate over each row for the current user
        for i in range(num_interactions):
            if i + 10 < num_interactions:
                # Get the next 10 item_ids
                next_items = user_data.loc[i + 1:i + 10, 'item_id'].tolist()
            else:
                # Handle cases where there are fewer than 10 remaining items
                next_items = user_data.loc[i + 1:, 'item_id'].tolist()
                
                if len(next_items) < 10:
                    if padding_strategy == 'repeat':
                        next_items = next_items + [next_items[-1]] * (10 - len(next_items))
                    elif padding_strategy == 'pad_token':
                        next_items = next_items + [pad_token] * (10 - len(next_items))
                    elif padding_strategy == 'none':
                        # Skip rows where next_10_items do not exist
                        continue
            
            # Check if a list contains the same different or same item_ids all through the list
            #if len(set(next_items)) > 1:

            next_10_items_data.append((user_data.loc[i, 'user_id'],
                                    user_data.loc[i, 'item_id'],
                                    user_data.loc[i, 'date'],
                                    ", ".join(next_items))
                                    )

    # Create a new DataFrame with the collected data
    df_next_10_items = pd.DataFrame(next_10_items_data, columns=['user_id', 'item_id', 'date', 'next_10_items'])
    return df_next_10_items


class LinearWithLoRA(torch.nn.Module):
    def __init__(self, linear, rank, alpha):
        super().__init__()
        self.linear = linear
        self.lora = LoRALayer(
            linear.in_features, linear.out_features, rank, alpha
        )

    def forward(self, x):
        return self.linear(x) + self.lora(x)


class LoRALayer(torch.nn.Module):
    def __init__(self, in_dim, out_dim, rank, alpha):
        super().__init__()
        self.A = torch.nn.Parameter(torch.empty(in_dim, rank))
        torch.nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))  # similar to standard weight initialization
        self.B = torch.nn.Parameter(torch.zeros(rank, out_dim))
        self.alpha = alpha

    def forward(self, x):
        x = self.alpha * (x @ self.A @ self.B)
        return x


def replace_linear_with_lora(model, rank, alpha):
    for name, module in model.named_children():
        if isinstance(module, torch.nn.Linear):
            # Replace the Linear layer with LinearWithLoRA
            setattr(model, name, LinearWithLoRA(module, rank, alpha))
        else:
            # Recursively apply the same function to child modules
            replace_linear_with_lora(module, rank, alpha)



def custom_collate_fn(
    batch,
    pad_token_id=None,
    ignore_index=-100,
    allowed_max_length=None,
    device="cpu"
    ):
    # Find the longest sequence in the batch
    batch_max_length = max(len(item)+1 for item in batch)

    # Pad and prepare inputs and targets
    inputs_lst, targets_lst = [], []

    for item in batch:
        new_item = item.copy()
        # Add an <|endoftext|> token
        new_item += [pad_token_id]
        # Pad sequences to max_length
        padded = new_item + [pad_token_id] * (batch_max_length - len(new_item))
        inputs = torch.tensor(padded[:-1])  # Truncate the last token for inputs
        targets = torch.tensor(padded[1:])  # Shift +1 to the right for targets

        # New: Replace all but the first padding tokens in targets by ignore_index
        mask = targets == pad_token_id
        indices = torch.nonzero(mask).squeeze()
        if indices.numel() > 1:
            targets[indices[1:]] = ignore_index

        # New: Optionally truncate to maximum sequence length
        if allowed_max_length is not None:
            inputs = inputs[:allowed_max_length]
            targets = targets[:allowed_max_length]

        inputs_lst.append(inputs)
        targets_lst.append(targets)

    # Convert list of inputs and targets to tensors and transfer to target device
    inputs_tensor = torch.stack(inputs_lst).to(device)
    targets_tensor = torch.stack(targets_lst).to(device)

    return inputs_tensor, targets_tensor



def plot_losses(epochs_seen, tokens_seen, train_losses, val_losses, plot_name):
    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Plot training and validation loss against epochs
    ax1.plot(epochs_seen, train_losses, label="Training loss")
    ax1.plot(epochs_seen, val_losses, linestyle="-.", label="Validation loss")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax1.legend(loc="upper right")
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))  # only show integer labels on x-axis

    # Create a second x-axis for tokens seen
    ax2 = ax1.twiny()  # Create a second x-axis that shares the same y-axis
    ax2.plot(tokens_seen, train_losses, alpha=0)  # Invisible plot for aligning ticks
    ax2.set_xlabel("Tokens seen")

    fig.tight_layout()  # Adjust layout to make room
    print(f"Plot saved as {plot_name}")
    plt.savefig(plot_name)
    # plt.show()


def main(phi3_prompt=False, lora=False):
    #######################################
    # Print package versions
    #######################################
    print()
    pkgs = [
        "matplotlib",  # Plotting library
        "tiktoken",    # Tokenizer
        "torch",       # Deep learning library
        "tqdm",        # Progress bar
        "tensorflow",  # For OpenAI's pretrained weights
    ]
    for p in pkgs:
        print(f"{p} version: {version(p)}")
    print(50*"-")



    #######################################
    # 1. PROCESSING DATA
    #######################################

    PATH_REVIEWS_DATA = "data/raw/Appliances.jsonl"
    PATH_META_DATA = "data/raw/meta_Appliances.jsonl"

    data_reviews_df = pd.DataFrame(load_json_data(PATH_REVIEWS_DATA))
    data_meta_df = pd.DataFrame(load_json_data(PATH_META_DATA))

    # Join user and item metadata
    data_reviews_df.drop(columns=["images", "title"], inplace=True)
    full_df = data_reviews_df.merge(data_meta_df, on=["parent_asin"], how="inner")

    # 1.01 Select features
    #_______________________________________________________________________________

    selected_cols = [
                        # User reviews data
                        "rating", "text", "parent_asin", "user_id", "timestamp",
                        # Product meta data
                        'main_category', 'title', 'details', 'price', 
                    ]

    full_df_selected = full_df[full_df["verified_purchase"] == True]
    full_df_selected = full_df_selected[selected_cols]
    del full_df

    ## 1.02 Data user and item to custom format
    # Prepare the data by setting the user_id and item_id to a special token format
    #_______________________________________________________________________________
    full_df_selected["date"] = cast_unix_to_date(full_df_selected["timestamp"])
    full_df_selected = full_df_selected.sort_values(by=["date", "user_id"])

    full_df_selected["user_id"]= "<|user_" + full_df_selected["user_id"] + "|>"
    full_df_selected["item_id"] = "<|item_" + full_df_selected["parent_asin"] + "|>"

    full_df_selected['item_details'] = full_df_selected['details'].apply(details_to_sentence)

    # 1.03  Extract 10 items purchased by the user
    #       Extract the next 10 items purchased by the user
    #_______________________________________________________________________________
    df_model = get_next_10_items(full_df_selected, 'pad_token')

    join_keys = ["user_id", "item_id", "date"]

    tmp_df = full_df_selected.drop(["timestamp", "details", "parent_asin"], axis=1).drop_duplicates().reset_index(drop=True)
    df_model_ready = df_model.merge(tmp_df, on=join_keys, how="inner")

    del df_model, tmp_df, full_df_selected

    # Rename the columns to descriptive names

    new_cols_names = {
                        "rating": "Rating of the item by user from 1 to 5",
                        "text": "Text of the user review",

                        "main_category": "Main category of the item",
                        'title': "Item name", 

                        'item_details': "Item details",
                        'item_id': 'Item id',
                        'price': 'Price USD'
                    }

    df_model_ready = df_model_ready.rename(columns=new_cols_names)


    # 1.03 Tokenizer
    #        Lets the identify user_id, item and "<|endoftext|>" as additional tokens
    #_______________________________________________________________________________
    special_user_item_ids = ["<|endoftext|>"] + df_model_ready["user_id"].unique().tolist() + df_model_ready["Item id"].unique().tolist()
    special_user_item_ids = set(special_user_item_ids)

    tokenizer = tiktoken.get_encoding("gpt2")   
    #tokenizer = tiktoken.get_encoding("o200k_base")

    end_of_text_id = np.array(tokenizer.encode("<|endoftext|>", allowed_special=special_user_item_ids)).item()


    #######################################
    # 2. DATA PREP - Data Loaders, Custom collate, etc.
    #######################################

    ## 2.01 Data preparation - input, output and instruction
    #_______________________________________________________________________________

    df_model_ready.rename(columns={"user_id": "input", 'next_10_items': "output"}, inplace=True)

    aux_cols =[
                'Item id','Rating of the item by user from 1 to 5','Text of the user review',
                'Main category of the item','Item name','Price USD','Item details'
            ]

    # Convert integer columns to string
    df_model_ready['Rating of the item by user from 1 to 5'] = df_model_ready['Rating of the item by user from 1 to 5'].map(str)
    df_model_ready['Price USD'] = df_model_ready['Price USD'].map(str)



    user_question = "Given a user purchased an an item with the following details, predict the next 10 items the user would  purchase. "
    df_model_ready["instruction"] =  user_question + df_model_ready[aux_cols].apply(lambda x: '. '.join(f"{col} is {value}" for col, value in x.items()), axis=1)


    ## 2.02 Split to train, validation and test
    #_______________________________________________________________________________
    train_df, val_df, test_df = split_data_temporal(df_model_ready, train_ratio=0.85, val_ratio=0.1, test_ratio=0.05)

    train_df= train_df[["input", "output", "instruction"]]
    val_df= val_df[["input", "output", "instruction"]]
    test_df= test_df[["input", "output", "instruction"]]


    train_data = train_df.to_dict(orient="records")
    test_data = test_df.to_dict(orient="records")
    val_data = val_df.to_dict(orient="records")

    del train_df, val_df, test_df


    print("Training set length:", len(train_data))
    print("Validation set length:", len(val_data))
    print("Test set length:", len(test_data))
    print(50*"-")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    print(50*"-")

    # 2.03  Set custom_collate_fn
    #_______________________________________________________________________________
    customized_collate_fn = partial(custom_collate_fn, pad_token_id=end_of_text_id, allowed_max_length=1024, device=device)
    if phi3_prompt:
        CustomDataset = AmazonDatasetPhi
    else:
        CustomDataset = AmazonDatasetAlpaca
    
    num_workers = 0
    batch_size = 4

    torch.manual_seed(123)

    # 2.04  Data Loaders
    #_______________________________________________________________________________
    train_dataset = CustomDataset(train_data, tokenizer, special_user_item_ids)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        collate_fn=customized_collate_fn,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
    )

    val_dataset = CustomDataset(val_data, tokenizer, special_user_item_ids)
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        collate_fn=customized_collate_fn,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
    )


    #######################################
    # 3. Load pretrained model
    #######################################
    BASE_CONFIG = {
        "vocab_size": 50257,     # Vocabulary size
        "context_length": 1024,  # Context length
        "drop_rate": 0.0,        # Dropout rate
        "qkv_bias": True         # Query-key-value bias
    }

    model_configs = {
        "gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
        "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
        "gpt2-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
        "gpt2-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
    }

    CHOOSE_MODEL = "gpt2-medium (355M)"

    BASE_CONFIG.update(model_configs[CHOOSE_MODEL])

    model_size = CHOOSE_MODEL.split(" ")[-1].lstrip("(").rstrip(")")
    settings, params = download_and_load_gpt2(model_size=model_size, models_dir="gpt2")

    model = GPTModel(BASE_CONFIG)
    load_weights_into_gpt(model, params)
    model.eval()
    model.to(device)

    print("Loaded model:", CHOOSE_MODEL)
    print(50*"-")


    if lora:
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total trainable parameters before: {total_params:,}")

        for param in model.parameters():
            param.requires_grad = False

        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total trainable parameters after: {total_params:,}")
        replace_linear_with_lora(model, rank=16, alpha=16)

        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total trainable LoRA parameters: {total_params:,}")
        model.to(device)


    #######################################
    # 4 Finetuning the model
    #######################################

    # Initial Loss
    #_______________________________________________________________________________
    print("Initial losses")
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=4)
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=4)

    print("   Training loss:", train_loss)
    print("   Validation loss:", val_loss)
    print("   Percentage change:", round(100 * (val_loss - train_loss)/train_loss, 2), "%")


    # Training
    #_______________________________________________________________________________
    start_time = time.time()

    num_epochs = 2
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.00005, weight_decay=0.1)

    torch.manual_seed(123)

    start_context = format_input_phi(val_data[0]) if phi3_prompt else format_input_alpaca(val_data[0])

    train_losses, val_losses, tokens_seen = train_model_simple(
        model, train_loader, val_loader, optimizer, device,
        num_epochs=num_epochs, eval_freq=5, eval_iter=5,
        start_context=start_context, tokenizer=tokenizer,
        special_chars=special_user_item_ids,
    )
    end_time = time.time()
    execution_time_minutes = (end_time - start_time) / 60
    print(f"Training completed in {execution_time_minutes:.2f} minutes.")

    # Plot losses
    #_______________________________________________________________________________
    epochs_tensor = torch.linspace(0, num_epochs, len(train_losses))
    plot_name = "loss-plot.pdf"
    if phi3_prompt:
        plot_name = plot_name.replace(".pdf", "-phi3-prompt.pdf")
    if lora:
        plot_name = plot_name.replace(".pdf", "-lora.pdf")
    if not any([phi3_prompt, lora]):
        plot_name = plot_name.replace(".pdf", "-baseline.pdf")

    plot_losses(epochs_tensor, tokens_seen, train_losses, val_losses, plot_name)
    print(50*"-")


    #######################################
    # Saving results
    #######################################
    print("Generating responses")
    for i, entry in tqdm(enumerate(test_data), total=len(test_data)):

        input_text = format_input_phi(entry) if phi3_prompt else format_input_alpaca(entry)

        token_ids = generate(
            model=model,
            idx=text_to_token_ids(input_text, tokenizer, special_user_item_ids).to(device),
            max_new_tokens=256,
            context_size=BASE_CONFIG["context_length"],
            eos_id=end_of_text_id,
        )
        generated_text = token_ids_to_text(token_ids, tokenizer)

        if phi3_prompt:
            response_text = generated_text[len(input_text):].replace("<|assistant|>:", "").strip()
        else:
            response_text = generated_text[len(input_text):].replace("### Response:", "").strip()

        test_data[i]["model_response"] = response_text

    test_data_path = "data/output/amazon-reviews-data-with-response.json"
    file_name = f"{re.sub(r'[ ()]', '', CHOOSE_MODEL) }-sft.pth"

    if phi3_prompt:
        test_data_path = test_data_path.replace(".json", "-phi3-prompt.json")
        file_name = file_name.replace(".pth", "-phi3-prompt.pth")
    if lora:
        test_data_path = test_data_path.replace(".json", "-lora.json")
        file_name = file_name.replace(".pth", "-lora.pth")
    if not any([phi3_prompt, lora]):
        test_data_path = test_data_path.replace(".json", "-baseline.json")
        file_name = file_name.replace(".pth", "-baseline.pth")

    with open(test_data_path, "w") as file:
        json.dump(test_data, file, indent=4)  # "indent" for pretty-printing
    print(f"Responses saved as {test_data_path}")


    torch.save(model.state_dict(), file_name)
    print(f"Model saved as {file_name}")



if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(
        description="Amazon reviews instruction data finetuned using a GPT model"
    )
    options = {"baseline", "phi3_prompt", "lora"}
    parser.add_argument(
        "--run_solution",
        type=str,
        default="last_block",
        help=(
            f"Which experiment to run. Options: {options}."
        )
    )
    args = parser.parse_args()

    args = parser.parse_args()

    if args.run_solution == "baseline":
        main() # use alpaca style and no - lora
    elif args.run_solution == "phi3_prompt":
        main(phi3_prompt=True) # use phi3 style and no -lora
    elif args.run_solution == "lora":
        main(lora=True) # use alpaca and lora
    else:
        raise ValueError(f"{args.run_solution} is not a valid --args.run_solution option. Options: {options}")









   







    





    

