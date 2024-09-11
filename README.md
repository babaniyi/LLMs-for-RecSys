# **LLM-based Product Recommender System**

This repository contains a project focused on building a product recommendation system using Large Language Models (LLMs). The system is fine-tuned on Amazon reviews data to predict the next products a user may purchase based on their historical interactions. The recommendation model leverages GPT-2 as a foundation and incorporates custom padding and data processing strategies to handle sequential product recommendations.
It is interesting to note that in the test data where the user did not make an purchase, the model (can) predicts items the user could purchase which is similar to collaborative filtering in traditional recommender systems.

---

## **Project Overview**

The goal of this project is to fine-tune a pre-trained GPT-2 model to generate personalized product recommendations for users, based on historical reviews and item details. The project follows a sequential recommendation strategy, where the model predicts the next products a user may purchase. 

### **Key Features:**
- **Sequential Recommendation System**: Predicts the next (maximum) 10 items a user might purchase, based on their review history.
- **Custom Data Processing**: Handles temporal data and processes the sequence of user interactions, ensuring proper padding and handling of missing data.
- **Fine-tuning GPT-2**: Fine-tuned GPT-2 medium model to handle recommendation tasks using user reviews, product metadata, and timestamps.
- **Evaluation Metrics**: Supports evaluation with metrics such as Recall@10, Precision@10, and more.
- **Custom Padding Strategies**: The model can handle missing data with customizable padding strategies (repeat, special token, or no padding).

---

## **Table of Contents**

1. [Project Overview](#project-overview)
2. [Data](#data)
3. [Model Architecture](#model-architecture)
4. [Installation](#installation)
5. [Usage](#usage)
6. [Training the Model](#training-the-model)
7. [Evaluation](#evaluation)
8. [Contributing](#contributing)
9. [License](#license)

---

## **Data**

This project uses Amazon Appliance Reviews data for training, validation, and testing. The dataset consists of:
- **User Reviews**: Including `user_id`, `review_text`, `item_id`, and `timestamp`.
- **Product Metadata**: Including `item_id`, `title`, `category`, `features`, `price`, and more.

Each row in the dataset is processed to generate the next 10 items that the user interacted with. Data is split into training, validation, and test sets, keeping the temporal nature of user interactions intact.

[Download data here](https://amazon-reviews-2023.github.io/)

---

## **Model Architecture**

The project uses the GPT-2 medium architecture, consisting of:
- **Embedding Dimension**: 1024
- **Number of Layers**: 24
- **Number of Attention Heads**: 16

The model is fine-tuned to take as input:
- **User Reviews**: User-written text.
- **Item Metadata**: Key features like title, category, and other details.

### **Loss Function**:
We calculate the loss using cross-entropy between the predicted next items and the actual items the user interacted with.

---

## **Installation**

To run this project locally, follow these steps:

1. **Clone the Repository**:
    ```bash
    git clone https://github.com/yourusername/recsys-llm.git
    cd recsys-llm
    ```

2. **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

3. **Set Up the Data**:
    - Download the data `Appliances.jsonl` and `meta_Appliances.jsonl` [data here](https://amazon-reviews-2023.github.io/)
    - Place the data in `data/raw/` folder

4. **Configure Environment Variables**:
    If using CUDA, ensure your environment is properly set up for PyTorch GPU usage.


To run the script. Go to the command line and run the following:

If you want to run the model using Alpaca-style prompt **without** LORA (which is our baseline model). Run 
```bash
python gpt-experiment.py --run_solution baseline
```

Use Alpaca-style prompt **with** LORA
```bash
python gpt-experiment.py --run_solution alpaca_and_lora
```

If you want to run the model using Phi3-style prompt **without** LORA
```bash
python gpt-experiment.py --run_solution phi3_prompt
```

```bash
If you want to run the model using Phi3-style prompt **with** LORA
python gpt-experiment.py --run_solution phi3_and_lora
```


---

## **Usage**

### **Data Preprocessing**

To curate quality training data, we focus on users that has purchased at least 10 materials.
The data should be preprocessed before training. The `get_next_10_items` function processes user interaction data and ensures that each entry contains a list of the next 10 items based on historical interactions.

```python
df_with_next_10_items = get_next_10_items_optimized(df, padding_strategy='repeat', pad_token=50257)
```

### **Train/Test/Validation Split**
You can split the data while preserving its temporal time order to ensure there is no data and target leakage using the `temporal_train_val_test_split` function:

```python
train_df, val_df, test_df = split_data_temporal(df_with_next_10_items, train_ratio=0.7, val_ratio=0.1, test_ratio=0.2)
```


---

## **Training the Model**
First, we calculate the initial training and validation set loss before we start training (the goal is to minimize the loss). The initial train and validation losses can be visualised in a plot `loss-plot-...pdf` which is generated by the model.

Finally, to train the model on the data, we use the `train_model_simple` function:

```python
train_losses, val_losses, tokens_seen = train_model_simple(
        model, train_loader, val_loader, optimizer, device,
        num_epochs=num_epochs, eval_freq=5, eval_iter=5,
        start_context=start_context, tokenizer=tokenizer,
        special_chars=special_user_item_ids,
        )
```

Make sure that you have your CUDA environment set up properly if using GPUs. The model will output training and validation loss metrics after every evaluation step.

An example of data entry is given below. 
- **Input:** We set the input as the user_id in the following format <|user|>.
- **Output:** We set it as the next items purchased by the user and encoded as <|item_id|>. If items purchased are exhausted, it's encoded using a special token <|endoftext|>
- **Prompt:** We instruct the model with the following prompt `Given a user purchased an an item with the following details, predict the next 10 items the user would  purchase.`

Example of the processed data sent to the model.
```
{
    "input": "<|user_AE2BFR2EGPHCYISLCTPOX2AQHKVQ|>",
    "output": "<|item_B07FTFD1XB|>, <|endoftext|>, <|endoftext|>, <|endoftext|>, <|endoftext|>, <|endoftext|>, <|endoftext|>, <|endoftext|>, <|endoftext|>, <|endoftext|>",
    "instruction": "Given a user purchased an an item with the following details, predict the next 10 items the user would  purchase. Item id is <|item_B00Y5UZD5I|>. Rating of the item by user from 1 to 5 is 2.0. Text of the user review is To expensive, the quality is questionable... The coating is thin and becomes damaged very easy! I would not recommend for these reasons. Main category of the item is Tools & Home Improvement. Item name is GE WD28X10399 Upper Rack. Price USD is nan. Item details is manufacturer is: GE. part number is: WD28X10399. item weight is: 0.32 ounces. product dimensions is: 4.1 x 0.4 x 2.5 inches. item model number is: WD28X10399. item package quantity is: 1. certification is: Certified frustration-free, Not Applicable. included components is: Appliance-replacement-parts, Appliance Parts & Accessories. batteries included? is: No. batteries required? is: No. best sellers rank is: {'Tools & Home Improvement': 604524, 'Parts & Accessories': 91543}. date first available is: November 21, 2013.",
    },
```


---

## **Evaluation**

The model supports evaluation metrics like Recall, Precision, Mean Reciprocal Rank. After generating predictions for a batch of reviews, the recommended items are compared with the actual next items to calculate these metrics.

You can evaluate the model using the `evaluate_recall_precision` function:

```python
evaluate_metrics(output_list, k=10)
```
Example of the output list which includes the output and model response.
```
[{
    "input": "<|user_AE2BFR2EGPHCYISLCTPOX2AQHKVQ|>",
    "output": "<|item_B07FTFD1XB|>, <|endoftext|>, <|endoftext|>, <|endoftext|>, <|endoftext|>, <|endoftext|>, <|endoftext|>, <|endoftext|>, <|endoftext|>, <|endoftext|>",
    "model_response": "<|item_B0081E9HRY|>, <|item_B07CP1KY9M|>, <|item_B07MWVCVR4|>, <|item_B07QVKSMKK|>, <|item_B07PJ8H3W5|>, <|item_B07PJ8H3W5|>, <|item_B07PJ8H3W5|>, <|item_B07V3ZF517|>, <|item_B07V3ZF517|>,"
    },
    
    ]
```

---


## **Contributing**

We welcome contributions to this project! Please follow the steps below if you would like to contribute:

1. Fork the repository.
2. Create a feature branch: `git checkout -b feature/feature-name`.
3. Commit your changes: `git commit -m 'Add feature'`.
4. Push to the branch: `git push origin feature/feature-name`.
5. Open a pull request.

---

## **License**

This project is licensed under the MIT License. See the `LICENSE` file for more details.

---

## **Contact**

For any questions or issues, feel free to open an issue on this repository or reach out to `horlaneyee@gmail.com`.

---

### **Acknowledgements**
- Sebastian Raschka: For his wonderful book/course that serves as the main bedrock and motivation of this project. [Building LLMs from Scratch](https://www.manning.com/books/build-a-large-language-model-from-scratch)


### **Citation**
If you use this project or its code, please consider citing it as follows:

```
@article{llms-for-product-recsys,
    title       = {LLM-based Product Recommender System},
    author      = {Babaniyi Olaniyi},
    journal     = {babaniyi.com},
    year        = {2024},
    month       = {September},
    github      = {https://github.com/babaniyi/LLMs-for-RecSys}
}
```