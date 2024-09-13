# **LLM-based Product Recommender System**

This repository contains a project focused on building a product recommendation system using Large Language Models (LLMs). The system is fine-tuned on Amazon appliance reviews and product metadata to predict the **exact** next products a user may purchase based on their historical interactions. The recommendation model leverages GPT-2 as a foundation and incorporates custom padding and data processing strategies to handle sequential product recommendations.

---

## **Project Overview**
This project explores the use of large language models (LLMs) for product recommendation systems, leveraging their natural language understanding capabilities to predict which items a user might purchase or review next. The primary focus is not solely on achieving high prediction accuracy but rather on learning if LLMs can be effectively applied in the recommendation domain.

This code was run using [Lightning AI](lightning.ai) L4 GPU.

### **Key Features:**
- **Sequential Recommendation System**: Predicts the next (maximum) 10 items a user might purchase in the future, based on their review history.
- **Custom Data Processing**: Handles temporal data (order by time to prevent data/target/information leakage) and processes the sequence of user interactions, ensuring proper padding and handling of missing data.
- **Fine-tuning GPT-2**: Fine-tuned GPT-2 medium model to handle recommendation tasks using user reviews, product metadata, and timestamps.
- **Evaluation Metrics**: Supports evaluation with metrics such as Recall, Precision, MRR, Hit Rate, Normalized Discounted Cumulative Gain and more.
- **Custom Padding Strategies**: The model can handle missing data with customizable padding strategies (repeat, special token, or no padding).


#### Achieved Metrics:
- **Precision@10**: $2.2$%
- **Recall@10**: $5.5$%
- **Mean Reciprocal Rank (MRR)**: $0.04$%
- **Hit Rate at 10 (HR@10)**: $5.7$%
- **Normalized Discounted Cumulative Gain at 10 (NDCG@10)**: $0.04$%

These metrics provide insight into the model's ability to recommend relevant products. The precision@10 indicates that, on average, 2.2% of the top-10 recommended items are correct. Recall@10 suggests the model can retrieve nearly 6% of all relevant items. The MRR score of 0.041 shows that correct recommendations are ranked relatively low in the list. HR@10 reflects the percentage of times the correct product was recommended within the top-10, and NDCG@10 assesses both the ranking and relevance of the predicted items.

> While these numbers are modest and indicate that the model is far from perfect in recommending the exact next items, they still offer valuable insight into the potential of LLMs in capturing user intent and product features.

To reiterate, the primary focus of this project is not just to maximize performance but to explore whether LLMs can offer a viable approach to recommendation tasks. The experiment is ongoing, and future improvements will focus on refining the model and comparing its performance with these baseline approaches (such as collaborative filtering and matrix factorization). If one is focused on prediction accuracy alone, we could train the model on predicting the next product category a customer would purchase instead of exact items.

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

This project uses Amazon Appliance Reviews data for training, validation, and testing. The dataset can be broadly categorised into:
- **User Reviews**: Including `user_id`, `review_text`, `parent_asin`, and `timestamp`.
- **Product Metadata**: Including `parent_asin`, `title`, `category`, `features`, `price`, and more.

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
    git clone https://github.com/babaniyi/LLMs-for-RecSys.git
    cd LLMs-for-RecSys
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

If you want to run the model using Phi3-style prompt **with** LORA
```bash
python gpt-experiment.py --run_solution phi3_and_lora
```


---

## **Usage**

### **Data Preprocessing**

To curate quality training data, we focus on users who have purchased at least 10 materials.
The data should be preprocessed before training. The `get_next_10_items` function processes user interaction data and ensures that each entry contains a list of the next 10 items based on historical interactions.

```python
df_with_next_10_items = get_next_10_items_optimized(df, padding_strategy='repeat', pad_token=50257)
```

### **Train/Test/Validation Split**
You can split the data while preserving its temporal time order to ensure no data and target leakage using the `temporal_train_val_test_split` function:

```python
train_df, val_df, test_df = split_data_temporal(df_with_next_10_items, train_ratio=0.7, val_ratio=0.1, test_ratio=0.2)
```


---

## **Training the Model**
First, we calculate the initial training and validation set loss before we start training (the goal is to minimize the loss). The initial train and validation losses can be visualised in a plot `loss-plot-...pdf` which is generated by the model.

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

The model supports evaluation metrics like **Recall**, **Precision**, **Mean Reciprocal Rank (MRR)**, **Hit Rate at 10 (HR@10)**, and **Normalized Discounted Cumulative Gain at 10 (NDCG@10)**. After generating predictions for a batch of reviews, the recommended items are compared with the actual next items to calculate these metrics.

### Evaluation Metrics:
- **Precision@10**: Proportion of relevant items in the top-10 predictions.
- **Recall@10**: Proportion of relevant items retrieved from the actual items.
- **MRR**: The reciprocal rank of the first relevant item in the top-10 recommendations.
- **HR@10**: Binary value indicating whether any relevant item appears in the top-10 predictions (1 if yes, 0 if no).
- **NDCG@10**: Discounted Cumulative Gain normalized over the ideal ranking, taking into account the rank position of relevant items.

You can evaluate the model using the `evaluate_metrics` function:

```python
evaluate_metrics(output_list, k=10)
```

Example of the output list, which includes the output and model response:

```python
[{
    "input": "<|user_AE2BFR2EGPHCYISLCTPOX2AQHKVQ|>",
    "output": "<|item_B07FTFD1XB|>, <|endoftext|>, <|endoftext|>, <|endoftext|>, <|endoftext|>, <|endoftext|>, <|endoftext|>, <|endoftext|>, <|endoftext|>, <|endoftext|>",
    "model_response": "<|item_B0081E9HRY|>, <|item_B07CP1KY9M|>, <|item_B07MWVCVR4|>, <|item_B07QVKSMKK|>, <|item_B07PJ8H3W5|>, <|item_B07PJ8H3W5|>, <|item_B07PJ8H3W5|>, <|item_B07V3ZF517|>, <|item_B07V3ZF517|>,"
}]
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
> Raschka, Sebastian. Build A Large Language Model (From Scratch). Manning, 2024. ISBN: 978-1633437166.

> @article{hou2024bridging,
  title={Bridging Language and Items for Retrieval and Recommendation},
  author={Hou, Yupeng and Li, Jiacheng and He, Zhankui and Yan, An and Chen, Xiusi and McAuley, Julian},
  journal={arXiv preprint arXiv:2403.03952},
  year={2024}
}

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
