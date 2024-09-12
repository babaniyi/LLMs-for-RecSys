
## **LLM-based Product Recommender System**

This repository contains a project focused on building a product recommendation system using Large Language Models (LLMs). The system is fine-tuned on Amazon appliances reviews and product metadata to predict the **exact** next products a user may purchase based on their historical interactions. The recommendation model leverages GPT-2 as a foundation and incorporates custom padding and data processing strategies to handle sequential product recommendations.

---

## **Project Overview**
This project explores the use of large language models (LLMs) for product recommendation systems, leveraging their natural language understanding capabilities to predict which items a user might purchase or review next. The primary focus is not solely on achieving high prediction accuracy but rather on learning if LLMs can be effectively applied in the recommendation domain.

### **Key Features:**
- **Sequential Recommendation System**: Predicts the next (maximum) 10 items a user might purchase in the future, based on their review history.
- **Custom Data Processing**: Handles temporal data (order by time to prevent data/target/information leakage) and processes the sequence of user interactions, ensuring proper padding and handling of missing data.
- **Fine-tuning GPT-2**: Fine-tuned GPT-2 medium model to handle recommendation tasks using user reviews, product metadata, and timestamps.
- **Evaluation Metrics**: Supports evaluation with metrics such as Recall, Precision, MRR, HR@10, and NDCG@10.
- **Custom Padding Strategies**: The model can handle missing data with customizable padding strategies (repeat, special token, or no padding).

#### Achieved Metrics:
- **Precision@10**: 2.19%
- **Recall@10**: 5.36%
- **Mean Reciprocal Rank (MRR)**: 0.041
- **Hit Rate at 10 (HR@10)**: 5.65%
- **Normalized Discounted Cumulative Gain at 10 (NDCG@10)**: 0.0429

These metrics provide insight into the model's ability to recommend relevant products. The precision@10 indicates that, on average, 2.19% of the top-10 recommended items are correct. Recall@10 suggests the model can retrieve nearly 5.36% of all relevant items. The MRR score of 0.041 shows that correct recommendations are ranked relatively low in the list. HR@10 reflects the percentage of times the correct product was recommended within the top-10, and NDCG@10 assesses both the ranking and relevance of the predicted items.

> While these numbers are modest and indicate that the model is far from perfect in recommending the exact next items, they still offer valuable insight into the potential of LLMs in capturing user intent and product features. The next iteration will involve comparing the LLM-based system's performance with more traditional baseline methods, such as collaborative filtering and matrix factorization, to assess its relative effectiveness.

To reiterate, the primary focus of this project is not just to maximize performance but to explore whether LLMs can offer a viable approach to recommendation tasks. The experiment is ongoing, and future improvements will focus on refining the model and comparing its performance with these baseline approaches. If one is focused on prediction accuracy alone, we could train the model on predicting the next product category a customer would purchase instead of exact items.

---

## **Evaluation**

The model supports evaluation metrics like **Recall**, **Precision**, **Mean Reciprocal Rank (MRR)**, **Hit Rate at 10 (HR@10)**, and **Normalized Discounted Cumulative Gain at 10 (NDCG@10)**. After generating predictions for a batch of reviews, the recommended items are compared with the actual next items to calculate these metrics.

You can evaluate the model using the `evaluate_metrics` function:

```python
evaluate_metrics(output_list, k=10)
```

### Evaluation Metrics:
- **Precision@10**: Proportion of relevant items in the top-10 predictions.
- **Recall@10**: Proportion of relevant items retrieved from the actual items.
- **MRR**: The reciprocal rank of the first relevant item in the top-10 recommendations.
- **HR@10**: Binary value indicating whether any relevant item appears in the top-10 predictions (1 if yes, 0 if no).
- **NDCG@10**: Discounted Cumulative Gain normalized over the ideal ranking, taking into account the rank position of relevant items.

### Updated Results

After applying these metrics, the following evaluation results were obtained:

```python
Average Metrics: {
    'precision@10': 0.0219, 
    'recall@10': 0.0536, 
    'mrr': 0.0413, 
    'hr@10': 0.0565, 
    'ndcg@10': 0.0429
}
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
