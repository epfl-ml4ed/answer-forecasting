# Student Answer Forecasting: Transformer-Driven Answer Choice Prediction for Language Learning
This repository is the official implementation of the EDM 2024 paper entitled ["Student Answer Forecasting: Transformer-Driven Answer Choice Prediction for Language Learning"]() written by [Elena Grazia Gado*](https://www.linkedin.com/in/elena-grazia-gado-a73646248/), [Tommaso Martorella*](https://www.linkedin.com/in/tommymarto/), [Luca Zunino*](https://www.linkedin.com/in/lucazunino/), [Paola Mejia-Domenzain](), [Vinitra Swamy](http://github.com/vinitra), [Jibril Frej](https://github.com/Jibril-Frej), and [Tanja Käser](https://people.epfl.ch/tanja.kaeser/?lang=en).

<img width="100%" alt="Student Answer Forecasting pipeline" src="https://github.com/epfl-ml4ed/answer-forecasting/assets/41111850/af8e5929-19da-4f0b-936a-99aac9e96e7f">

## Overview

Intelligent Tutoring Systems (ITS) enhance personalized learning by predicting student answers to provide immediate and customized instruction. However, recent research has primarily focused on the correctness of the answer rather than the student's performance on specific answer choices, limiting insights into students' thought processes and potential misconceptions. To address this gap, we present MCQStudentBert, a new model to predict student answers. Our goal is to move beyond the traditional binary correct-incorrect prediction models to forecast the specific answer choices students are likely to make. This enables practitioners to easily extend the model to new answer choices or remove answer choices for the same multiple-choice question (MCQ) without retraining the model. Our pipeline leverages the capabilities of Large Language Models (LLMs) to integrate contextual understanding of students' answering history along with the text of the questions and answers. In particular, we compare MLP, LSTM, BERT, and Mistral 7B architectures to generate embeddings from students' past interactions, which are then incorporated into a finetuned BERT's answer-forecasting mechanism. We apply our pipeline to a dataset of language learning MCQ, gathered from an ITS with over 10,000 students to explore the predictive accuracy of MCQStudentBert, which incorporates student interaction patterns, in comparison to correct answer prediction and traditional mastery-learning feature-based approaches. This research contributes a novel student answer forecasting case study in German (often not represented in traditional educational tasks) that incorporates students' history alongside question context to predict the exact answer choice(s) the student will choose. Through these contributions, MCQStudentBert opens the door to more personalized content, modularization, and granular support.

#### *alternative title*: "BERT goes to EPFL: MCQ prediction with a Muppet twist!"

## Models
We present four model architectures to generate embeddings from past interactions (MLP with mastery features, LSTM with QA pairs, LernnaviBERT with QA pairs, Mistral-7B Instruct with QA pairs) and two integration strategies for Student Answer Prediction with interaction embeddings (MCQStudentBertSum and MCQStudentBertCat)

You can load a model and compute predictions (inference) with the following code snippet:
```python
import torch
import pandas as pd
from transformers import AutoModelForCausalLM, AutoModel, AutoTokenizer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
token = my_hf_token

# load Mistral 7B Instruct to be used as the embedding model
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1", token=token)
model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1", torch_dtype=torch.float16, token=token).to(device)

# load MCQStudentBert
model_bert = AutoModel.from_pretrained("epfl-ml4ed/MCQStudentBertCat", trust_remote_code=True, token=token).to(device)
tokenizer_bert = AutoTokenizer.from_pretrained("dbmdz/bert-base-german-uncased")

with torch.no_grad():
    # create interactions list and use them to create the student embedding
    interactions = pd.DataFrame([
        {"question": question_text, "choice": student_answer},
        ...
    ])
    joined_interactions = f"{tokenizer.sep_token}".join(interactions.apply(lambda x: f"Q: {x['question']}{tokenizer.sep_token}A: {x['choice']}", axis=1).values)

    embeddings = model(
        **tokenizer(joined_interactions, return_tensors="pt", truncation=True, max_length=4096).to(device),
        output_hidden_states=True
    ).hidden_states[-1].squeeze(0).mean(0)

    # use MCQStudentBert for Student Answer Forecasting
    output = torch.nn.functional.sigmoid(
        model_bert(
            tokenizer_bert(last_question, return_tensors="pt").input_ids.to(device),
            embeddings.to(torch.float32)
        ).cpu()
    ).item() > 0.5

    print(output)
```

## Scripts
We extensively evaluate our models on a large data set including a comprehensive set of student interactions, question contexts, and answer choices from a real-world ITS with more than 10000 students. With our analyses, we target the following two research questions, addressed through experiments in this repository:

**(RQ1)** How can we design a performant embedding for student interactions in German?
**(RQ2)** How can we integrate these student interaction embeddings to improve the performance of an answer forecasting model?

Running the code:
- `data` is the folder in which the Lernnavi data should be placed in the format:

```
data
|-- original
|   `-- data
|       |-- documents.csv.gz
|       |-- events.csv.gz
|       |-- topics_translated.csv
|       |-- transactions.csv.gz
|       `-- users.csv.gz
```

- the folder `src/exploration` contains the initial exploration (analysis on raw data) and some initial data extraction for the question/answer pairs
- the files in `src/feature_engineering` contain the data processing and feature engineering for the student's data (masteries extraction, mastery-question time matching, ...)
- the files in `src/student_embeddings` contain all the code required for the embedding creation (`embedding_creation.ipynb`), the autoencoder training (`autoencoder.ipynb`, `train_lstm.py`, and `autoencoder_lstm.ipynb`),  and the embedding visualization (`visualization.ipynb`)
- the file `preprocessing.py` is used to adjust the data obtained from `src/feature_engineering` notebooks for the use in the `autoencoder` notebooks (e.g. train/val/test split, qna expansion)
- the files `Answer_prediction_STEP_1_Language_Modelling.ipynb`, `Answer_prediction_STEP_2_Correct_MCQs_Training - *.ipynb`, and `Answer_prediction_STEP_3_Student_MCQs_Training - *.ipynb` contains the code and illustration for the creation of LernnaviBERT, training on MCQ prediction and finetuning on student-based MCQ forecasting, respectively

## Contributing 

This code is provided for educational purposes and aims to facilitate reproduction of our results, and further research 
in this direction. We have done our best to document, refactor, and test the code before publication.

If you find any bugs or would like to contribute new models, training protocols, etc, please let us know. Feel free to file issues and pull requests on the repo and we will address them as we can.

## Citations
If you find this code useful in your work, please cite our paper:

```
Gado, E., Martorella, T., Zunino, L., Mejia-Domenzain, P., Swamy, V., Frej, J., Käser, T. (2024). 
Student Answer Forecasting: Transformer-Driven Answer Choice Prediction for Language Learning. 
In: Proceedings of the Conference on Educational Data Mining (EDM 2024). 
```

## License
This code is free software: you can redistribute it and/or modify it under the terms of the [MIT License](LICENSE).

This software is distributed in the hope that it will be useful, but without any warranty; without even the implied warranty of merchantability or fitness for a particular purpose. See the [MIT License](LICENSE) for details.
