# Student Answer Forecasting: Transformer-Driven Answer Choice Prediction for Language Learning
## *alternative title*: "BERT goes to EPFL: MCQ prediction with a Muppet twist!"

<br/>
<br/>

The repository is structured as follows:
- data is the folder in which should be placed the original data and all the subsequent obtained data. Can be downloaded from [here](https://drive.google.com/drive/folders/14PoILQG5sK7tSWMJtBfNUH_3UkXzR_dD?usp=sharing). The link has restricted access for data confidentiality reasons.
- the folder `src/exploration` contains the initial exploration (analysis on raw data) and some initial data extraction for the question/answer pairs
- the files in `src/feature_engineering` contain the data processing and feature engineering for the student's data (masteries extraction, mastery-question time matching, ...)
- the files in `src/student_embeddings` contain all the code required for the embedding creation (`embedding_creation.ipynb`), the autoencoder training (`autoencoder.ipynb`, `train_lstm.py`, and `autoencoder_lstm.ipynb`),  and the embedding visualization (`visualization.ipynb`)
- the file `preprocessing.py` is used to adjust the data obtained from `src/feature_engineering` notebooks for the use in the `autoencoder` notebooks (e.g. train/val/test split, qna expansion)
- the files `Answer_prediction_STEP_1_Language_Modelling.ipynb`, `Answer_prediction_STEP_2_Correct_MCQs_Training - *.ipynb`, and `Answer_prediction_STEP_3_Student_MCQs_Training - *.ipynb` contains the code and illustration for the creation of LernnaviBERT, training on MCQ prediction and finetuning on student-based MCQ forecasting, respectively

To recreate the results and the data from scratch, the following steps should be followed:
1. Place the original data in the `data/original/data` folder
2. Run the `src/exploration/exploration.ipynb` notebook
3. Run the `src/feature_engineering/feature_engineering.ipynb` and `src/feature_engineering/feature_engineering_2.ipynb` notebooks
4. Run the `src/preprocessing.py` file
5. Run an `autoencoder` notebook (takes several hours)
6. (OPTIONAL): Run `src/answer_prediction/Answer_prediction_STEP_1` and `STEP_2` to perform domain adaptation and correct-answer finetuning.
7. Run the `Answer_prediction_STEP_3_Student_MCQs_Training.ipynb` notebook for the student-answer training (takes several hours)

<br/>
<br/>

## **Team ExtraPizzaFostersLearning 🍕** 
***Luca Zunino***, ***Elena Gado***, ***Tommaso Martorella***, ***Paola Mejia***