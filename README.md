# Answer Forecasting: ML-Driven Predictions in Lernnavi
## *alternative title*: BERT goes to EPFL: MCQ prediction with a Muppet twist!

<br/>
<br/>

The repository is structured as follows:
- data is the folder in which should be placed the original data and all the subsequent obtained data. Can be downloaded from [here](https://drive.google.com/drive/folders/14PoILQG5sK7tSWMJtBfNUH_3UkXzR_dD?usp=sharing). The link has restricted access for data confidentiality reasons.
- the folder `exploration` contains the initial exploration (analysis on raw data) and some initial data extraction for the question/answer pairs
- the files `feature_engineering.ipynb` and `feature_engineering_2.ipynb` contain the feature engineering for the student's data (masteries extraction, mastery-question time matching, ...)
- the files `autoencoder.ipynb` and `autoencoder_lstm.ipynb` contain the different models we used to create the student's embeddings
- the file `preprocessing.py` is adjust the data obtained from `feature_engineering` notebooks for the use in the `autoencoder` notebooks (e.g. train/val/test split)
- the files `Answer_prediction_STEP_1_Language_Modelling.ipynb`, `Answer_prediction_STEP_2_Correct_MCQs_Training.ipynb`, and `Answer_prediction_STEP_3_Student_MCQs_Training.ipynb` contains the code and illustration for the creation of LernnaviBERT, training on MCQ prediction and finetuning on student-based MCQ forecasting, respectively

To recreate the results and the data from scratch, the following steps should be followed:
1. Place the original data in the `data\original\data` folder
2. Run the `exploration\exploration.ipynb` notebook
3. Run the `feature_engineering.ipynb` and `feature_engineering_2.ipynb` notebooks (takes quite a while)
4. Run the `autoencoder_lstm.ipynb` notebook (takes several hours)
5. Run the `Answer_prediction_STEP_3_Student_MCQs_Training.ipynb` notebook (takes several hours)


## **Team ExtraPizzaFostersLearning**
***Luca Zunino***, ***Elena Gado***, ***Tommaso Martorella***