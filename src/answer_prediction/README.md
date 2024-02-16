This folder contains the code which is necessary in order to run the answer prediction pipeline.

The answer prediction pipeline is composed of three steps, each step is completed in a separate notebook available in the main folder of the repository:
- STEP 1: Answer_prediction_STEP_1_Language_Modelling.ipynb
- STEP 2: Answer_prediction_STEP_2_Correct_MCQs_Training.ipynb
- STEP 3: Answer_prediction_STEP_3_Student_MCQs_Training.ipynb
The notebooks have been run on Google Colab and are self-contained, in the sense that they install all the necessary packages in order to properly run. Since they are computationally very intensive, a GPU is required, and a T4 GPU is probably not sufficient (or extremely slow).

Additional files needed to run the code:
- a pkl obtained after running an `autoencoder` file or the `embedding creation` file
- `all_data_qna_expanded.pkl` and `MULTIPLE_CHOICE_german.pkl` (obtained after running the steps 1 to 4 described in the main README file)
These files need to be placed in the appropriate data folder as detailed in the individual notebooks.

All the models obtained as the output of the notebooks are available on HuggingFace Hub.