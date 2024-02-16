from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, matthews_corrcoef
import torch
from tqdm import tqdm
from copy import deepcopy
from transformers import get_constant_schedule_with_warmup
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

random_seed = 0


def generate_synced_qna_embeddings(dataframe_path, embeddings_path, qna_path):
    data = pd.read_pickle(dataframe_path)
    embeddings = torch.load(embeddings_path)
    qna = pd.read_pickle(qna_path)

    data["embeddings"] = list(embeddings.numpy())
    pooled_embeddings = data.groupby(["question_index", "start_time", "user_id"])["embeddings"].mean()
    return qna.merge(pooled_embeddings, on=["user_id", "start_time"])


# Function to calculate metrics
def metrics(y_pred, y_true):
    # Accuracy
    acc_sc = accuracy_score(y_true, y_pred)
    # F1 score
    f1_0, f1_1 = f1_score(y_true, y_pred, average = None)
    # Confusion matrix
    con_mat = confusion_matrix(y_true, y_pred)
    # Matthews correlation coefficient
    mcc = matthews_corrcoef(y_true, y_pred)
    return acc_sc, f1_0, f1_1, con_mat, mcc


# Class to build the training and validation datasets
class QuestionsDataset(torch.utils.data.Dataset):
    def __init__(self, tokenizer, dataset, student_embeddings, maximum_sentence_length=512):
        
        self.id_to_letter = {0: "A", 1: "B", 2: "C", 3: "D", 4: "E", 5: "F", 6: "G", 7: "H", 8: "I", 9: "J", 10: "K", 11: "L"}
        self.padding_id = tokenizer.pad_token_id
        self.processed_samples = []
        self.student_embeddings = student_embeddings

        # Weights for the BCEWithLogitsLoss (to balance the classes)
        weights = [0.0, 0.0]
        total_counter = 0

        # Iterate over the dataset
        for _, entry in tqdm(dataset.iterrows()):
            # Get the selected answer (in the form of a list of letters)
            selected_answers = list(entry["student_answer"])
            # Get the ids of the question (which is the same for all the answers)
            question_ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(entry["question"])[:maximum_sentence_length])
            # Iterate over each answer choice
            for i, ans in enumerate(entry["choices"]):
                # Get the ids of the answer choice
                ans_ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(ans)[:maximum_sentence_length])
                # Build the input ids for the answer choice
                input_ids = tokenizer.build_inputs_with_special_tokens(question_ids, ans_ids)
                # Get the label for the answer choice
                label = 0 if self.id_to_letter[i] not in selected_answers else 1
                # Update the weight for the label
                weights[label] += 1
                # Get the student embedding
                student_embedding = student_embeddings[entry["user_id"]]
                # Get the student features
                embedding_size = student_embedding.shape[0]
                student_feature = np.zeros(embedding_size, dtype = np.float32)
                # Append the processed sample
                self.processed_samples.append({"ids": input_ids, "label": label, "student_embedding": student_embedding, "student_features": student_feature})
                total_counter = total_counter + 1

        # Compute the weights (to balance the classes)
        weights = [total_counter / weights[0], total_counter / weights[1]]
        self.weights = weights

    def __len__(self):
        return len(self.processed_samples)
    
    def __getitem__(self, idx):
        return deepcopy(self.processed_samples[idx])

    def pad(self, inputs):
        max_len = max(list(map(len, inputs)))
        return [inp + [self.padding_id] * (max_len - len(inp)) for inp in inputs]
    
    def get_sample(self, idx):
        return {"ids": self.processed_samples[idx]["ids"], "label": self.processed_samples[idx]["label"], "student_embedding": self.processed_samples[idx]["student_embedding"], "student_features": self.processed_samples[idx]["student_features"]}
    
    def collate_batch(self, batch):
        ids = [dictionary["ids"] for dictionary in batch]
        labels = [dictionary["label"] for dictionary in batch]
        student_embeddings = [dictionary["student_embedding"] for dictionary in batch]
        student_features = [dictionary["student_features"] for dictionary in batch]
        return torch.tensor(self.pad(ids)), torch.tensor(np.array(labels), dtype=torch.float), torch.tensor(np.array(student_embeddings), dtype=torch.float), torch.tensor(np.array(student_features), dtype=torch.float)


# Class to build the training and validation datasets
class QuestionsDatasetWithSyncedEmbeddings(torch.utils.data.Dataset):
    def __init__(self, tokenizer, dataset, maximum_sentence_length=512):

        self.dataset = dataset
        self.id_to_letter = {0: "A", 1: "B", 2: "C", 3: "D", 4: "E", 5: "F", 6: "G", 7: "H", 8: "I", 9: "J", 10: "K", 11: "L"}
        self.padding_id = tokenizer.pad_token_id
        self.processed_samples = []

        # Weights for the BCEWithLogitsLoss (to balance the classes)
        weights = [0.0, 0.0]
        total_counter = 0

        # Iterate over the dataset
        for _, entry in tqdm(dataset.iterrows()):
            # Get the selected answer (in the form of a list of letters)
            selected_answers = list(entry["student_answer"])
            # Get the ids of the question (which is the same for all the answers)
            question_ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(entry["question"])[:maximum_sentence_length])
            # Iterate over each answer choice
            for i, ans in enumerate(entry["choices"]):
                # Get the ids of the answer choice
                ans_ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(ans)[:maximum_sentence_length])
                # Build the input ids for the answer choice
                input_ids = tokenizer.build_inputs_with_special_tokens(question_ids, ans_ids)
                # Get the label for the answer choice
                label = int(entry["student_answer"][i])
                # Update the weight for the label
                weights[label] += 1
                # Get the student embedding
                student_embedding = entry["embeddings"]
                # Get the student features
                embedding_size = student_embedding.shape[0]
                student_feature = np.zeros(embedding_size, dtype = np.float32)
                # Append the processed sample
                self.processed_samples.append({"ids": input_ids, "label": label, "student_embedding": student_embedding, "student_features": student_feature})
                total_counter = total_counter + 1

        # Compute the weights (to balance the classes)
        weights = [total_counter / weights[0], total_counter / weights[1]]
        self.weights = weights

    def __len__(self):
        return len(self.processed_samples)
    
    def __getitem__(self, idx):
        return deepcopy(self.processed_samples[idx])

    def pad(self, inputs):
        max_len = max(list(map(len, inputs)))
        return [inp + [self.padding_id] * (max_len - len(inp)) for inp in inputs]
    
    def get_sample(self, idx):
        return {"ids": self.processed_samples[idx]["ids"], "label": self.processed_samples[idx]["label"], "student_embedding": self.processed_samples[idx]["student_embedding"], "student_features": self.processed_samples[idx]["student_features"]}
    
    def collate_batch(self, batch):
        ids = [dictionary["ids"] for dictionary in batch]
        labels = [dictionary["label"] for dictionary in batch]
        student_embeddings = [dictionary["student_embedding"] for dictionary in batch]
        student_features = [dictionary["student_features"] for dictionary in batch]
        return torch.tensor(self.pad(ids)), torch.tensor(np.array(labels), dtype=torch.float), torch.tensor(np.array(student_embeddings), dtype=torch.float), torch.tensor(np.array(student_features), dtype=torch.float)


# Class to build the training and validation datasets (correct answers)
class QuestionsDatasetCorrect(torch.utils.data.Dataset):
    def __init__(self, tokenizer, dataset, embedding_size, maximum_sentence_length=512):
        
        self.id_to_letter = {0: "A", 1: "B", 2: "C", 3: "D", 4: "E", 5: "F", 6: "G", 7: "H", 8: "I", 9: "J", 10: "K", 11: "L"}
        self.padding_id = tokenizer.pad_token_id
        self.processed_samples = []

        # Weights for the BCEWithLogitsLoss (to balance the classes)
        weights = [0.0, 0.0]
        total_counter = 0

        # Iterate over the dataset
        for _, entry in tqdm(dataset.iterrows()):
            # Get the ids of the question (which is the same for all the answers)
            question_ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(entry["question"])[:maximum_sentence_length])
            # Iterate over each answer choice
            for i, ans in enumerate(entry["choices"]):
                # Get the ids of the answer choice
                ans_ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(ans)[:maximum_sentence_length])
                # Build the input ids for the answer choice
                input_ids = tokenizer.build_inputs_with_special_tokens(question_ids, ans_ids)
                # Get the label for the answer choice
                label = 1 if entry["correct"][i] == True else 0
                # Update the weight for the label
                weights[label] += 1
                # Get the student embedding
                student_embedding = np.zeros(embedding_size, dtype = np.float32)
                # Get the student features
                student_features = np.zeros(embedding_size, dtype = np.float32)
                # Append the processed sample
                self.processed_samples.append({"ids": input_ids, "label": label, "student_embedding": student_embedding, "student_features": student_features})
                total_counter = total_counter + 1

        # Compute the weights (to balance the classes)
        weights = [total_counter / weights[0], total_counter / weights[1]]
        self.weights = weights

    def __len__(self):
        return len(self.processed_samples)
    
    def __getitem__(self, idx):
        return deepcopy(self.processed_samples[idx])

    def pad(self, inputs):
        max_len = max(list(map(len, inputs)))
        return [inp + [self.padding_id] * (max_len - len(inp)) for inp in inputs]
    
    def get_sample(self, idx):
        return {"ids": self.processed_samples[idx]["ids"], "label": self.processed_samples[idx]["label"], "student_embedding": self.processed_samples[idx]["student_embedding"], "student_features": self.processed_samples[idx]["student_features"]}
    
    def collate_batch(self, batch):
        ids = [dictionary["ids"] for dictionary in batch]
        labels = [dictionary["label"] for dictionary in batch]
        student_embeddings = [dictionary["student_embedding"] for dictionary in batch]
        student_features = [dictionary["student_features"] for dictionary in batch]
        return torch.tensor(self.pad(ids)), torch.tensor(np.array(labels), dtype=torch.float), torch.tensor(np.array(student_embeddings), dtype=torch.float), torch.tensor(np.array(student_features), dtype=torch.float)

    
# Function to train the model
def train(model, train_data, val_data, device, epochs_num, batch_size, lr, warmup_p, max_gradient_norm, embedding_required=False, use_embedding=True):
    best_accuracy = 0
    generator = torch.Generator().manual_seed(random_seed)
    train_dataloader = torch.utils.data.DataLoader(train_data, sampler=torch.utils.data.RandomSampler(train_data, generator=generator), batch_size=batch_size, collate_fn=train_data.collate_batch)
    optimizer = torch.optim.AdamW(params=model.parameters(), lr=lr)
    # Learning rate scheduler
    scheduler = get_constant_schedule_with_warmup(optimizer=optimizer, num_warmup_steps=int(warmup_p*epochs_num*len(train_dataloader)))

    # Loss function (with weights to balance the classes)
    pos_weight_value = (train_data.weights[1] / train_data.weights[0])
    pos_weight_tensor = torch.tensor([pos_weight_value], dtype=torch.float).to(device)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor).to(device)

    model.zero_grad()
    model.train()

    for epoch in range(epochs_num):
        total_loss = 0
        train_step = 0
        for batch in tqdm(train_dataloader, desc="Training"):
            optimizer.zero_grad()
            train_step = train_step + 1
            ids, labels, student_embeddings, student_features = tuple(input_t.to(device) for input_t in batch)
            if embedding_required and use_embedding:
                output = model(ids, student_embeddings)
                logits = model.classifier(output)
            elif embedding_required and not use_embedding:
                output = model(ids, student_features)
                logits = model.classifier(output)
            else:
                output = model(ids)
                # last_hidden_state[:, 0, :] is the CLS token (effective for classification)
                logits = model.classifier(output.last_hidden_state[:, 0, :])
            loss = criterion(logits.squeeze(1), labels)
            loss.backward()
            # Clip the gradients for stability
            torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=max_gradient_norm)
            total_loss = total_loss + loss.mean().item()
            optimizer.step()
            scheduler.step()
        train_loss = total_loss / train_step

        # Evaluate the model on the validation set
        val_loss, acc_score, f1_0, f1_1, con_matrix, mcc = evaluate(model, val_data, device, batch_size, train_weights = train_data.weights, embedding_required = embedding_required)
        print(f'Epoch: {epoch}')
        print(f'Training loss: {train_loss:.3f}')
        print(f'Validation loss: {val_loss:.3f}')
        print(f'Accuracy score: {acc_score:.3f}')
        print(f'F1 score for class 0: {f1_0:.3f}')
        print(f'F1 score for class 1: {f1_1:.3f}')
        print(f'Confusion matrix:\n{con_matrix}')
        print(f'MCC: {mcc:.3f}')
        sns.set(font_scale=1.4)
        sns.heatmap(data = pd.DataFrame(con_matrix), annot=True, xticklabels=['0', '1'], yticklabels=['0', '1'], cmap="Blues")
        plt.title('Confusion Matrix')
        plt.show()


# Function to evaluate the model
def evaluate(model, val_data, device, batch_size, train_weights, embedding_required=False, use_embedding=True):
    eval_loss = 0
    eval_step = 0
    batch_labels = []
    batch_preds = []
    eval_dataloader = torch.utils.data.DataLoader(val_data, sampler=torch.utils.data.SequentialSampler(val_data), batch_size=batch_size, collate_fn=val_data.collate_batch)
    
    # Loss function (with weights to balance the classes)
    pos_weight_value = (train_weights[1] / train_weights[0])
    pos_weight_tensor = torch.tensor([pos_weight_value], dtype=torch.float).to(device)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor).to(device)

    model.eval()

    for batch in tqdm(eval_dataloader, desc="Evaluation"):
        eval_step = eval_step + 1
        with torch.no_grad():
            ids, labels, student_embeddings, student_features = tuple(input_t.to(device) for input_t in batch)
            if embedding_required and use_embedding:
                output = model(ids, student_embeddings)
                logits = model.classifier(output)
            elif embedding_required and not use_embedding:
                output = model(ids, student_features)
                logits = model.classifier(output)
            else:
                output = model(ids)
                # last_hidden_state[:, 0, :] is the CLS token (effective for classification)
                logits = model.classifier(output.last_hidden_state[:, 0, :])
            loss = criterion(logits.squeeze(1), labels)
            eval_loss = eval_loss + loss.mean().item()
            batch_labels.append(labels.detach().cpu().numpy())
            batch_preds.append((logits.detach().cpu().numpy() > 0.5).astype(int))
    pred_labels = np.concatenate(batch_preds)
    
    eval_loss = eval_loss / eval_step
    true_labels = list(np.concatenate(batch_labels))
    acc_score, f1_0, f1_1, con_matrix, mcc = metrics(pred_labels, true_labels)
    return eval_loss, acc_score, f1_0, f1_1, con_matrix, mcc