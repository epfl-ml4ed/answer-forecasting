import torch
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
from statistics import mean
from tqdm.auto import tqdm
from pathlib import Path
from torch.nn import MSELoss
import torch.nn as nn


BASE_DATA_DIR = "../../data"
QNA_DATA_DIR = f"{BASE_DATA_DIR}/lernnavi/qna"
ORIGINAL_DATA_DIR = f"{BASE_DATA_DIR}/original/data"


# lstm autoencoder from https://github.com/shobrook/sequitur

class Encoder(nn.Module):
    def __init__(self, input_dim, out_dim, h_dims, h_activ, out_activ):
        super(Encoder, self).__init__()

        layer_dims = [input_dim] + h_dims + [out_dim]
        self.num_layers = len(layer_dims) - 1
        self.layers = nn.ModuleList()
        for index in range(self.num_layers):
            layer = nn.LSTM(
                input_size=layer_dims[index],
                hidden_size=layer_dims[index + 1],
                num_layers=1,
                batch_first=True,
            )
            self.layers.append(layer)

        self.h_activ, self.out_activ = h_activ, out_activ

    def forward(self, x):
        x = x.unsqueeze(0)
        for index, layer in enumerate(self.layers):
            x, (h_n, c_n) = layer(x)

            if self.h_activ and index < self.num_layers - 1:
                x = self.h_activ(x)
            elif self.out_activ and index == self.num_layers - 1:
                return self.out_activ(h_n).squeeze()

        return h_n.squeeze()


class Decoder(nn.Module):
    def __init__(self, input_dim, out_dim, h_dims, h_activ):
        super(Decoder, self).__init__()

        layer_dims = [input_dim] + h_dims + [h_dims[-1]]
        self.num_layers = len(layer_dims) - 1
        self.layers = nn.ModuleList()
        for index in range(self.num_layers):
            layer = nn.LSTM(
                input_size=layer_dims[index],
                hidden_size=layer_dims[index + 1],
                num_layers=1,
                batch_first=True,
            )
            self.layers.append(layer)

        self.h_activ = h_activ
        self.dense_matrix = nn.Parameter(
            torch.rand((layer_dims[-1], out_dim), dtype=torch.float), requires_grad=True
        )

    def forward(self, x, seq_len):
        x = x.repeat(seq_len, 1).unsqueeze(0)
        for index, layer in enumerate(self.layers):
            x, (h_n, c_n) = layer(x)

            if self.h_activ and index < self.num_layers - 1:
                x = self.h_activ(x)

        return torch.mm(x.squeeze(0), self.dense_matrix)


class LSTM_AE(nn.Module):
    def __init__(
        self,
        input_dim,
        encoding_dim,
        h_dims=[],
        h_activ=nn.Sigmoid(),
        out_activ=nn.Tanh(),
    ):
        super(LSTM_AE, self).__init__()

        self.encoder = Encoder(input_dim, encoding_dim, h_dims, h_activ, out_activ)
        self.decoder = Decoder(encoding_dim, input_dim, h_dims[::-1], h_activ)

    def forward(self, x):
        seq_len = x.shape[0]
        x = self.encoder(x)
        x = self.decoder(x, seq_len)

        return x


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def instantiate_model(model, train_set, encoding_dim, **kwargs):
    if model.__name__ in ("LINEAR_AE", "LSTM_AE"):
        return model(train_set[-1].shape[-1], encoding_dim, **kwargs)
    elif model.__name__ == "CONV_LSTM_AE":
        if len(train_set[-1].shape) == 3:  # 2D elements
            return model(train_set[-1].shape[-2:], encoding_dim, **kwargs)
        elif len(train_set[-1].shape) == 4:  # 3D elements
            return model(train_set[-1].shape[-3:], encoding_dim, **kwargs)


@torch.no_grad()
def validate_model(model, val_set, epoch, criterion, device):
    model.eval()
    
    losses = []
    for x in tqdm(val_set, desc=f"Val Epoch {epoch: 3d}"):
        x = x.to(device)
        x_prime = model(x)
        
        loss = criterion(x_prime, x)
        losses.append(loss.item())
        
    return mean(losses)
        

def train_model(
    model, train_set, val_set, verbose, lr, epochs, denoise, clip_value, device=None, save_path=Path("./checkpoints"), skip_epochs=0
):
    if device is None:
        device = get_device()
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = MSELoss(reduction="sum")

    mean_losses = []
    val_losses = []
    for epoch in range(1, epochs + 1):
        if epoch <= skip_epochs:
            continue

        model.train()

        # # Reduces learning rate every 50 epochs
        # if not epoch % 50:
        #     for param_group in optimizer.param_groups:
        #         param_group["lr"] = lr * (0.993 ** epoch)

        losses = []

        for x in tqdm(train_set, desc=f"Train Epoch {epoch: 3d}"):
            x = x.to(device, non_blocking=True)
            
            optimizer.zero_grad()

            # Forward pass
            x_prime = model(x)

            loss = criterion(x_prime, x)

            # Backward pass
            loss.backward()
           

            # Gradient clipping on norm
            if clip_value is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)

            optimizer.step()

            losses.append(loss.item())

        mean_loss = mean(losses)
        mean_losses.append(mean_loss)
        
        torch.save(model.state_dict(), str(save_path / f"model_{epoch:03d}.pt"))

        val_loss = validate_model(model, val_set, epoch, criterion, device)
        val_losses.append(val_loss)
        
        if verbose:
            print(f"Epoch: {epoch}, Train loss: {mean_loss}, Validation Loss: {val_loss}")

    return mean_losses, val_losses


@torch.no_grad()
def get_encodings(model, train_set, device=None):
    if device is None:
        device = get_device()
    model.eval()
    encodings = [model.encoder(x.to(device)) for x in tqdm(train_set)]
    return encodings


def quick_train(
    model,
    train_set,
    val_set,
    encoding_dim,
    verbose=False,
    lr=1e-3,
    epochs=100,
    clip_value=1,
    denoise=False,
    device=None,
    save_path=Path("../../checkpoints"),
    chekpoint_path=None,
    **kwargs,
):
    model = instantiate_model(model, train_set, encoding_dim, **kwargs)

    skip_epochs = 0
    if chekpoint_path:
        model.load_state_dict(torch.load(chekpoint_path))
        skip_epochs = int(chekpoint_path.stem.split("_")[-1])


    save_path.mkdir(parents=True, exist_ok=True)
    train_losses, val_losses = train_model(
        model, train_set, val_set, verbose, lr, epochs, denoise, clip_value, device, save_path, skip_epochs=skip_epochs
    )

    return model.encoder, model.decoder, train_losses, val_losses


class MCQDataset(torch.utils.data.Dataset):
    """
        
    """

    _nlp_model = None

    @property
    def nlp_model(self):
        if MCQDataset._nlp_model:
            return MCQDataset._nlp_model
        
        from sentence_transformers import SentenceTransformer
        MCQDataset._nlp_model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
        return MCQDataset._nlp_model


    def __init__(self, datapath, seq_len=5, device=get_device()):
        self.datapath = datapath
        self.seq_len = seq_len
        self.device = device

        import os
        self.df = pd.read_pickle(self.datapath)
            
        # preprocess topic data
        self.df['question_embedding'] = self._create_q_embeddings()
        self.df['answer_embedding']= self._create_a_embeddings()

    def _create_q_embeddings(self):
        # create embeddings for each topic
        embeddings = self.nlp_model.to(self.device).encode(self.df["question"], show_progress_bar=True, batch_size=2048)
        return list(map(lambda x: np.squeeze(x), np.split(embeddings, embeddings.shape[0])))
    def _create_a_embeddings(self):
        # create embeddings for each topic
        embeddings = self.nlp_model.to(self.device).encode(self.df["choice"], show_progress_bar=True, batch_size=2048)
        return list(map(lambda x: np.squeeze(x), np.split(embeddings, embeddings.shape[0])))
       
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if idx < 0:
            idx = len(self.df) + idx

        df2 = self.df[self.df["user_id"] == self.df.iloc[idx]["user_id"]].reset_index()
        df2 = df2.sort_values(by="start_time").reset_index(drop=True)
        indx = df2[df2["index"] == idx].index[0]

        
        if indx >= self.seq_len:
            seq_before = df2.iloc[indx-self.seq_len+1 : indx+1]
        else:
            seq_before = df2.iloc[0: indx+1]


        data = torch.stack(
            seq_before.apply(lambda x: np.concatenate((x['question_embedding'], x['answer_embedding'])), axis=1)
              .apply(lambda x: torch.tensor(x, dtype=torch.float32))
              .tolist()
        )

        return data
    

if __name__ == "__main__":
    import sys

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_dataset = MCQDataset(f"{QNA_DATA_DIR}/train/qna_expanded.pkl")
    val_dataset = MCQDataset(f"{QNA_DATA_DIR}/validation/qna_expanded.pkl")

    torch.cuda.empty_cache()

    trains = []
    vals = []
    seq_lens = [int(sys.argv[1])]
    h_dims = [int(sys.argv[2])]
    x = None

    for i in seq_lens:
        train_dataset.seq_len = i
        val_dataset.seq_len = i

        for j in h_dims:
            print("=======================================")
            print("TRAINING BEGINS")
            print(f"seq_len: {i}")
            print(f"hidden_dims: {j}")
            print("=======================================")

            save_path = Path(f"../../checkpoints/seq_len_{i}_h_dims_{j}")
            last_checkpoint = None
            # directory might be empty
            if len(list(save_path.glob("*.pt"))) > 0:
                last_checkpoint = sorted(list(save_path.glob("*.pt")))[-1]
                print(f"Resuming from checkpoint {last_checkpoint}")

            (
                encoder,
                decoder,
                train_losses,
                val_losses
            ) = quick_train(LSTM_AE, train_dataset, val_dataset, encoding_dim=384, verbose=True, epochs=100, h_dims=[384]*j, save_path=save_path, chekpoint_path=last_checkpoint)
            trains.append(train_losses)
            vals.append(val_losses)

    results = pd.DataFrame({
        "train": trains,
        "val": vals,
        "train_min": np.min(trains, axis=1),
        "val_min": np.min(vals, axis=1),
        "seq_len": seq_lens * len(h_dims),
        "h_dims": [[i] * len(seq_lens) for i in h_dims]
    })

    results.to_pickle(f"../../checkpoints/lstm_results_seq_len_{'_'.join([str(x) for x in seq_lens])}_h_dims_{'_'.join([str(x) for x in h_dims])}{f'_part{x}' if x is not None else ''}.pkl")
