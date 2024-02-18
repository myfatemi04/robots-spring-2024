import torch
import torch.nn as nn
import torch.utils.data
import tqdm

"""
Here, the goal is to create a generative model
for sequences.

To train such a sequence autoencoder, maybe we could
do contrastive learning and train separate encoders
and decoders over the embeddings. Tbh that's the
most obvious way to train it...

Could use an LSTM, could use a transformer. Transformers
tend to have more stable training I think.

Sampling frequency will be fixed, such that token 1 is always for 0s,
token 5 is always for 0.5s, etc. (Sampling rate may differ from this
example in the actual model).
"""

"""
LSTM parameters:
    input_size: The number of expected features in the input `x`
    hidden_size: The number of features in the hidden state `h`
    num_layers: Number of recurrent layers. E.g., setting ``num_layers=2``
        would mean stacking two LSTMs together to form a `stacked LSTM`,
        with the second LSTM taking in outputs of the first LSTM and
        computing the final results. Default: 1
    bias: If ``False``, then the layer does not use bias weights `b_ih` and `b_hh`.
        Default: ``True``
    batch_first: If ``True``, then the input and output tensors are provided
        as `(batch, seq, feature)` instead of `(seq, batch, feature)`.
        Note that this does not apply to hidden or cell states. See the
        Inputs/Outputs sections below for details.  Default: ``False``
    dropout: If non-zero, introduces a `Dropout` layer on the outputs of each
        LSTM layer except the last layer, with dropout probability equal to
        :attr:`dropout`. Default: 0
    bidirectional: If ``True``, becomes a bidirectional LSTM. Default: ``False``
    proj_size: If ``> 0``, will use LSTM with projections of corresponding size. Default: 0
"""

# Train the encoder with contrastive learning.
class TrajectoryEncoder(nn.Module):
    def __init__(self, d_model=32, nhead=2, max_sequence_length=100):
        super().__init__()

        self.max_sequence_length = max_sequence_length
        self.temporal_embeddings = nn.Embedding(max_sequence_length + 1, d_model)
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=d_model * 4,
                dropout=0.1,
                batch_first=True,
            ),
            num_layers=2,
        )
        self.project_coordinates = nn.Linear(3, d_model)

    def forward(self, sequence: torch.Tensor):
        # Assume no batch is passed in.
        # Add temporal positional embeddings.
        # Append a special `embedding` token for readout.
        sequence = sequence + self.temporal_embeddings(torch.arange(sequence.shape[0]).cuda())
        embedding_token = self.temporal_embeddings(torch.tensor([self.max_sequence_length]).cuda())
        encoded_sequence = self.encoder.forward(src=torch.cat((sequence, embedding_token), dim=0))
        embedding = encoded_sequence[-1]
        return embedding

# Maybe LSTM at some point? But i don't want to think about vanishing gradient and gradient explosion :(
# lstm = nn.LSTM(input_size=64, hidden_size=64, num_layers=4, bias=True, batch_first=True, dropout=0, bidirectional=False)

import pickle

with open("dataset.pkl", "rb") as f:
    dataset = pickle.load(f)

# Training Loop
model = TrajectoryEncoder().cuda()
optim = torch.optim.Adam(model.parameters(), lr=1e-3)
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=(len(dataset) // 32) * 10, eta_min=1e-5)

for epoch in range(10):
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=lambda x: x)
    for batch in (pbar := tqdm.tqdm(dataloader, total=len(dataset), desc='Training epoch ' + str(epoch + 1))):
        embeddings = []
        for (times, X, Y, Z) in batch:
            X = torch.tensor(X)
            Y = torch.tensor(Y)
            Z = torch.tensor(Z)
            xyz = torch.stack((X, Y, Z), axis=1).cuda().float()
            embedding = model.forward(model.project_coordinates(xyz))
            embeddings.append(embedding)

        embeddings = torch.stack(embeddings, dim=0)
        embeddings = embeddings / torch.linalg.norm(embeddings, dim=-1, keepdim=True)
        scores = embeddings @ embeddings.T @ embeddings

        loss = torch.nn.functional.cross_entropy(scores, torch.arange(len(batch)).cuda())
        optim.zero_grad()
        loss.backward()
        optim.step()
        lr_scheduler.step()

        pbar.update(len(batch))
        pbar.set_postfix({'loss': loss.item()})
    pbar.close()
