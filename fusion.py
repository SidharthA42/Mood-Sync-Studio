import torch
import torch.nn as nn
import numpy as np
from utils import emotion_to_sentiment, label_order_image, label_order_text
import os

# ---------- Rule-based fusion ----------
def rule_based_fusion(vis_emotion, txt_sentiment):
    mapped_vis = emotion_to_sentiment.get(vis_emotion, "neutral")
    mismatch = mapped_vis != txt_sentiment
    result = "MISMATCH" if mismatch else "ALIGNED"
    return result, mapped_vis

# ---------- Learned fusion (optional) ----------
class LearnedFusion(nn.Module):
    def __init__(self, input_size=10, hidden_size=8):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 2)  # aligned vs mismatch
        )

    def forward(self, vis_probs, txt_probs):
        # vis_probs: list len 7, txt_probs: list len 3
        x = torch.cat([torch.tensor(vis_probs), torch.tensor(txt_probs)], dim=-1).unsqueeze(0)
        return self.fc(x)

def load_learned_model():
    model = LearnedFusion()
    # If a pre-trained weight file exists, load it, else train on synthetic data
    if os.path.exists("learned_fusion.pth"):
        model.load_state_dict(torch.load("learned_fusion.pth", map_location=torch.device('cpu')))
    else:
        # Train a quick synthetic model (for demo)
        train_synthetic_model(model)
        torch.save(model.state_dict(), "learned_fusion.pth")
    model.eval()
    return model

def train_synthetic_model(model):
    # Generate simple synthetic data: 250 aligned, 250 mismatched
    X, y = [], []
    for _ in range(250):
        # Aligned: same sentiment
        vis = np.zeros(7)
        vis[np.random.randint(0,7)] = np.random.uniform(0.6, 1.0)
        txt = np.zeros(3)
        sent_idx = np.random.randint(0,3)
        txt[sent_idx] = np.random.uniform(0.6, 1.0)
        X.append(np.concatenate([vis, txt]))
        y.append(0)  # aligned

    for _ in range(250):
        vis = np.zeros(7)
        vis[np.random.randint(0,7)] = np.random.uniform(0.6, 1.0)
        txt = np.zeros(3)
        sent_idx = np.random.randint(0,3)
        txt[sent_idx] = np.random.uniform(0.6, 1.0)
        # mismatch: map vis emotion to opposite sentiment
        # but easier: just assign random text sentiment different from vis mapping
        X.append(np.concatenate([vis, txt]))
        y.append(1)  # mismatch

    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.long)
    dataset = torch.utils.data.TensorDataset(X, y)
    loader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    ce = nn.CrossEntropyLoss()
    for epoch in range(20):
        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            out = model(batch_x[:,:7], batch_x[:,7:])
            loss = ce(out.squeeze(0), batch_y)
            loss.backward()
            optimizer.step()