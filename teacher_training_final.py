import os
import numpy as np
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from nilearn import datasets
from nilearn.connectome import ConnectivityMeasure
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
from torch_geometric.nn import GATConv
from sklearn.model_selection import train_test_split

# Step 1: Fetch AAL Atlas Data
def fetch_aal_data():
    atlas_aal = datasets.fetch_atlas_aal(version='SPM12')
    atlas_labels = atlas_aal['labels']

    data = datasets.fetch_abide_pcp(
        pipeline="cpac", band_pass_filtering=True, global_signal_regression=False, derivatives="rois_aal"
    )

    phenotypic_info = data['phenotypic']
    time_series_aal = data['rois_aal']

    labels = []
    for subject_info in phenotypic_info.itertuples():
        diagnostic_group = subject_info.DX_GROUP
        if diagnostic_group == 1:
            labels.append(0)  # Control
        elif diagnostic_group == 2:
            labels.append(1)  # ASD
        else:
            labels.append(-1)  # Unknown

    return time_series_aal, labels, atlas_labels

# Step 2: Create Graph Data
def create_graph(time_series, labels):
    conn_measure = ConnectivityMeasure(kind='correlation')
    graphs = []
    valid_labels = []

    for ts, label in zip(time_series, labels):
        if label != -1:  # Skip unknown labels
            corr_matrix = conn_measure.fit_transform([ts])[0]  # Compute correlation matrix
            graphs.append(corr_matrix)
            valid_labels.append(label)

    return graphs, valid_labels

# Helper: Create edge index from adjacency matrix
def adjacency_to_edge_index(adjacency_matrix):
    rows, cols = np.where(adjacency_matrix > 0)
    edge_index = torch.tensor(np.array([rows, cols]), dtype=torch.long)
    return edge_index

class GraphTransformerTeacher(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads=8):
        super(GraphTransformerTeacher, self).__init__()
        self.conv1 = GATConv(in_channels, hidden_channels, heads=heads, concat=True)
        self.conv2 = GATConv(hidden_channels * heads, out_channels, heads=heads, concat=False)

    def forward(self, x, edge_index):
        x, attn_weights1 = self.conv1(x, edge_index, return_attention_weights=True)
        x = F.relu(x)
        x, attn_weights2 = self.conv2(x, edge_index, return_attention_weights=True)
        return x, attn_weights1[1], attn_weights2[1]

# Step 5: Simulate Teacher with Transformer
def train_teacher_model(graphs, labels, device):
    attention_matrices = []  # Store attention matrices for each graph
    predictions = []

    # Train-Test Split
    train_graphs, test_graphs, train_labels, test_labels = train_test_split(graphs, labels, test_size=0.2, random_state=42)

    in_channels = train_graphs[0].shape[0]
    hidden_channels = 128
    out_channels = 2  # Binary classification (Control/ASD)

    teacher_model = GraphTransformerTeacher(in_channels, hidden_channels, out_channels).to(device)

    optimizer = optim.Adam(teacher_model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    # Training Loop
    for epoch in range(10):
        teacher_model.train()
        epoch_loss = 0.0

        for graph, label in zip(train_graphs, train_labels):
            node_features = torch.tensor(graph, dtype=torch.float32).to(device)
            edge_index = adjacency_to_edge_index(graph).to(device)
            label_tensor = torch.tensor([label], dtype=torch.long).to(device)

            optimizer.zero_grad()

            outputs, attn1, attn2 = teacher_model(node_features, edge_index)
            graph_output = outputs.mean(dim=0).unsqueeze(0)  # Aggregate over nodes

            loss = criterion(graph_output, label_tensor)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        
        print(f"Epoch {epoch + 1}, Loss: {epoch_loss:.4f}")
        
    # Evaluation
    teacher_model.eval()
    with torch.no_grad():
        for graph, label in zip(test_graphs, test_labels):
            node_features = torch.tensor(graph, dtype=torch.float32).to(device)
            edge_index = adjacency_to_edge_index(graph).to(device)

            outputs, attn1, attn2 = teacher_model(node_features, edge_index)
            graph_output = outputs.mean(dim=0).unsqueeze(0)
            
            # Store attention matrices
            attention_matrices.append((attn1.cpu().numpy(), attn2.cpu().numpy()))

            pred = torch.argmax(graph_output, dim=1).cpu().item()
            predictions.append(pred)

    # Calculate accuracy
    teacher_accuracy = accuracy_score(test_labels, predictions)
    print(f"Teacher Model Accuracy: {teacher_accuracy * 100:.2f}%")

    return teacher_accuracy, attention_matrices

# Main script
if __name__ == "__main__":
    # Set the device for training (GPU or CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load data
    time_series, labels, atlas_labels = fetch_aal_data()
    graphs, valid_labels = create_graph(time_series, labels)

    # Train Teacher Model
    teacher_accuracy, attention_matrices = train_teacher_model(graphs, valid_labels, device)

    # Save the attention matrices for further analysis
    with open("attention_matrices.pkl", "wb") as f:
        pickle.dump(attention_matrices, f)

    print(f"Teacher Model Accuracy: {teacher_accuracy * 100:.2f}%")
