import os
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from nilearn import datasets
from nilearn.connectome import ConnectivityMeasure

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

# Step 3: Prepare Data for GraTeD-MLP
def prepare_data(graphs):
    features = [np.mean(graph, axis=1) for graph in graphs]  # Example: Node features as mean connectivity
    adjacency_matrices = graphs  # Adjacency matrices are the correlation matrices
    return np.array(features), np.array(adjacency_matrices)

# Step 4: Simulated Teacher Model
import torch

def simulate_teacher(graphs):
    # Load attention matrices from the provided file
    with open("attention_matrices.pkl", "rb") as f:
        attention_matrices = pickle.load(f)

    U_matrices = []
    V_matrices = []

    for attention_matrix in attention_matrices:
        # Convert the attention matrix to a PyTorch tensor and move it to GPU
        attention_tensor = torch.tensor(np.array(attention_matrix), device="cuda", dtype=torch.float32)
        
        # Perform Singular Value Decomposition (SVD) on GPU
        U, _, V = torch.linalg.svd(attention_tensor, full_matrices=False)
        U_matrices.append(U.cpu().numpy())  # Move back to CPU and convert to numpy
        V_matrices.append(V.cpu().numpy())  # Move back to CPU and convert to numpy

    return U_matrices, V_matrices



# Step 5: Define GraTeD-MLP Architecture
class GraTeDMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GraTeDMLP, self).__init__()
        # Feature Branch
        self.feature_branch = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU()
        )
        # Structure Branch for U
        self.structure_branch_U = nn.Sequential(
            nn.Linear(input_dim, 128),  # Intermediate size
            nn.ReLU(),
            nn.Linear(128, 116)  # Match U_reduced size
        )
        # Structure Branch for V
        self.structure_branch_V = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 116)  # Match V_reduced size
        )
        # Gating mechanism
        self.gating_layer = nn.Linear(hidden_dim + 116 * 2, output_dim)

    def forward(self, x):
        feature_output = self.feature_branch(x)  # Shape: [batch_size, hidden_dim]
        structure_output_U = self.structure_branch_U(x)  # Shape: [batch_size, 116]
        structure_output_V = self.structure_branch_V(x)  # Shape: [batch_size, 116]

        # Concatenate outputs
        concatenated = torch.cat([feature_output, structure_output_U, structure_output_V], dim=1)
        final_output = self.gating_layer(concatenated)  # Shape: [batch_size, output_dim]
        return final_output, structure_output_U, structure_output_V

#Resizing the attention matrix to match the suitable size
def resize_attention_matrices(matrices, target_batch_size, target_dim):
    resized = []
    for matrix in matrices:
        # Flatten the matrix along nodes and heads
        flattened_matrix = matrix.reshape(-1, matrix.shape[-1])  # Shape: [num_nodes * num_heads, hidden_dim]

        # Reduce to target_dim by averaging and pad/truncate to target_dim
        reduced_matrix = np.mean(flattened_matrix, axis=0)  # Shape: [hidden_dim]
        if len(reduced_matrix) < target_dim:
            # Pad if smaller
            reduced_matrix = np.pad(reduced_matrix, (0, target_dim - len(reduced_matrix)), mode='constant')
        else:
            # Truncate if larger
            reduced_matrix = reduced_matrix[:target_dim]

        resized.append(reduced_matrix)

    # Ensure batch size consistency
    while len(resized) < target_batch_size:
        resized.append(np.zeros(target_dim))  # Pad with zeros
    resized = resized[:target_batch_size]  # Truncate if too large

    return np.array(resized)


# Step 6: Train GraTeD-MLP with Distillation
def train_grated_mlp(features, labels, U_matrices, V_matrices):
    target_batch_size = features.shape[0]  # Match batch size of input features
    target_dim = 116  # Match model's structure branch output

    # Resize U and V matrices
    U_resized = resize_attention_matrices(U_matrices, target_batch_size, target_dim)
    V_resized = resize_attention_matrices(V_matrices, target_batch_size, target_dim)

    # Convert resized matrices to tensors
    U_tensor = torch.tensor(np.array(U_resized), dtype=torch.float32, device="cuda")
    V_tensor = torch.tensor(np.array(V_resized), dtype=torch.float32, device="cuda")

    # Prepare features and labels
    features_tensor = torch.tensor(features, dtype=torch.float32, device="cuda")
    labels_tensor = torch.tensor(labels, dtype=torch.long, device="cuda")

    # Initialize the GraTeD-MLP model
    input_dim = features.shape[1]
    hidden_dim = 128
    output_dim = 2
    model = GraTeDMLP(input_dim, hidden_dim, output_dim).to("cuda")
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(2000):
        model.train()
        optimizer.zero_grad()

        outputs, structure_output_U, structure_output_V = model(features_tensor)

        # Classification Loss
        classification_loss = criterion(outputs, labels_tensor)

        # Structural Losses
        U_loss = nn.MSELoss()(structure_output_U, U_tensor)
        V_loss = nn.MSELoss()(structure_output_V, V_tensor)

        # Total Loss
        total_loss = classification_loss + 0.1 * (U_loss + V_loss)
        total_loss.backward()
        optimizer.step()
        print(f"Epoch {epoch + 1}, Loss: {total_loss.item():.4f}")


# Main Script
if __name__ == "__main__":
    time_series, labels, atlas_labels = fetch_aal_data()
    graphs, valid_labels = create_graph(time_series, labels)
    features, adjacency_matrices = prepare_data(graphs)
    U_matrices, V_matrices = simulate_teacher(graphs)
    train_grated_mlp(features, valid_labels, U_matrices, V_matrices)
    