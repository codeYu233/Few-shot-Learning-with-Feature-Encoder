import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.io import loadmat

# Step 1: LSTM-based Classifier
class ModelLSTM(nn.Module):
    def __init__(self, num_regions, time_series_len, hidden_size, num_layers, num_classes):
        super(ModelLSTM, self).__init__()
        
        self.num_regions = num_regions
        self.time_series_len = time_series_len
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.conv1 = nn.Conv1d(in_channels=num_regions, out_channels=64, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        
        self.lstm = nn.LSTM(input_size=64, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        
        self.fc = nn.Linear(hidden_size, num_classes)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x, return_features=False):
        x = self.conv1(x)
        x = self.relu1(x)

        x = x.permute(0, 2, 1)

        h_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        out, _ = self.lstm(x, (h_0, c_0))

        features = out[:, -1, :]  # Extract features from the last hidden state

        logits = self.fc(features)
        if return_features:
            return logits, features
        else:
            return self.softmax(logits)

# Step 2: Prototypical Network
class PrototypicalNetwork(nn.Module):
    def __init__(self, feature_dim, hidden_dim):
        super(PrototypicalNetwork, self).__init__()
        self.fc1 = nn.Linear(feature_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def compute_prototypes(support_features, support_labels, n_way):
    prototypes = []
    for cls in range(n_way):
        cls_features = support_features[support_labels == cls]
        prototype = cls_features.mean(dim=0)
        prototypes.append(prototype)
    return torch.stack(prototypes)

def euclidean_dist(x, y):
    return ((x - y) ** 2).sum(dim=-1)

# Step 3: Train Classification Model
def train_classifier(model, train_loader, val_loader, device, num_epochs=50, lr=0.001):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    model.to(device)

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        val_acc = evaluate_classifier(model, val_loader, device)
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss / len(train_loader):.4f}, Validation Accuracy: {val_acc:.4f}")

def evaluate_classifier(model, val_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total

# Step 4: Extract Features
def extract_features(model, loader, device):
    model.eval()
    features = []
    labels = []
    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device)
            _, feature = model(inputs, return_features=True)
            features.append(feature.cpu())
            labels.append(targets)
    return torch.cat(features), torch.cat(labels)

# Step 5: Train Prototypical Network
def train_prototypical_network(model, features, labels, n_way, k_shot, q_queries, num_tasks, lr, device):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    model.to(device)

    for task in range(num_tasks):
        classes = torch.randperm(len(torch.unique(labels)))[:n_way]
        support_idx, query_idx = [], []
        for cls in classes:
            cls_indices = (labels == cls).nonzero(as_tuple=True)[0]
            support_idx.append(cls_indices[:k_shot])
            query_idx.append(cls_indices[k_shot:k_shot + q_queries])
        support_idx = torch.cat(support_idx)
        query_idx = torch.cat(query_idx)

        support_features = features[support_idx].to(device)
        query_features = features[query_idx].to(device)
        support_labels = labels[support_idx]
        query_labels = labels[query_idx].to(device)

        support_embeddings = model(support_features)
        query_embeddings = model(query_features)

        prototypes = compute_prototypes(support_embeddings, support_labels, n_way)
        dists = torch.stack([euclidean_dist(query_embeddings, proto) for proto in prototypes]).t()
        loss = criterion(-dists, query_labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(f"Task {task + 1}/{num_tasks}, Loss: {loss.item():.4f}")

def test_prototypical_network(model, features, labels, n_way, k_shot, q_queries, device,num_tasks):
    model.eval()
    accuracies = []
    with torch.no_grad():
        for _ in range(num_tasks):  # Test 20 tasks
            classes = torch.randperm(len(torch.unique(labels)))[:n_way]
            support_idx, query_idx = [], []
            for cls in classes:
                cls_indices = (labels == cls).nonzero(as_tuple=True)[0]
                support_idx.append(cls_indices[:k_shot])
                query_idx.append(cls_indices[k_shot:k_shot + q_queries])
            support_idx = torch.cat(support_idx)
            query_idx = torch.cat(query_idx)

            support_features = features[support_idx].to(device)
            query_features = features[query_idx].to(device)
            support_labels = labels[support_idx]
            query_labels = labels[query_idx].to(device)

            support_embeddings = model(support_features)
            query_embeddings = model(query_features)

            prototypes = compute_prototypes(support_embeddings, support_labels, n_way)
            dists = torch.stack([euclidean_dist(query_embeddings, proto) for proto in prototypes]).t()
            preds = torch.argmin(dists, dim=1)

            accuracy = (preds == query_labels).float().mean().item()
            accuracies.append(accuracy)

    mean_accuracy = np.mean(accuracies)
    print(f"Mean Accuracy: {mean_accuracy:.4f}")
    return mean_accuracy

data_OCD_fMRI = loadmat('')
classes = ['']
datasets = [data_OCD_fMRI[cls] for cls in classes]

nc_data = datasets[0]
duplicated_nc_data = np.tile(nc_data, (1, 1, 1))  

num_samples_to_generate = 1 * nc_data.shape[0]  
generated_nc_data = duplicated_nc_data+np.random.normal(0, 0.01, size=(num_samples_to_generate, nc_data.shape[1], nc_data.shape[2]))

augmented_nc_data = np.concatenate([nc_data, generated_nc_data], axis=0)
datasets[0] = augmented_nc_data

nc_labels = np.zeros(nc_data.shape[0])  
generated_labels = np.ones(generated_nc_data.shape[0])  
y_combined = np.hstack([nc_labels, generated_labels, np.ones(datasets[1].shape[0]) * 2])  


labels = [torch.full((data.shape[0],), i, dtype=torch.long) for i, data in enumerate(datasets)]
X = torch.tensor(np.vstack(datasets), dtype=torch.float32)
y = torch.cat(labels)
from sklearn.model_selection import StratifiedKFold
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
hidden_size = 128
num_layers = 2
num_classes = len(classes)
all_accuracy=[]

for fold, (train_idx, test_idx) in enumerate(kf.split(X,y)):
    print(f"\n=== Fold {fold + 1} ===")
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=16, shuffle=True)
    
    
    y_test_combined = y_combined[test_idx]

    valid_test_idx = (y_test_combined == 0) | (y_test_combined == 2)
    X_test = X_test[valid_test_idx]
    y_test = y_test[valid_test_idx]
    
    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=16, shuffle=False)

    classifier = ModelLSTM(num_regions=X.shape[1], time_series_len=X.shape[2], hidden_size=hidden_size,
                           num_layers=num_layers, num_classes=num_classes)
    print("Training classifier...")
    train_classifier(classifier, train_loader, test_loader, device, num_epochs=25, lr=0.0001)

    print("Extracting features...")
    train_features, train_labels = extract_features(classifier, train_loader, device)
    test_features, test_labels = extract_features(classifier, test_loader, device)

    proto_net = PrototypicalNetwork(feature_dim=hidden_size, hidden_dim=64)
    print("Training prototypical network...")
    train_prototypical_network(proto_net, train_features, train_labels, n_way=2, k_shot=10, q_queries=10,
                                num_tasks=25, lr=0.001, device=device)

    print("Testing prototypical network...")
    mean_accuracy2=test_prototypical_network(proto_net, test_features, test_labels, n_way=2, k_shot=5, q_queries=3, device=device,num_tasks=10)
    all_accuracy.append(mean_accuracy2)
print("5 fold Mean Accuracy:",np.mean(all_accuracy))