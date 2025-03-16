import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import seaborn as sns

# Step 1: Classification Network
class Classifier(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, hidden_size3, num_classes):
        super(Classifier, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.bn1 = nn.BatchNorm1d(hidden_size1)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.bn2 = nn.BatchNorm1d(hidden_size2)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_size2, hidden_size3)
        self.bn3 = nn.BatchNorm1d(hidden_size3)
        self.relu3 = nn.ReLU()
        self.fc4 = nn.Linear(hidden_size3, num_classes)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x, return_features=False):
        x=self.fc1(x)
        if x.size(0) > 1:
            x = self.bn1(x)
        x=self.relu1(x)
        x=self.fc2(x)
        if x.size(0) > 1:
            x = self.bn2(x)
        x=self.relu2(x)
        x=self.fc3(x)
        if x.size(0) > 1: 
            x = self.bn3(x)
        features=self.relu3(x)
        
        logits = self.fc4(features)
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

# Step 3: Compute Prototypes
def compute_prototypes(support_features, support_labels, n_way):
    prototypes = []
    for cls in range(n_way):
        cls_features = support_features[support_labels == cls]
        prototype = cls_features.mean(dim=0)
        prototypes.append(prototype)
    return torch.stack(prototypes)

def euclidean_dist(x, y):
    return ((x - y) ** 2).sum(dim=-1)

# Step 4: Train Classification Model
def train_classifier(model, train_loader, val_loader, device, num_classes, num_epochs=50, lr=0.001):
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

# Step 5: Evaluate Classification Model
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

# Step 6: Extract Features using Classification Model
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

# Step 7: Train Prototypical Network
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

# Step 8: Test Prototypical Network
def test_prototypical_network(model, features, labels, n_way, k_shot, q_queries, device,num_tasks):
    model.eval()
    accuracies = []
    with torch.no_grad():
        for _ in range(num_tasks):  
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



from sklearn.model_selection import KFold
from torch.utils.data import Subset

if __name__ == "__main__":
    from scipy.io import loadmat
    data_PPMI = loadmat("")
    classes = [""]
    datasets = [data_PPMI[cls] for cls in classes]
    
    nc_data = datasets[0]
    num_samples_to_generate = nc_data.shape[0]

    generated_nc_data = nc_data + np.random.normal(0, 0.01, size=nc_data.shape)

    augmented_nc_data = np.vstack([nc_data, generated_nc_data])
    datasets[0] = augmented_nc_data
    
    
    labels = [torch.full((data.shape[0],), i, dtype=torch.long) for i, data in enumerate(datasets)]
    X = torch.tensor(np.vstack(datasets), dtype=torch.float32)
    y = torch.cat(labels)
    
    nc_labels = np.zeros(nc_data.shape[0])
    generated_labels = np.ones(generated_nc_data.shape[0])  
    y_combined = np.hstack([nc_labels, generated_labels, np.ones(datasets[1].shape[0]) * 2])

    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fold = 1
    all_accuracies = []

    for train_index, test_index in kf.split(X,y):
        print(f"\n=== Fold {fold} ===")
        
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        y_test_combined = y_combined[test_index]
        
        valid_test_idx = (y_test_combined == 0) | (y_test_combined == 2)
        X_test = X_test[valid_test_idx]
        y_test = y_test[valid_test_idx]
        

        train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=16, shuffle=True)
        test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=16, shuffle=False)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        classifier = Classifier(input_size=X.shape[1], hidden_size1=128, hidden_size2=64, hidden_size3=32, num_classes=len(classes))
        print("Training classifier...")
        train_classifier(classifier, train_loader, test_loader, device, num_classes=len(classes), num_epochs=20, lr=0.0005)

        print("Extracting features...")
        train_features, train_labels = extract_features(classifier, train_loader, device)
        test_features, test_labels = extract_features(classifier, test_loader, device)

        print("Training prototypical network...")
        proto_net = PrototypicalNetwork(feature_dim=32, hidden_dim=16)
        train_prototypical_network(proto_net, train_features, train_labels, n_way=2, k_shot=20, q_queries=10, num_tasks=50, lr=0.001, device=device)

        print("Testing prototypical network...")
        accuracy = test_prototypical_network(proto_net, test_features, test_labels, n_way=2, k_shot=10, q_queries=5, device=device,num_tasks=20)
        all_accuracies.append(accuracy)

        fold += 1

    mean_accuracy = np.mean(all_accuracies)
    print(f"\nMean Accuracy across 5 folds: {mean_accuracy:.4f}")
