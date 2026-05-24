import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
import time

import config.config as config

class DecisionTreeModel:
    def __init__(self, **params):
        default_params = {
            'random_state': config.RANDOM_SEED,
            'max_depth': None,
            'min_samples_split': 2,
            'min_samples_leaf': 1,
            'ccp_alpha': 0.0,
            'criterion': 'gini',
            'class_weight': None
        }
        default_params.update(params)
        self.model = DecisionTreeClassifier(**default_params)
        self.params = default_params
    
    def fit(self, X, y):
        self.model.fit(X, y)
        return None
    
    def predict(self, X):
        predictions = self.model.predict(X)
        return predictions
    
    def predict_proba(self, X):
        return self.model.predict_proba(X)
    
    def get_params(self):
        return self.params
    
    def get_tree_info(self):
        return {
            'n_leaves': self.model.get_n_leaves(),
            'max_depth': self.model.get_depth(),
            'n_features': self.model.n_features_in_,
            'n_classes': self.model.n_classes_
        }

class KNNModel:
    def __init__(self, **params):
        default_params = {
            'n_neighbors': 5,
            'weights': 'uniform',
            'metric': 'euclidean',
            'n_jobs': -1
        }
        default_params.update(params)
        self.model = KNeighborsClassifier(**default_params)
        self.params = default_params
    
    def fit(self, X, y):
        self.model.fit(X, y)
        return None
    
    def predict(self, X):
        predictions = self.model.predict(X)
        return predictions
    
    def predict_proba(self, X):
        return self.model.predict_proba(X)
    
    def get_params(self):
        return self.params

class SVMModel:
    def __init__(self, **params):
        default_params = {
            'random_state': config.RANDOM_SEED,
            'C': 1.0,
            'kernel': 'rbf',
            'gamma': 'scale',
            'probability': True,
            'max_iter': 1000,
            'class_weight': None
        }
        default_params.update(params)

        if default_params['kernel'] == 'linear' and 'gamma' in default_params:
            del default_params['gamma']
        
        self.model = SVC(**default_params)
        self.params = default_params
    
    def fit(self, X, y):
        self.model.fit(X, y)
        return None
    
    def predict(self, X):
        predictions = self.model.predict(X)
        return predictions
    
    def predict_proba(self, X):
        return self.model.predict_proba(X)
    
    def get_params(self):
        return self.params
    
    def get_svm_info(self):
        return {
            'n_support': self.model.n_support_,
            'support_vectors_shape': self.model.support_vectors_.shape,
            'n_classes': len(self.model.classes_)
        }

class NeuralNetworkSklearn:
    def __init__(self, **params):
        default_params = {
            'random_state': config.RANDOM_SEED,
            'hidden_layer_sizes': (100,),
            'activation': 'relu',
            'solver': 'sgd',
            'alpha': 0.0001,
            'learning_rate': 'constant',
            'learning_rate_init': 0.01,
            'max_iter': 100,
            'early_stopping': True,
            'validation_fraction': 0.1,
            'n_iter_no_change': 10,
            'verbose': False
        }
        default_params.update(params)
        default_params['solver'] = 'sgd'
        
        self.model = MLPClassifier(**default_params)
        self.params = default_params
        self.loss_curve = None
        self.validation_scores = None
    
    def fit(self, X, y):
        self.model.fit(X, y)
        self.loss_curve = self.model.loss_curve_
        if hasattr(self.model, 'validation_scores_'):
            self.validation_scores = self.model.validation_scores_
        return None
    
    def predict(self, X):
        predictions = self.model.predict(X)
        return predictions
    
    def predict_proba(self, X):
        return self.model.predict_proba(X)
    
    def get_params(self):
        return self.params
    
    def get_training_info(self):
        return {
            'n_iter': self.model.n_iter_,
            'n_layers': self.model.n_layers_,
            'n_outputs': self.model.n_outputs_,
            'loss': self.model.loss_,
            'best_loss': self.model.best_loss_ if hasattr(self.model, 'best_loss_') else None
        }
    
    def get_loss_curve(self):
        return self.loss_curve, self.validation_scores


# ============================================================================
# NEURAL NETWORK (PyTorch)
# ============================================================================

class MLPPyTorch(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size, activation='relu'):
        super(MLPPyTorch, self).__init__()

        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'gelu':
            self.activation = nn.GELU()
        elif activation == 'silu':
            self.activation = nn.SiLU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()

        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(self.activation)
            prev_size = hidden_size
        
        layers.append(nn.Linear(prev_size, output_size))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)


class NeuralNetworkPyTorch:
    def __init__(self, input_size, output_size, hidden_sizes=[100], 
                 learning_rate=0.01, weight_decay=0.0001, batch_size=64,
                 max_epochs=500, early_stopping_patience=20, activation='relu',
                 device=None):

        self.input_size = input_size
        self.output_size = output_size
        self.hidden_sizes = hidden_sizes
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.early_stopping_patience = early_stopping_patience
        self.activation = activation
        
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        self.model = MLPPyTorch(input_size, hidden_sizes, output_size, activation)
        self.model.to(self.device)
        
        self.optimizer = optim.SGD(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            momentum=0.0,
            nesterov=False
        )
        self.criterion = nn.CrossEntropyLoss()
        
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        self.best_val_loss = float('inf')
        self.best_epoch = 0
        
    def fit(self, X_train, y_train, X_val=None, y_val=None):
        X_train_tensor = torch.FloatTensor(X_train).to(self.device)
        y_train_tensor = torch.LongTensor(y_train).to(self.device)
        
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(
            train_dataset, 
            batch_size=self.batch_size, 
            shuffle=True
        )
        
        if X_val is not None and y_val is not None:
            X_val_tensor = torch.FloatTensor(X_val).to(self.device)
            y_val_tensor = torch.LongTensor(y_val).to(self.device)
            use_validation = True
        else:
            use_validation = False
        
        start_time = time.time()
        patience_counter = 0
        
        for epoch in range(self.max_epochs):
            self.model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for batch_X, batch_y in train_loader:
                outputs = self.model(batch_X)
                loss = self.criterion(outputs, batch_y)
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                train_loss += loss.item() * batch_X.size(0)
                _, predicted = torch.max(outputs, 1)
                train_total += batch_y.size(0)
                train_correct += (predicted == batch_y).sum().item()
            
            epoch_train_loss = train_loss / train_total
            epoch_train_acc = train_correct / train_total
            
            self.train_losses.append(epoch_train_loss)
            self.train_accuracies.append(epoch_train_acc)
            
            if use_validation:
                self.model.eval()
                with torch.no_grad():
                    val_outputs = self.model(X_val_tensor)
                    val_loss = self.criterion(val_outputs, y_val_tensor)
                    _, val_predicted = torch.max(val_outputs, 1)
                    val_acc = (val_predicted == y_val_tensor).sum().item() / len(y_val_tensor)
                
                self.val_losses.append(val_loss.item())
                self.val_accuracies.append(val_acc)
                
                if val_loss.item() < self.best_val_loss:
                    self.best_val_loss = val_loss.item()
                    self.best_epoch = epoch
                    patience_counter = 0
                    self.best_model_state = self.model.state_dict().copy()
                else:
                    patience_counter += 1
                
                if patience_counter >= self.early_stopping_patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    self.model.load_state_dict(self.best_model_state)
                    break
        
        fit_time = time.time() - start_time
        return fit_time
    
    def predict(self, X):
        start_time = time.time()
        
        self.model.eval()
        X_tensor = torch.FloatTensor(X).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(X_tensor)
            _, predicted = torch.max(outputs, 1)
        
        predict_time = time.time() - start_time
        return predicted.cpu().numpy(), predict_time
    
    def predict_proba(self, X):
        self.model.eval()
        X_tensor = torch.FloatTensor(X).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(X_tensor)
            probabilities = torch.softmax(outputs, dim=1)
        
        return probabilities.cpu().numpy()
    
    def get_params(self):
        return {
            'hidden_sizes': self.hidden_sizes,
            'learning_rate': self.learning_rate,
            'weight_decay': self.weight_decay,
            'batch_size': self.batch_size,
            'max_epochs': self.max_epochs,
            'activation': self.activation,
            'optimizer': 'SGD (no momentum)'
        }
    
    def get_training_curves(self):
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_accuracies': self.train_accuracies,
            'val_accuracies': self.val_accuracies,
            'best_epoch': self.best_epoch
        }
    
    def save_model(self, path):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'params': self.get_params(),
            'training_curves': self.get_training_curves()
        }, path)
    
    def load_model(self, path):
        """Load model from file."""
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])


# ============================================================================
# MODEL FACTORY
# ============================================================================

def create_model(algorithm, **params):
    if algorithm == 'dt':
        return DecisionTreeModel(**params)
    elif algorithm == 'knn':
        return KNNModel(**params)
    elif algorithm == 'svm':
        return SVMModel(**params)
    elif algorithm == 'nn_sklearn':
        return NeuralNetworkSklearn(**params)
    elif algorithm == 'nn_pytorch':
        return NeuralNetworkPyTorch(**params)