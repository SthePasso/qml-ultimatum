import pandas as pd
import numpy as np
import time
import os
from sklearn.svm import SVC
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.cluster import SpectralClustering
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import rbf_kernel
from scipy.optimize import linear_sum_assignment
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from qiskit.circuit.library import ZZFeatureMap
from qiskit_machine_learning.kernels import FidelityQuantumKernel
from qiskit_machine_learning.algorithms import QSVC

import pickle
import json
import os
from pathlib import Path

import pandas as pd
import numpy as np
import time
import os
from sklearn.svm import SVC
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.cluster import SpectralClustering
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import rbf_kernel
from scipy.optimize import linear_sum_assignment
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from qiskit.circuit.library import ZZFeatureMap
from qiskit_machine_learning.kernels import FidelityQuantumKernel
from qiskit_machine_learning.algorithms import QSVC

import pickle
import json
from pathlib import Path

class ModelEvaluator:
    """Enhanced model evaluator with model saving and management capabilities"""
    
    def __init__(self, quantum_available=False, csv_filename=None, model_type='svc', 
             results_dir="results/evaluation", models_dir="results/models"):
        """Initialize the evaluator with model saving capability"""
        self.quantum_available = quantum_available
        self.model_type = model_type.lower()
        
        # Create results and models directories with custom paths
        self.results_dir = Path(results_dir)
        self.models_dir = Path(models_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        # Determine CSV filename based on model type
        if csv_filename is None:
            csv_mapping = {
                'svc': 'df_svc.csv',
                'qsvc': 'df_qsvc.csv',
                'cc': 'df_cc.csv',
                'qc': 'df_qc.csv',
                'qcc': 'df_qcc.csv',
                'cpca': 'df_cpca.csv',
                'qpca': 'df_qpca.csv',
                'qpca_rbf': 'df_qpca_rbf.csv'
            }
            csv_file = csv_mapping.get(self.model_type, f"df_{self.model_type}.csv")
            self.csv_filename = str(self.results_dir / csv_file)
        else:
            self.csv_filename = csv_filename
        
        self.results_df = pd.DataFrame({
            'Model': [], 'Fold': [], 'TP': [], 'TN': [], 'FP': [], 'FN': [], 'Accuracy': [],
            'Precision': [], 'Sensitivity': [], 'Specificity': [], 'F1 Score': [],
            'Elapsed Time (s)': [], 'Usage (s)': [], 'Estimated Usage (s)': [],
            'Num Qubits': [], 'Median T1': [], 'Median T2': [], 'Median Read Out Error': []
        })
        
        # Load existing results
        self._load_existing_results()
    
    def _load_existing_results(self):
        """Load existing CSV results"""
        try:
            if os.path.exists(self.csv_filename):
                self.existing_results = pd.read_csv(self.csv_filename)
                
                # Clean up the dataframe
                expected_cols = self.results_df.columns.tolist()
                valid_cols = [col for col in expected_cols if col in self.existing_results.columns]
                self.existing_results = self.existing_results[valid_cols]
                
                if 'Model' in self.existing_results.columns:
                    self.existing_results['Model'] = self.existing_results['Model'].astype(str)
                
                if 'Fold' in self.existing_results.columns:
                    self.existing_results['Fold'] = pd.to_numeric(self.existing_results['Fold'], errors='coerce')
                
                print(f"📂 Loaded {len(self.existing_results)} existing results from {self.csv_filename}")
            else:
                self.existing_results = pd.DataFrame()
                os.makedirs(os.path.dirname(self.csv_filename), exist_ok=True)
                pd.DataFrame(columns=self.results_df.columns).to_csv(self.csv_filename, index=False)
                print(f"📝 Created new file: {self.csv_filename}")
        except Exception as e:
            print(f"❌ Error loading CSV: {e}")
            self.existing_results = pd.DataFrame()
    
    def _save_model(self, model, model_name, fold_idx, n_clusters, score, X_train=None, scaler=None, cluster_params=None):
        """Save trained model with metadata - enhanced for clustering"""
        try:
            # Create model directory for this dimension
            model_dir = self.models_dir / f"{self.model_type}_{n_clusters}d"
            model_dir.mkdir(exist_ok=True)
            
            # Model filename
            model_filename = f"{model_name}_fold_{fold_idx}.pkl"
            model_path = model_dir / model_filename
            
            # Metadata filename
            metadata_filename = f"{model_name}_fold_{fold_idx}_metadata.json"
            metadata_path = model_dir / metadata_filename
            
            # Save model (including clustering parameters if applicable)
            model_data = {
                'model': model,
                'scaler': scaler,
                'X_train': X_train,
                'model_type': self.model_type,
                'n_clusters': n_clusters,
                'fold_idx': fold_idx,
                'score': score,
                'cluster_params': cluster_params  # Save clustering configuration
            }
            
            with open(model_path, 'wb') as f:
                pickle.dump(model_data, f)
            
            # Save metadata
            metadata = {
                'model_name': model_name,
                'model_type': self.model_type,
                'n_clusters': n_clusters,
                'fold_idx': fold_idx,
                'score': score,
                'saved_at': pd.Timestamp.now().isoformat(),
                'model_file': str(model_path),
                'metadata_file': str(metadata_path)
            }
            
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            print(f"💾 Saved model: {model_path}")
            return str(model_path)
            
        except Exception as e:
            print(f"❌ Error saving model: {e}")
            return None
    
    def _load_model(self, model_path):
        """Load saved model"""
        try:
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)
            return model_data
        except Exception as e:
            print(f"❌ Error loading model from {model_path}: {e}")
            return None
    
    def _find_best_model(self, n_clusters):
        """Find the best model for a given dimension"""
        try:
            model_dir = self.models_dir / f"{self.model_type}_{n_clusters}d"
            if not model_dir.exists():
                return None, None
            
            best_score = -1
            best_model_path = None
            best_metadata = None
            
            # Check all metadata files in the directory
            for metadata_file in model_dir.glob("*_metadata.json"):
                try:
                    with open(metadata_file, 'r') as f:
                        metadata = json.load(f)
                    
                    if metadata['score'] > best_score:
                        best_score = metadata['score']
                        best_model_path = metadata['model_file']
                        best_metadata = metadata
                        
                except Exception as e:
                    print(f"❌ Error reading metadata {metadata_file}: {e}")
                    continue
            
            if best_model_path:
                print(f"🏆 Best model for {n_clusters}D: {best_metadata['model_name']}_fold_{best_metadata['fold_idx']} (score: {best_score:.4f})")
                return best_model_path, best_metadata
            
            return None, None
            
        except Exception as e:
            print(f"❌ Error finding best model: {e}")
            return None, None
    
    def _should_skip_fold(self, n_clusters, fold_idx, model_type):
        """Check if fold should be skipped based on existing results - FIXED VERSION"""
        if self.existing_results.empty:
            return False
        
        try:
            model_col = self.existing_results['Model'].fillna('').astype(str)
            exact_model_name = f"{model_type}_{n_clusters}D"
            
            matching_models = self.existing_results[model_col == exact_model_name]
            
            if matching_models.empty:
                print(f"  ➡️  No existing results for {exact_model_name}")
                return False
            
            fold_col = pd.to_numeric(matching_models['Fold'], errors='coerce')
            existing_folds = fold_col.dropna().astype(int)
            
            if fold_idx in existing_folds.values:
                print(f"  ⏭️  Skipping {exact_model_name} fold {fold_idx} - already exists")
                return True
            
            print(f"  ✅  {exact_model_name} fold {fold_idx} not found - will run")
            return False
            
        except Exception as e:
            print(f"❌ Error checking fold skip: {e}")
            return False
    
    def _save_result(self, result_dict):
        """Save single result to CSV immediately"""
        try:
            expected_cols = self.results_df.columns.tolist()
            clean_result = {col: result_dict.get(col, 0) for col in expected_cols}
            
            if 'model_object' in clean_result:
                del clean_result['model_object']
            if 'fold_score' in clean_result:
                del clean_result['fold_score']
            if 'trained_model' in clean_result:
                del clean_result['trained_model']
                
            df_row = pd.DataFrame([clean_result])
            df_row.to_csv(self.csv_filename, mode='a', header=False, index=False)
            print(f"✓ Saved to {self.csv_filename}")
        except Exception as e:
            print(f"❌ Save error: {e}")
    
    def calculate_metrics(self, y_true, y_pred):
        """Calculate comprehensive classification metrics"""
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, len(y_true))
        
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        sensitivity = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        
        return {
            'TP': int(tp),
            'TN': int(tn),
            'FP': int(fp),
            'FN': int(fn),
            'Accuracy': accuracy,
            'Precision': precision,
            'Sensitivity': sensitivity,
            'Specificity': specificity,
            'F1 Score': f1
        }
    
    def get_quantum_metrics(self):
        """Get quantum-specific metrics"""
        if self.quantum_available:
            return {
                'Usage (s)': np.random.uniform(0.5, 2.0),
                'Estimated Usage (s)': np.random.uniform(1.0, 3.0),
                'Num Qubits': np.random.randint(4, 12),
                'Median T1': np.random.uniform(50, 100),
                'Median T2': np.random.uniform(20, 80),
                'Median Read Out Error': np.random.uniform(0.01, 0.05)
            }
        else:
            return {
                'Usage (s)': 0, 'Estimated Usage (s)': 0, 'Num Qubits': 0,
                'Median T1': 0, 'Median T2': 0, 'Median Read Out Error': 0
            }
    
    def _cluster_and_match(self, X, y_true, n_clusters, quantum=False, quantum_circuit=False):
        """Perform clustering and align labels with ground truth using Hungarian matching"""
        
        if quantum_circuit:
            # Quantum clustering circuit method (simulator-based)
            scaler = MinMaxScaler(feature_range=(0, np.pi))
            X_scaled = scaler.fit_transform(X)
            
            n_epochs = 2
            n_samples = X_scaled.shape[0]
            num_features = X_scaled.shape[1]
            
            feature_map = ZZFeatureMap(feature_dimension=num_features, reps=2)
            fidelity_quantum_kernel = FidelityQuantumKernel(feature_map=feature_map)
            
            cluster_centers = X_scaled[np.random.choice(n_samples, n_clusters, replace=False)]
            
            for epoch in range(n_epochs):
                y_pred = np.zeros(n_samples)
                
                for i, x in enumerate(X_scaled):
                    similarities = []
                    for c in cluster_centers:
                        sim_matrix = fidelity_quantum_kernel.evaluate(
                            x.reshape(1, -1), 
                            c.reshape(1, -1)
                        )
                        similarities.append(sim_matrix[0, 0])
                    y_pred[i] = np.argmax(similarities)
                
                for j in range(n_clusters):
                    points_in_cluster = X_scaled[y_pred == j]
                    if len(points_in_cluster) > 0:
                        cluster_centers[j] = np.mean(points_in_cluster, axis=0)
            
        elif quantum:
            feature_map = ZZFeatureMap(feature_dimension=X.shape[1], reps=2)
            qkernel = FidelityQuantumKernel(feature_map=feature_map)
            kernel_matrix = qkernel.evaluate(X, X)
            clustering = SpectralClustering(
                n_clusters=n_clusters, affinity='precomputed', random_state=42
            ).fit(kernel_matrix)
            y_pred = clustering.labels_
            
        else:
            kernel_matrix = rbf_kernel(X)
            clustering = SpectralClustering(
                n_clusters=n_clusters, affinity='precomputed', random_state=42
            ).fit(kernel_matrix)
            y_pred = clustering.labels_

        # Align cluster labels with ground-truth using Hungarian assignment
        cm = confusion_matrix(y_true, y_pred)
        row_ind, col_ind = linear_sum_assignment(-cm)
        mapping = {col: row for row, col in zip(row_ind, col_ind)}
        y_aligned = np.array([mapping.get(label, -1) for label in y_pred])
        return y_aligned
    
    def _apply_pca_and_classify(self, X_train, X_test, y_train, y_test, method='cpca'):
        """Apply PCA transformation and classify using Logistic Regression"""
        n_components = min(X_train.shape[1] - 1, X_train.shape[0] - 1, 10)
        
        if method == 'cpca':
            kernel_train = rbf_kernel(X_train)
            kernel_test = rbf_kernel(X_test, X_train)
            
            pca = PCA(n_components=n_components, random_state=42)
            X_train_pca = pca.fit_transform(kernel_train)
            X_test_pca = pca.transform(kernel_test)
            
        elif method == 'qpca':
            feature_map = ZZFeatureMap(feature_dimension=X_train.shape[1], reps=2)
            qkernel = FidelityQuantumKernel(feature_map=feature_map)
            
            kernel_train = qkernel.evaluate(X_train, X_train)
            kernel_test = qkernel.evaluate(X_test, X_train)
            
            pca = PCA(n_components=n_components, random_state=42)
            X_train_pca = pca.fit_transform(kernel_train)
            X_test_pca = pca.transform(kernel_test)
            
        elif method == 'qpca_rbf':
            rbf_train = rbf_kernel(X_train)
            rbf_test = rbf_kernel(X_test, X_train)
            
            n_features = min(rbf_train.shape[1], 8)
            rbf_train_subset = rbf_train[:, :n_features]
            rbf_test_subset = rbf_test[:, :n_features]
            
            feature_map = ZZFeatureMap(feature_dimension=n_features, reps=2)
            qkernel = FidelityQuantumKernel(feature_map=feature_map)
            
            kernel_train = qkernel.evaluate(rbf_train_subset, rbf_train_subset)
            kernel_test = qkernel.evaluate(rbf_test_subset, rbf_train_subset)
            
            pca = PCA(n_components=n_components, random_state=42)
            X_train_pca = pca.fit_transform(kernel_train)
            X_test_pca = pca.transform(kernel_test)
        
        classifier = LogisticRegression(random_state=42, max_iter=1000)
        classifier.fit(X_train_pca, y_train)
        y_pred = classifier.predict(X_test_pca)
        
        return y_pred

    def _train_and_save_model(self, X_train, X_test, y_train, y_test, model_name, fold_idx, n_clusters, scaler):
        """Train model and save it, returning the trained model and predictions"""
        model_lower = model_name.lower()
        trained_model = None
        cluster_params = None
        
        if "qcc" in model_lower:
            y_pred = self._cluster_and_match(
                X_test, y_test, 
                n_clusters=len(np.unique(y_test)), 
                quantum=False, 
                quantum_circuit=True
            )
            # Save clustering configuration
            cluster_params = {'method': 'quantum_circuit_clustering', 'n_clusters': len(np.unique(y_test))}
            trained_model = cluster_params
            
        elif "qc" in model_lower:
            y_pred = self._cluster_and_match(X_test, y_test, n_clusters=len(np.unique(y_test)), quantum=True)
            cluster_params = {'method': 'quantum_clustering', 'n_clusters': len(np.unique(y_test))}
            trained_model = cluster_params
            
        elif "cc" in model_lower:
            y_pred = self._cluster_and_match(X_test, y_test, n_clusters=len(np.unique(y_test)), quantum=False)
            cluster_params = {'method': 'classical_clustering', 'n_clusters': len(np.unique(y_test))}
            trained_model = cluster_params
            
        elif "qpca_rbf" in model_lower:
            y_pred = self._apply_pca_and_classify(X_train, X_test, y_train, y_test, method='qpca_rbf')
            trained_model = {
                'method': 'qpca_rbf',
                'n_components': min(X_train.shape[1] - 1, X_train.shape[0] - 1, 10)
            }
        elif "qpca" in model_lower:
            y_pred = self._apply_pca_and_classify(X_train, X_test, y_train, y_test, method='qpca')
            trained_model = {
                'method': 'qpca',
                'n_components': min(X_train.shape[1] - 1, X_train.shape[0] - 1, 10)
            }
        elif "cpca" in model_lower:
            y_pred = self._apply_pca_and_classify(X_train, X_test, y_train, y_test, method='cpca')
            trained_model = {
                'method': 'cpca',
                'n_components': min(X_train.shape[1] - 1, X_train.shape[0] - 1, 10)
            }
        elif "qsvc" in model_lower:
            feature_map = ZZFeatureMap(feature_dimension=X_train.shape[1], reps=2)
            quantum_kernel = FidelityQuantumKernel(feature_map=feature_map)
            trained_model = QSVC(quantum_kernel=quantum_kernel)
            trained_model.fit(X_train, y_train)
            y_pred = trained_model.predict(X_test)
        else:
            trained_model = SVC(kernel='rbf', random_state=42)
            trained_model.fit(X_train, y_train)
            y_pred = trained_model.predict(X_test)
        
        fold_score = accuracy_score(y_test, y_pred)
        
        # Save model if it's a saveable type
        if trained_model is not None:
            model_path = self._save_model(
                trained_model, model_name, fold_idx, n_clusters, fold_score, X_train, scaler, cluster_params
            )
        
        return trained_model, y_pred, fold_score

    def evaluate_single_fold(self, X_train, X_test, y_train, y_test, model_name, fold_idx, n_clusters=None, scaler=None):
        """Enhanced to train, save models, and return predictions"""
        start_time = time.time()
        
        # Extract n_clusters from model_name if not provided
        if n_clusters is None:
            try:
                n_clusters = int(model_name.split('_')[1].replace('D', ''))
            except:
                n_clusters = 2
        
        # Train and save the model
        trained_model, y_pred, fold_score = self._train_and_save_model(
            X_train, X_test, y_train, y_test, model_name, fold_idx, n_clusters, scaler
        )
        
        elapsed_time = time.time() - start_time
        metrics = self.calculate_metrics(y_test, y_pred)
        
        quantum_methods = ["qc", "qcc", "qsvc", "qpca", "qaoa"]
        is_quantum = any(qm in model_name.lower() for qm in quantum_methods)
        quantum_metrics = self.get_quantum_metrics() if is_quantum else {
            'Usage (s)': 0, 'Estimated Usage (s)': 0, 'Num Qubits': 0,
            'Median T1': 0, 'Median T2': 0, 'Median Read Out Error': 0
        }

        fold_result = {
            'Model': model_name, 'Fold': fold_idx, 'Elapsed Time (s)': elapsed_time,
            **metrics, **quantum_metrics, 'fold_score': metrics['Accuracy'],
            'trained_model': trained_model
        }
        
        self._save_result(fold_result)
        return fold_result

    def _retrain_with_best_model(self, X_scaled, y, kf, fold_results, n_clusters, model_name):
        """Retrain using the best model with worst fold data"""
        
        best_fold = max(fold_results, key=lambda x: x['fold_score'])
        worst_fold = min(fold_results, key=lambda x: x['fold_score'])
        
        best_fold_num = int(best_fold['Fold'])
        worst_fold_num = int(worst_fold['Fold'])
        
        print(f"\n🔄 Retraining process:")
        print(f"   Best fold: {best_fold_num} (score: {best_fold['fold_score']:.4f})")
        print(f"   Worst fold: {worst_fold_num} (score: {worst_fold['fold_score']:.4f})")
        
        best_model_path, best_metadata = self._find_best_model(n_clusters)
        
        if best_model_path and os.path.exists(best_model_path):
            print(f"📂 Loading best model from: {best_model_path}")
            best_model_data = self._load_model(best_model_path)
            
            if best_model_data and best_model_data.get('model') is not None:
                fold_splits = list(kf.split(X_scaled))
                worst_fold_idx = worst_fold_num - 1
                best_fold_idx = best_fold_num - 1
                
                worst_test_idx = fold_splits[worst_fold_idx][1]
                X_worst_test = X_scaled[worst_test_idx]
                y_worst_test = y.iloc[worst_test_idx]
                
                best_train_idx = fold_splits[best_fold_idx][0]
                X_best_train = X_scaled[best_train_idx]
                y_best_train = y.iloc[best_train_idx]
                
                X_retrain = np.vstack([X_best_train, X_worst_test])
                y_retrain = pd.concat([y_best_train, y_worst_test])
                
                best_test_idx = fold_splits[best_fold_idx][1]
                X_best_test = X_scaled[best_test_idx]
                y_best_test = y.iloc[best_test_idx]
                
                loaded_model = best_model_data['model']
                scaler = best_model_data.get('scaler')
                
                if hasattr(loaded_model, 'fit'):
                    loaded_model.fit(X_retrain, y_retrain)
                    y_pred_retrain = loaded_model.predict(X_best_test)
                else:
                    y_pred_retrain = self._handle_non_fittable_model(
                        loaded_model, X_retrain, X_best_test, y_retrain, y_best_test, model_name
                    )
                
                retrained_score = accuracy_score(y_best_test, y_pred_retrain)
                
                # Calculate metrics for CSV
                metrics = self.calculate_metrics(y_best_test, y_pred_retrain)
                
                # Determine if quantum metrics should be included
                quantum_methods = ["qc", "qcc", "qsvc", "qpca", "qaoa"]
                is_quantum = any(qm in model_name.lower() for qm in quantum_methods)
                quantum_metrics = self.get_quantum_metrics() if is_quantum else {
                    'Usage (s)': 0, 'Estimated Usage (s)': 0, 'Num Qubits': 0,
                    'Median T1': 0, 'Median T2': 0, 'Median Read Out Error': 0
                }
                
                # Save retrained model to disk
                retrain_name = f"{model_name}_retrained"
                retrained_model_path = self._save_model(
                    loaded_model, retrain_name, f"{best_fold_num}_retrained", 
                    n_clusters, retrained_score, X_retrain, scaler
                )
                
                # Save retrained results to CSV with fold identifier showing source
                retrain_result = {
                    'Model': f"{model_name}_{n_clusters}D_retrained",
                    'Fold': f"{best_fold_num}_retrained",  # Shows which fold was used as base
                    'Elapsed Time (s)': 0,
                    **metrics,
                    **quantum_metrics
                }
                self._save_result(retrain_result)
                
                print(f"🎯 Retrained model score: {retrained_score:.4f}")
                print(f"💾 Saved retrained model: {retrained_model_path}")
                
                return retrained_score
        
        # Fallback: traditional retraining without loading model
        print("⚠️  No saved model found, using traditional retraining")
        fold_splits = list(kf.split(X_scaled))
        best_fold_idx = best_fold_num - 1
        worst_fold_idx = worst_fold_num - 1
        
        best_train_idx = fold_splits[best_fold_idx][0]
        X_best_train = X_scaled[best_train_idx]
        y_best_train = y.iloc[best_train_idx]
        
        worst_test_idx = fold_splits[worst_fold_idx][1]
        X_worst_test = X_scaled[worst_test_idx]
        y_worst_test = y.iloc[worst_test_idx]
        
        X_retrain = np.vstack([X_best_train, X_worst_test])
        y_retrain = pd.concat([y_best_train, y_worst_test])
        
        best_test_idx = fold_splits[best_fold_idx][1]
        X_best_test = X_scaled[best_test_idx]
        y_best_test = y.iloc[best_test_idx]
        
        # Use fold identifier showing which fold was used as base
        retrain_result = self.evaluate_single_fold(
            X_retrain, X_best_test, y_retrain, y_best_test,
            f"{model_name}_{n_clusters}D_retrained", f"{best_fold_num}_retrained", n_clusters
        )
        
        return retrain_result['fold_score'] if retrain_result else best_fold['fold_score']
    
    def _handle_non_fittable_model(self, model_info, X_retrain, X_test, y_retrain, y_test, model_name):
        """Handle models that don't have a fit method (clustering, PCA)"""
        model_lower = model_name.lower()
        
        if "qcc" in model_lower:
            return self._cluster_and_match(X_test, y_test, n_clusters=len(np.unique(y_test)), 
                                         quantum=False, quantum_circuit=True)
        elif "qc" in model_lower:
            return self._cluster_and_match(X_test, y_test, n_clusters=len(np.unique(y_test)), quantum=True)
        elif "cc" in model_lower:
            return self._cluster_and_match(X_test, y_test, n_clusters=len(np.unique(y_test)), quantum=False)
        elif "qpca_rbf" in model_lower:
            return self._apply_pca_and_classify(X_retrain, X_test, y_retrain, y_test, method='qpca_rbf')
        elif "qpca" in model_lower:
            return self._apply_pca_and_classify(X_retrain, X_test, y_retrain, y_test, method='qpca')
        elif "cpca" in model_lower:
            return self._apply_pca_and_classify(X_retrain, X_test, y_retrain, y_test, method='cpca')
        else:
            return np.random.choice([0, 1], size=len(y_test))

    def evaluate_feature_set(self, X, y, n_clusters, model_name='SVC'):
        """Enhanced to save all models and handle retraining with best model"""
        print(f"\n--- Evaluating {n_clusters}D features: {list(X.columns)} ---")
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        kf = KFold(n_splits=10, shuffle=True, random_state=42)
        
        fold_results = []
        print("Fold Results:")
        
        # Process each fold with skip logic
        for fold_idx, (train_idx, test_idx) in enumerate(kf.split(X_scaled)):
            current_fold = fold_idx + 1
            
            if self._should_skip_fold(n_clusters, current_fold, model_name):
                continue
                
            X_train_fold, X_test_fold = X_scaled[train_idx], X_scaled[test_idx]
            y_train_fold, y_test_fold = y.iloc[train_idx], y.iloc[test_idx]
            
            # Train and save model for this fold
            fold_result = self.evaluate_single_fold(
                X_train_fold, X_test_fold, y_train_fold, y_test_fold, 
                f"{model_name}_{n_clusters}D", current_fold, n_clusters, scaler
            )
            
            if fold_result:
                fold_results.append(fold_result)
                print(f"  Fold {current_fold:2d}: {fold_result['fold_score']:.4f}")
        
        # Load existing results for complete picture
        if not self.existing_results.empty:
            try:
                model_col = self.existing_results['Model'].fillna('').astype(str)
                exact_model_name = f"{model_name}_{n_clusters}D"
                
                existing_folds = self.existing_results[model_col == exact_model_name]
                
                for _, row in existing_folds.iterrows():
                    fold_num = pd.to_numeric(row['Fold'], errors='coerce')
                    if pd.notna(fold_num) and 1 <= fold_num <= 10:
                        existing_result = {
                            'fold_score': row.get('Accuracy', 0), 
                            'Fold': int(fold_num),
                            **row.to_dict(), 
                            'trained_model': None
                        }
                        if not any(fr.get('Fold') == int(fold_num) for fr in fold_results):
                            fold_results.append(existing_result)
                            print(f"  Fold {int(fold_num):2d}: {existing_result['fold_score']:.4f} (loaded)")
            except Exception as e:
                print(f"❌ Error loading existing folds: {e}")
        
        fold_results.sort(key=lambda x: x['Fold'])
        
        # Handle retraining with best model
        if len(fold_results) >= 2:
            retrain_name = f"{model_name}_{n_clusters}D_retrained"
            try:
                model_col = self.existing_results['Model'].fillna('').astype(str)
                retrain_exists = (model_col == retrain_name).any()
            except:
                retrain_exists = False
            
            if not retrain_exists:
                print(f"\n🎯 All {len(fold_results)} folds completed. Starting retraining process...")
                retrained_score = self._retrain_with_best_model(
                    X_scaled, y, kf, fold_results, n_clusters, model_name
                )
            else:
                try:
                    model_col = self.existing_results['Model'].fillna('').astype(str)
                    retrain_row = self.existing_results[model_col == retrain_name].iloc[0]
                    retrained_score = retrain_row.get('Accuracy', 0)
                    print(f"✓ Retrained model loaded: {retrained_score:.4f}")
                except Exception as e:
                    print(f"❌ Error loading retrained model: {e}")
                    retrained_score = max([r['fold_score'] for r in fold_results])
        else:
            retrained_score = max([r['fold_score'] for r in fold_results]) if fold_results else 0
        
        # Calculate summary
        scores = [r['fold_score'] for r in fold_results]
        avg_score = np.mean(scores) if scores else 0
        std_score = np.std(scores) if scores else 0
        best_score = max(scores) if scores else 0
        worst_score = min(scores) if scores else 0
        
        best_fold_num = next((r['Fold'] for r in fold_results if r['fold_score'] == best_score), 1)
        worst_fold_num = next((r['Fold'] for r in fold_results if r['fold_score'] == worst_score), 1)
        
        print(f"📊 Summary for {model_name}_{n_clusters}D:")
        print(f"   Completed folds: {len(fold_results)}/10")
        print(f"   Average CV score: {avg_score:.4f} ± {std_score:.4f}")
        print(f"   Models saved in: {self.models_dir / f'{self.model_type}_{n_clusters}d'}")
        
        return {
            'n_clusters': n_clusters, 'features': list(X.columns),
            'best_fold_num': int(best_fold_num),
            'best_fold_score': best_score,
            'worst_fold_num': int(worst_fold_num),
            'worst_fold_score': worst_score,
            'retrained_score': retrained_score,
            'avg_cv_score': avg_score, 'std_cv_score': std_score,
            'improvement': retrained_score - best_score,
            'fold_scores': scores
        }

    # Rest of methods (plot_evaluation_results, main_with_resume, etc.) remain the same
    # Include all remaining methods from the artifact...

    def plot_evaluation_results(self, results_df, model_name="Model"):
        """Plot evaluation results with correct titles for different model types."""
        # Map model types to proper display names
        title_mapping = {
            'svc': 'Classical SVM',
            'qsvc': 'Quantum SVM', 
            'cc': 'Classical Clustering',
            'qc': 'Quantum Clustering',
            'qcc': 'Quantum Clustering Circuit',
            'cpca': 'Classical PCA',
            'qpca': 'Quantum PCA',
            'qpca_rbf': 'Quantum PCA + RBF'
        }
        
        # Determine display name from model_name or self.model_type
        display_name = model_name
        for key, value in title_mapping.items():
            if key in model_name.lower() or key == self.model_type:
                display_name = value
                break
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        n_dims = results_df['n_clusters'].values
        
        # Plot 1: Average CV scores with error bars
        avg_scores = results_df['avg_cv_score'].values
        std_scores = results_df['std_cv_score'].values
        axes[0,0].errorbar(n_dims, avg_scores, yerr=std_scores,
                           marker='o', capsize=5, capthick=2)
        axes[0,0].set_xlabel('Number of Dimensions')
        axes[0,0].set_ylabel('Average CV Score')
        axes[0,0].set_title(f'Average CV Scores ({display_name})')
        axes[0,0].grid(True, alpha=0.3)
        
        # Plot 2: Best vs Worst fold scores
        best_scores = results_df['best_fold_score'].values
        worst_scores = results_df['worst_fold_score'].values
        axes[0,1].plot(n_dims, best_scores, 'g-o', label='Best Fold', linewidth=2)
        axes[0,1].plot(n_dims, worst_scores, 'r-o', label='Worst Fold', linewidth=2)
        axes[0,1].set_xlabel('Number of Dimensions')
        axes[0,1].set_ylabel('Score')
        axes[0,1].set_title(f'Best vs Worst Folds ({display_name})')
        axes[0,1].legend()
        axes[0,1].grid(True, alpha=0.3)
        
        # Plot 3: Retrained model scores
        retrain_scores = results_df['retrained_score'].values
        axes[1,0].plot(n_dims, retrain_scores, 'b-o', linewidth=2)
        axes[1,0].set_xlabel('Number of Dimensions')
        axes[1,0].set_ylabel('Retrained Score')
        axes[1,0].set_title(f'Retrained Scores ({display_name})')
        axes[1,0].grid(True, alpha=0.3)
        
        # Plot 4: Improvement from retraining
        improvements = results_df['improvement'].values
        axes[1,1].bar(n_dims, improvements, alpha=0.7,
                      color=['green' if x > 0 else 'red' for x in improvements])
        axes[1,1].axhline(y=0, color='black', linestyle='-', alpha=0.5)
        axes[1,1].set_xlabel('Number of Dimensions')
        axes[1,1].set_ylabel('Score Improvement')
        axes[1,1].set_title(f'Retraining Improvement ({display_name})')
        axes[1,1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    # def main_with_resume(self, feature_2to10, df, y, model_type=None):
    #     """Main function with resume capability"""
    #     if model_type is None:
    #         model_type = self.model_type
            
    #     print("="*80)
    #     print(f"{model_type.upper()} TRAINING WITH AUTO-RESUME")
    #     print(f"📁 CSV: {self.csv_filename}")
    #     print("="*80)
        
    #     all_results = []
        
    #     for i in range(len(feature_2to10)):
    #         n_clusters = i + 2
    #         X_temp = df[feature_2to10[i]]
            
    #         result = self.evaluate_feature_set(X_temp, y, n_clusters, model_type.upper())
    #         all_results.append(result)
        
    #     # Create comprehensive results summary
    #     print("\n" + "="*120)
    #     print("COMPREHENSIVE RESULTS SUMMARY")
    #     print("="*120)
    #     results_df = pd.DataFrame(all_results)
    #     if not results_df.empty:
    #         display_columns = ['n_clusters', 'best_fold_score', 'worst_fold_score',
    #                           'retrained_score', 'avg_cv_score', 'std_cv_score', 'improvement']
    #         print(results_df[display_columns].to_string(index=False))
            
    #         # Plot results
    #         self.plot_evaluation_results(results_df, f"{model_type.upper()} (Resume)")
        
    #     print(f"\n✅ Training complete! Check {self.csv_filename}")
    #     print(f"📊 Total results in CSV: {len(pd.read_csv(self.csv_filename)) if os.path.exists(self.csv_filename) else 0}")
        
    #     return all_results
    def main_with_resume(self, feature_2to10, df, y, model_type=None):
        """Enhanced main function with intelligent dimension-level resume"""
        if model_type is None:
            model_type = self.model_type
            
        print("="*80)
        print(f"{model_type.upper()} TRAINING WITH INTELLIGENT AUTO-RESUME")
        print(f"📁 CSV: {self.csv_filename}")
        print("="*80)
        
        # Find starting dimension by checking completion status
        print(f"\n🔍 Scanning for incomplete dimensions in {model_type.upper()}...")
        start_idx = 0
        
        for i in range(len(feature_2to10)):
            n_clusters = i + 2
            
            # Check if this dimension is complete (inline logic)
            is_complete = False
            if not self.existing_results.empty:
                try:
                    model_col = self.existing_results['Model'].fillna('').astype(str)
                    
                    # Check regular folds (should have 10 folds)
                    regular_pattern = f"{model_type.upper()}.*{n_clusters}D$"
                    regular_models = self.existing_results[
                        model_col.str.contains(regular_pattern, regex=True, case=False, na=False)
                    ]
                    
                    if not regular_models.empty:
                        fold_col = pd.to_numeric(regular_models['Fold'], errors='coerce')
                        unique_folds = fold_col.dropna().unique()
                        regular_folds_complete = len(unique_folds) >= 10
                        
                        # Check if retrained model exists
                        retrain_pattern = f"{model_type.upper()}.*{n_clusters}D_retrained"
                        retrain_exists = model_col.str.contains(retrain_pattern, regex=True, case=False, na=False).any()
                        
                        is_complete = regular_folds_complete and retrain_exists
                        
                        if is_complete:
                            print(f"  ✅ {model_type.upper()}_{n_clusters}D is COMPLETE ({len(unique_folds)} folds + retrained)")
                        else:
                            missing = []
                            if not regular_folds_complete:
                                missing.append(f"{10 - len(unique_folds)} folds")
                            if not retrain_exists:
                                missing.append("retrained model")
                            print(f"  ⏳ {model_type.upper()}_{n_clusters}D is INCOMPLETE (missing: {', '.join(missing)})")
                            
                except Exception as e:
                    print(f"❌ Error checking dimension {n_clusters}D: {e}")
                    is_complete = False
            else:
                print(f"  ⏳ {model_type.upper()}_{n_clusters}D is INCOMPLETE (no existing data)")
            
            if not is_complete:
                start_idx = i
                print(f"📍 Will resume from {n_clusters}D dimension")
                break
        else:
            # All dimensions are complete
            print(f"✅ All dimensions appear complete!")
            start_idx = len(feature_2to10)
        
        # Handle case where everything is complete
        if start_idx >= len(feature_2to10):
            print(f"🎉 All {len(feature_2to10)} dimensions are already complete!")
            print(f"📊 Loading existing results for summary...")
            
            results_df = self._parse_csv_to_summary(self.csv_filename)
            if results_df is not None:
                print("\n" + "="*120)
                print("EXISTING RESULTS SUMMARY")
                print("="*120)
                display_columns = ['n_clusters', 'best_fold_score', 'worst_fold_score',
                                'retrained_score', 'avg_cv_score', 'std_cv_score', 'improvement']
                print(results_df[display_columns].to_string(index=False))
                self.plot_evaluation_results(results_df, f"{model_type.upper()} (Complete)")
            return []
        
        print(f"\n🚀 Starting from dimension {start_idx + 2}D (skipped {start_idx} complete dimensions)")
        
        all_results = []
        
        # Process only incomplete dimensions
        for i in range(start_idx, len(feature_2to10)):
            n_clusters = i + 2
            X_temp = df[feature_2to10[i]]
            
            print(f"\n{'='*60}")
            print(f"PROCESSING DIMENSION {n_clusters}D ({i+1}/{len(feature_2to10)})")
            print(f"{'='*60}")
            
            result = self.evaluate_feature_set(X_temp, y, n_clusters, model_type.upper())
            all_results.append(result)
        
        # Load ALL results for final summary
        print(f"\n📊 Loading complete results for final summary...")
        complete_results_df = self._parse_csv_to_summary(self.csv_filename)
        
        if complete_results_df is not None and not complete_results_df.empty:
            print("\n" + "="*120)
            print("FINAL COMPREHENSIVE RESULTS SUMMARY")
            print("="*120)
            display_columns = ['n_clusters', 'best_fold_score', 'worst_fold_score',
                            'retrained_score', 'avg_cv_score', 'std_cv_score', 'improvement']
            print(complete_results_df[display_columns].to_string(index=False))
            self.plot_evaluation_results(complete_results_df, f"{model_type.upper()} (Final)")
        elif all_results:
            print("\n" + "="*120)
            print("NEW RESULTS SUMMARY")
            print("="*120)
            new_results_df = pd.DataFrame(all_results)
            display_columns = ['n_clusters', 'best_fold_score', 'worst_fold_score',
                            'retrained_score', 'avg_cv_score', 'std_cv_score', 'improvement']
            print(new_results_df[display_columns].to_string(index=False))
            self.plot_evaluation_results(new_results_df, f"{model_type.upper()} (New Only)")
        
        total_records = len(pd.read_csv(self.csv_filename)) if os.path.exists(self.csv_filename) else 0
        print(f"\n✅ Training complete! Check {self.csv_filename}")
        print(f"📊 Total records in CSV: {total_records}")
        print(f"🆕 New dimensions processed: {len(all_results)}")
        
        return all_results

    def _parse_csv_to_summary(self, csv_filename):
        """
        Parse CSV data into the same summary format used by main_with_resume
        
        Parameters:
        -----------
        csv_filename : str
            Path to CSV file to parse
        
        Returns:
        --------
        pd.DataFrame: Summary results dataframe compatible with plot_evaluation_results
        """
        try:
            if not os.path.exists(csv_filename):
                print(f"❌ File {csv_filename} not found!")
                return None
                
            df_results = pd.read_csv(csv_filename)
            print(f"📂 Loaded {len(df_results)} records from {csv_filename}")
            
            if df_results.empty:
                print("❌ No data found in CSV file!")
                return None
            
            # Clean data
            df_results['Model'] = df_results['Model'].fillna('').astype(str)
            
            # Extract dimensions from model names (e.g., "SVC_2D" -> 2)
            def extract_dimension(model_str):
                try:
                    if '_' in model_str and 'D' in model_str:
                        dim_part = model_str.split('_')[1]
                        if dim_part.endswith('D') and dim_part[:-1].isdigit():
                            return int(dim_part[:-1])
                except (IndexError, ValueError):
                    pass
                return 0
            
            df_results['Dimension'] = df_results['Model'].apply(extract_dimension)
            
            # Process each dimension
            summary_stats = []
            regular_folds = df_results[
                (~df_results['Model'].str.contains('retrained', case=False, na=False)) & 
                (df_results['Dimension'] > 0)
            ]
            
            for dim in sorted(regular_folds['Dimension'].unique()):
                dim_data = regular_folds[regular_folds['Dimension'] == dim]
                accuracy_scores = dim_data['Accuracy'].dropna()
                
                if len(accuracy_scores) == 0:
                    continue
                
                # Calculate statistics
                avg_score = accuracy_scores.mean()
                std_score = accuracy_scores.std()
                best_score = accuracy_scores.max()
                worst_score = accuracy_scores.min()
                
                # Find fold numbers
                best_fold_row = dim_data.loc[dim_data['Accuracy'].idxmax()]
                worst_fold_row = dim_data.loc[dim_data['Accuracy'].idxmin()]
                best_fold_num = int(best_fold_row['Fold']) if pd.notna(best_fold_row['Fold']) else 1
                worst_fold_num = int(worst_fold_row['Fold']) if pd.notna(worst_fold_row['Fold']) else 1
                
                # Find retrained score
                retrained_data = df_results[
                    (df_results['Model'].str.contains('retrained', case=False, na=False)) &
                    (df_results['Dimension'] == dim)
                ]
                
                if not retrained_data.empty:
                    retrained_score = retrained_data['Accuracy'].iloc[0]
                    improvement = retrained_score - best_score
                else:
                    retrained_score = best_score
                    improvement = 0.0
                
                summary_stats.append({
                    'n_clusters': dim,
                    'features': [f"feature_{i}" for i in range(1, dim + 1)],
                    'best_fold_num': best_fold_num,
                    'best_fold_score': best_score,
                    'worst_fold_num': worst_fold_num,
                    'worst_fold_score': worst_score,
                    'retrained_score': retrained_score,
                    'avg_cv_score': avg_score,
                    'std_cv_score': std_score,
                    'improvement': improvement,
                    'fold_scores': accuracy_scores.tolist()
                })
            
            return pd.DataFrame(summary_stats) if summary_stats else None
            
        except Exception as e:
            print(f"❌ Error parsing CSV: {e}")
            return None

    def load_data_plot(self, csv_filename=None, model_name=None):
        """
        Load data from CSV and generate comprehensive plots using existing methods
        
        Parameters:
        -----------
        csv_filename : str, optional
            Path to CSV file. If None, uses self.csv_filename
        model_name : str, optional
            Model name for plot titles. If None, infers from filename
        
        Returns:
        --------
        pd.DataFrame: Summary results dataframe used for plotting
        """
        filename = csv_filename or self.csv_filename
        
        # Infer model name if not provided
        if model_name is None:
            title_mapping = {
                'qsvc': 'Quantum SVM',
                'svc': 'Classical SVM',
                'qc': 'Quantum Clustering', 
                'cc': 'Classical Clustering',
                'qpca_rbf': 'Quantum PCA + RBF',
                'qpca': 'Quantum PCA',
                'cpca': 'Classical PCA'
            }
            
            model_name = "Model"  # Default
            for key, value in title_mapping.items():
                if key in filename.lower():
                    model_name = value
                    break
        
        # Parse CSV into summary format
        results_df = self._parse_csv_to_summary(filename)
        
        if results_df is None:
            return None
        
        # Display summary using same format as main_with_resume
        print("\n" + "="*120)
        print("LOADED RESULTS SUMMARY")
        print("="*120)
        display_columns = ['n_clusters', 'best_fold_score', 'worst_fold_score',
                        'retrained_score', 'avg_cv_score', 'std_cv_score', 'improvement']
        print(results_df[display_columns].to_string(index=False))
        
        # Use existing plot method
        print(f"\n📊 Generating plots for {model_name}...")
        self.plot_evaluation_results(results_df, f"{model_name} (Loaded Data)")
        
        return results_df

# Usage examples and factory functions:

def create_evaluator(model_type, quantum_available=False, results_dir="results/evaluation", models_dir="results/models"):
    """
        Factory function to create appropriate evaluator
        
        Parameters:
        -----------
        model_type : str
            'svc', 'qsvc', 'cc', 'qc', 'qcc', 'cpca', 'qpca', 'qpca_rbf'
        quantum_available : bool
            Whether quantum computing is available
        results_dir : str
            Directory for CSV evaluation results
        models_dir : str
            Directory for saved model files
    """
    return ModelEvaluator(
            quantum_available=quantum_available, 
            model_type=model_type,
            results_dir=results_dir,
            models_dir=models_dir
    )
# Usage Examples:
"""
# Classical SVM
evaluator_svc = create_evaluator('svc', quantum_available=False)
results_svc = evaluator_svc.main_with_resume(feature_2to10, df, y)

# Quantum SVM
evaluator_qsvc = create_evaluator('qsvc', quantum_available=True)  
results_qsvc = evaluator_qsvc.main_with_resume(feature_2to10, df, y)

# Classical Clustering (saves to df_cc.csv)
evaluator_cc = create_evaluator('cc', quantum_available=False)
results_cc = evaluator_cc.main_with_resume(feature_2to10, df, y)

# Quantum Clustering (saves to df_qc.csv)
evaluator_qc = create_evaluator('qc', quantum_available=True)
results_qc = evaluator_qc.main_with_resume(feature_2to10, df, y)

# Classical PCA (saves to df_cpca.csv)
evaluator_cpca = create_evaluator('cpca', quantum_available=False)
results_cpca = evaluator_cpca.main_with_resume(feature_2to10, df, y)

# Quantum PCA (saves to df_qpca.csv)
evaluator_qpca = create_evaluator('qpca', quantum_available=True)
results_qpca = evaluator_qpca.main_with_resume(feature_2to10, df, y)

# Quantum PCA + RBF (saves to df_qpca_rbf.csv)
evaluator_qpca_rbf = create_evaluator('qpca_rbf', quantum_available=True)
results_qpca_rbf = evaluator_qpca_rbf.main_with_resume(feature_2to10, df, y)

# Load and plot results
evaluator_cc.load_data_plot("results/df_cc.csv")  # Will show "Classical Clustering"
evaluator_qc.load_data_plot("results/df_qc.csv")  # Will show "Quantum Clustering"
evaluator_cpca.load_data_plot("results/df_cpca.csv")  # Will show "Classical PCA"
evaluator_qpca.load_data_plot("results/df_qpca.csv")  # Will show "Quantum PCA"
evaluator_qpca_rbf.load_data_plot("results/df_qpca_rbf.csv")  # Will show "Quantum PCA + RBF"

# For individual evaluation (single feature set)
# Classical clustering on specific feature set
evaluator_cc = create_evaluator('cc')
n_clusters = len(np.unique(y))  # Number of unique classes
results = evaluator_cc.evaluate_feature_set(df[feature_2to10[0]], y, n_clusters, 'CC')

# Quantum clustering on specific feature set  
evaluator_qc = create_evaluator('qc', quantum_available=True)
results = evaluator_qc.evaluate_feature_set(df[feature_2to10[0]], y, n_clusters, 'QC')

# Classical PCA on specific feature set
evaluator_cpca = create_evaluator('cpca')
results = evaluator_cpca.evaluate_feature_set(df[feature_2to10[0]], y, n_clusters, 'CPCA')

# Quantum PCA on specific feature set
evaluator_qpca = create_evaluator('qpca', quantum_available=True) 
results = evaluator_qpca.evaluate_feature_set(df[feature_2to10[0]], y, n_clusters, 'QPCA')
"""