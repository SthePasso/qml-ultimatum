from preprocessing import MinimalDataProcessor
from models import ModelEvaluator, create_evaluator


import pandas as pd
import os

# Define the dataset path
# dataset_path = "/Users/sthefaniepasso/.cache/kagglehub/datasets/saurabhshahane/classification-of-malwares/versions/1"
dataset_path = "/home/ats852/.cache/kagglehub/datasets/saurabhshahane/classification-of-malwares/versions/1"
files = os.listdir(dataset_path)
csv_files = [f for f in files if f.endswith('.csv')]
if csv_files:
    df = pd.read_csv(os.path.join(dataset_path, csv_files[0]))
    print("CSV found and send to df")  # Display first few rows
else:
    print("No CSV file found in the dataset directory.")
target = 'class'
y = df[target]
X = df.drop(columns=[target])

# PREPROCESSING*************************************************
processor = MinimalDataProcessor(
    dataset_path=dataset_path,
    target_col='class'
)
feature_2to10 = processor.run_all()
feature_2to10 = feature_2to10[:9]

# MODEL TRAIN*************************************************
# models = ["svc", "qsvc", "cc", "qc", "qcc", "cpca", "qpca", "qpca_rbf"]

models = ["svc", "qsvc", "cc", "qc", "cpca", "qpca", "qpca_rbf"]

for i in range(0,len(models)-1):
    evaluator_qc = create_evaluator(
        models[i], 
        quantum_available=True,
        results_dir="../results/evaluation",  # Relative to src/ directory
        models_dir="../results/models"         # Relative to src/ directory
    )
    results_qc = evaluator_qc.main_with_resume(feature_2to10, df, y)