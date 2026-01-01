from preprocessing import MinimalDataProcessor
from models import create_evaluator_small
import pandas as pd
import os

from preprocessing import MinimalDataProcessor
from models import ModelEvaluator, create_evaluator

# Load dataset
# dataset_path = "/Users/sthefaniepasso/.cache/kagglehub/datasets/saurabhshahane/classification-of-malwares/versions/1"
dataset_path = "/home/ats852/.cache/kagglehub/datasets/saurabhshahane/classification-of-malwares/versions/1"

files = os.listdir(dataset_path)
csv_files = [f for f in files if f.endswith('.csv')]
if csv_files:
    df = pd.read_csv(os.path.join(dataset_path, csv_files[0]))
    print("CSV found and loaded")
else:
    print("No CSV file found")
    exit()

target = 'class'
y = df[target]
X = df.drop(columns=[target])

# PREPROCESSING
processor = MinimalDataProcessor(
    dataset_path=dataset_path,
    target_col='class'
)
feature_2to10 = processor.run_all()
feature_2to10 = feature_2to10[:9]

# SMALL DATASET EXPERIMENTS
models = ["qpca_rbf"]
percentages = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]  # 10%, 20%, 30%, 40%, 50%

for model_type in models:
    print(f"\n{'='*80}")
    print(f"RUNNING {model_type.upper()} EXPERIMENTS")
    print(f"{'='*80}")
    
    evaluator_small = create_evaluator_small(
        model_type,
        quantum_available=True,
        results_dir="../results/evaluation",
        models_dir="../results/models"
    )
    
    results = evaluator_small.run_small_experiments(
        feature_2to10=feature_2to10,
        df=df,
        y=y,
        percentages=percentages,
        min_features=2,
        max_features=10,
        model_type=model_type
    )

print("\n✅ All experiments completed!")