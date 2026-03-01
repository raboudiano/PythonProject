import mlflow
from mlflow.entities import Run

mlflow.set_tracking_uri('sqlite:///mlruns.db')
exp = mlflow.get_experiment_by_name('Cirrhosis_Prediction')
if exp:
    runs = mlflow.search_runs(experiment_ids=[exp.experiment_id])
    print(f'Total runs: {len(runs)}\n')
    for i, run in enumerate(runs):
        if isinstance(run, Run):
            print(f'Run {i+1}:')
            print(f'  Run ID: {run.run_id}')
            print(f'  Tags: {dict(run.data.tags) if run.data.tags else "No tags"}')
            test_acc = run.data.metrics.get('test_accuracy', 'N/A')
            test_f1 = run.data.metrics.get('test_f1', 'N/A')
            print(f'  Test Accuracy: {test_acc}')
            print(f'  F1 Score: {test_f1}')
            print()
        else:
            print(f'Run {i+1}: Type is {type(run).__name__}')
            print(f'  Value: {run}')
            print()

