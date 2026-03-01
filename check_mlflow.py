import mlflow

mlflow.set_tracking_uri('sqlite:///mlruns.db')

print('=== MLflow Experiments ===')
experiments = mlflow.search_experiments()
for e in experiments:
    print(f'  - {e.name} (ID: {e.experiment_id})')

print('\n=== Runs in Cirrhosis_Prediction ===')
exp = mlflow.get_experiment_by_name('Cirrhosis_Prediction')
if exp:
    runs = mlflow.search_runs(experiment_ids=[exp.experiment_id])
    print(f'Total runs: {len(runs)}\n')
    
    # Get models from latest runs
    models_dict = {}
    for run in runs:
        if hasattr(run, 'data') and hasattr(run.data, 'tags'):
            model_name = run.data.tags.get('model_name', run.data.tags.get('mlflow.runName', 'Unknown'))
        else:
            model_name = 'Unknown'
        
        if hasattr(run, 'data') and hasattr(run.data, 'metrics'):
            metrics = run.data.metrics
        else:
            metrics = {}
        
        test_acc = metrics.get('test_accuracy', None)
        f1 = metrics.get('test_f1', None)
        
        if model_name != 'Unknown' and test_acc is not None:
            if model_name not in models_dict:
                models_dict[model_name] = {'accuracy': test_acc, 'f1': f1, 'count': 0}
            models_dict[model_name]['count'] += 1
    
    print('Models trained and logged to MLflow:')
    for model_name in sorted(models_dict.keys()):
        data = models_dict[model_name]
        print(f'  - {model_name}: Accuracy={data["accuracy"]:.4f}, F1={data["f1"]:.4f} ({data["count"]} runs)')
else:
    print('Experiment not found!')
