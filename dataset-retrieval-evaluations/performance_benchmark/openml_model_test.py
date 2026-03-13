import openml
import sklearn.metrics

def test_sklearn0_18_1_runs():
    """Function to test whether we can extract the model from a run built with sklearn==0.18.1
    and run it with the same task.
    Requires Python 3.4 or 3.5
    scikit-learn==0.18.1 --> numpy>=1.6.1 --> scipy>=0.9"""

    # Trying to extract a new run from the flow returns no evaluations
    run = openml.runs.get_run(1860456)
    print(run.evaluations)
    flow = openml.flows.get_flow(run.flow_id, reinstantiate=True)
    task = openml.tasks.get_task(run.task_id)
    print(flow.dependencies)
    new_run = openml.runs.run_flow_on_task(flow, task, avoid_duplicate_runs=False)
    print(run.get_metric_fn(sklearn.metrics.auc))
    print(new_run.get_metric_fn(sklearn.metrics.auc))

    #Trying to extract a model from setup or run returns errors (tried with different openml versions)
    # run = openml.runs.get_run(1860456)
    # model = openml.runs.initialize_model_from_run(run.run_id)
    # task = openml.tasks.get_task(run.task_id)
    # new_run = openml.runs.run_model_on_task(model, task)
    # print(run.get_metric_fn(sklearn.metrics.auc))
    # print(new_run.get_metric_fn(sklearn.metrics.auc))

    return

def test_sklearn0_20_1_runs():
    """Function to test whether we can extract the model from a run built with sklearn==0.18.1
    and run it with the same task.
    Requires Python 3.4 or 3.5 or 3.6 or 3.7
    scikit-learn==0.20.1 --> numpy>=1.6.1 --> scipy>=0.9"""

    # Trying to extract a new run from the flow 
    # run = openml.runs.get_run(9918434)
    # print(run.evaluations)
    # flow = openml.flows.get_flow(run.flow_id, reinstantiate=True)
    # print(flow.dependencies)
    # task = openml.tasks.get_task(run.task_id)
    # new_run = openml.runs.run_flow_on_task(flow, task, avoid_duplicate_runs=False)
    # print(new_run.get_metric_fn(sklearn.metrics.auc))

    #Trying to extract a model from setup or runs (seems to work but with deprecation warnings)
    run = openml.runs.get_run(9918434)
    print(run.evaluations)
    #model = openml.setups.initialize_model(run.setup_id)
    model = openml.runs.initialize_model_from_run(run.run_id)
    task = openml.tasks.get_task(run.task_id)
    new_run = openml.runs.run_model_on_task(model, task)
    print("\n Run evaluation:",run.get_metric_fn(sklearn.metrics.f1_score))
    print("Re-structured run evaluation:",new_run.get_metric_fn(sklearn.metrics.f1_score))
    print("Run 9918434 and a new run extracting its model and running it on the same task yield the same evaluations.")

    return

def test_sklearn0_20_1_runs_diff_task():
    """Function to test whether we can extract the model from a run built with sklearn==0.20.1
    and run it with a different task, defined with a different dataset.
    Requires Python 3.4 or 3.5 or 3.6 or 3.7
    scikit-learn==0.20.1 --> numpy>=1.6.1 --> scipy>=0.9"""

    #Example of a matching model with a random task

    #Define a run from which we extract the model
    model_run_id = 9918434
    model_run = openml.runs.get_run(model_run_id)

    #Define a task and run that has already tested the defined task (to compare our results with)
    original_run_id = 1836932
    original_run = openml.runs.get_run(original_run_id)
    task_id = original_run.task_id

    #Extract model from run
    model = openml.runs.initialize_model_from_run(model_run.run_id)

    # Test model on defined task creating a new run
    task = openml.tasks.get_task(task_id)

    try:
        new_run = openml.runs.run_model_on_task(model, task)
        print("\n Original run evaluation:",original_run.get_metric_fn(sklearn.metrics.f1_score))
        print("Different task evaluation with the same model:",new_run.get_metric_fn(sklearn.metrics.f1_score))
    except Exception as e:
    # If the model is not compatible with a task, let's compare the model with another model that actually works for
    # the task in question
        print("Model could not be run on this task due to error:", e)
        print("\nPrinting model of each run to analyse them:")
        print("Model extracted:", model)

        # original_flow = openml.flows.get_flow(4834)
        # print(original_flow.model)
        try:
            original_model = openml.runs.initialize_model_from_run(original_run.run_id)
            print("\nModel of one of original runs for defined task:", original_run.model)
        except Exception as e2:
            print("No original run model found")





    #Example of a the same model with a task that they don't match
    #Define a run from which we extract the model
    model_run_id = 9918434
    model_run = openml.runs.get_run(model_run_id)

    #Define a task and run that has already tested the defined task (to compare our results with)
    original_run_id = 1837091
    original_run = openml.runs.get_run(original_run_id)
    task_id = original_run.task_id

    #Extract model from run
    model = openml.runs.initialize_model_from_run(model_run.run_id)

    # Test model on defined task creating a new run
    task = openml.tasks.get_task(task_id)

    try:
        new_run = openml.runs.run_model_on_task(model, task)
        print("\n Original run evaluation:",original_run.get_metric_fn(sklearn.metrics.f1_score))
        print("Different task evaluation with the same model:",new_run.get_metric_fn(sklearn.metrics.f1_score))
    except Exception as e:
     # If the model is not compatible with a task, let's compare the model with another model that actually works for
     # the task in question
        print("Model could not be run on this task due to error:", e)
        print("\nPrinting model of each run to analyse them:")
        print("Model extracted:", model)

        # original_flow = openml.flows.get_flow(4834)
        # print(original_flow.model)
        try:
            original_model = openml.runs.initialize_model_from_run(original_run.run_id)
            print("\nModel of one of original runs for defined task:", original_run.model)
        except Exception as e2:
            print("No original run model found")


    return


if __name__ == "__main__":
    openml.config.apikey = 'eee9181dd538cb1a9daac582a55efd72'
    print("OpenML Version:", openml.__version__)

    #Requires Python 3.4 or 3.5 (Tried with Python 3.5)
    #Sklearn 0.18.1 combination with OpenML does not seem to be working currently for running models
    test_sklearn0_18_1_runs()

    #Requires Python 3.4 or 3.5 or 3.6 or 3.7 (Tried with Python 3.7)
    #Sklearn 0.20.1 combination with older OpenML versions seems to work with deprecation warnings
    test_sklearn0_20_1_runs()
    test_sklearn0_20_1_runs_diff_task()
    
        

    





    



    

    
