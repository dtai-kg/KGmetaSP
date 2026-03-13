import os
import pprint
import random
import openml
import signal

from test_pipeline import test_pipeline
from build_pipeline import *
from extract_pipeline_dict import get_pipeline_dict
from utils import append_id_to_file, read_id_file, read_json, save_json
from sklearn_param_map import *

def assert_pipelines(experiments_machine):
    """Function that executes all pipelines using their original dataset to assess their validity
    as independent pipelines for a dataset"""

    save_decisions = True

    # Load files
    if experiments_machine == "local":
        dataset2task_path = "data/dataset2task.json"
        task2run_path = "data/task2run.json"
        run2task_path = "data/run2task.json"
        run2flow_path = "data/run2flow.json"
        buggy_runs_path = "data/buggy_run_dicts.txt"
        faulty_runs_path = "data/faulty_runs.txt"
        valid_runs_path = "data/valid_runs.txt"
        valid_flows_path = "data/valid_flows.json"
        invalid_flows_path = "data/invalid_flows.json"
    elif experiments_machine == "apollo":
        dataset2task_path = "/apollo/ida/Bosch/ml-dataset-recommendation/evaluations/data/dataset2task.json"
        task2run_path = "/apollo/ida/Bosch/ml-dataset-recommendation/evaluations/data/task2run.json"
        run2task_path = "/apollo/ida/Bosch/ml-dataset-recommendation/evaluations/data/run2task.json"
        run2flow_path = "/apollo/ida/Bosch/ml-dataset-recommendation/evaluations/data/run2flow.json"
        buggy_runs_path = "/apollo/ida/Bosch/ml-dataset-recommendation/evaluations/data/buggy_run_dicts.txt"
        faulty_runs_path = "/apollo/ida/Bosch/ml-dataset-recommendation/evaluations/data/faulty_runs.txt"
        valid_runs_path = "/apollo/ida/Bosch/ml-dataset-recommendation/evaluations/data/valid_runs.txt"

    dataset2task = read_json(dataset2task_path)
    task2run = read_json(task2run_path)
    run2flow = read_json(run2flow_path)
    run2task = read_json(run2task_path)
    buggy_run_dicts = read_id_file(buggy_runs_path)  
    faulty_runs = read_id_file(faulty_runs_path)
    valid_runs = read_id_file(valid_runs_path)
    invalid_flows = read_json(invalid_flows_path)
    valid_flows = read_json(valid_flows_path)

    # print(len(valid_runs))
    # print(len(faulty_runs))
    # print((len(faulty_runs)) / (len(faulty_runs)+len(valid_runs)))

    # return 

    datasets = sorted([dataset for dataset in dataset2task], key = lambda dataset: int(dataset))
    
    #Loop through all benchmark datasets
    for dataset_id in datasets: 
        dataset = None

        #Loop through all available flows
        tasks = dataset2task[str(dataset_id)]
        dataset_flows = []
        for task in tasks:
            for run in task2run[str(task)]:
                flow = run2flow[str(run)]
                if flow not in dataset_flows:
                    dataset_flows.append(flow)

        for flow_id in dataset_flows: 
            #flow = openml.flows.get_flow(flow_id)
            print(f"\nDataset: {dataset_id}, Flow: {flow_id}")

            #Test a random run per flow
            run_id = select_random_run(flow_id, tasks, experiments_machine)
            task_id = run2task[str(run_id)]

            if run_id is None or run_id in faulty_runs or run_id in valid_runs or run_id in buggy_run_dicts: 
                print("Run already assessed. Skipping...")
                continue
            elif str(task_id) in valid_flows and flow_id in valid_flows[str(task_id)]:
                print("Flow already assessed as valid. Skipping...")
                continue
            elif str(task_id) in invalid_flows and flow_id in invalid_flows[str(task_id)]:
                print("Flow already assessed as invalid. Skipping...")
                continue

            print(f"Investigating run {run_id}")
            if not dataset:
                dataset = openml.datasets.get_dataset(dataset_id, 
                    download_qualities=False, download_data=False, 
                    download_features_meta_data=False, download_all_files=False)
                target = dataset.default_target_attribute
                X, y, _, _ = dataset.get_data(dataset_format="dataframe")

            run = openml.runs.get_run(int(run_id))
            task = openml.tasks.get_task(task_id)
            print("Task evaluation measure:", task.evaluation_measure)
            if task.evaluation_measure == "mean_absolute_error":
                task_type = "regression"
            else:
                task_type = "classification"
            
            #Generate run dictionary
            run_dict = get_pipeline_dict(run)
            print("\nRun dict:")
            pprint.pprint(run_dict)
            print("\nTask estimation procedure:")
            pprint.pprint(task.estimation_procedure)

            track_outdated_params(run_dict)

            #Test run on dataset
            #Save corresponding flow as valid or invalid
            scores = test_pipeline(X, y, target, run_dict, task.estimation_procedure, task_type, run_id)
            if save_decisions:
                if scores is None:
                    print("Faulty run found. Saving ID to falty runs...")
                    if run_id not in faulty_runs:
                        faulty_runs.append(run_id)
                        append_id_to_file(faulty_runs_path, run_id)
                    if str(task_id) not in invalid_flows: invalid_flows[str(task_id)] = []
                    invalid_flows[str(task_id)].append(flow_id)
                    save_json(invalid_flows_path, invalid_flows)
                else:
                    print("Saving ID to valid runs...")
                    print("Scores:", scores)
                    if run_id not in valid_runs:
                        valid_runs.append(run_id)
                        append_id_to_file(valid_runs_path, run_id + " Accuracy: " + str(scores["test_r2"]))
                    if str(task_id) not in valid_flows: valid_flows[str(task_id)] = []
                    valid_flows[str(task_id)].append(flow_id)
                    save_json(valid_flows_path, valid_flows)
            # return
            #break
        #break

    return

def select_random_run(flow_id, 
                      task_ids, 
                      experiments_machine, 
                      check_if_valid=False, 
                      faulty_runs=None):
    """Function that selects a random run from a flow, given exekg flows with multiple runs"""

    if experiments_machine == "local": root_dir = 'openml_exekgs/exekgs'
    elif experiments_machine == "apollo": root_dir = '/apollo/ida/Bosch/ml-dataset-recommendation/openml_exekgs/exekgs'
    subdir_name = "flow_" + str(flow_id) 
    task_subdirs = ["task_" + str(task_id) for task_id in task_ids]
    flow_paths = []

    for dirpath, dirnames, filenames in os.walk(root_dir):
        if subdir_name in dirnames:
            for task_subdir in task_subdirs:
                if task_subdir + "/" in os.path.join(dirpath, subdir_name):
                    flow_paths.append(os.path.join(dirpath, subdir_name))
    
    if len(flow_paths) == 0: return None

    random.seed(123)

    random_flow = random.choice(flow_paths)
    run_files = os.listdir(random_flow)
    random_run = random.choice(run_files)[4:-4]
    if check_if_valid == False: return random_run

    # for i in range(1, len(run_files)):
    #     if str(random_run) in faulty_runs:         
    #         random_run = run_files[i][4:-4]
    #         print("Retrying:", random_run)
    #     else: return random_run

    return None

def track_outdated_params(run_dict):
    """Recursive function that searches for and identifies outdated sklearn parameters and components"""

    outdated_params_path = "data/outdated_sklearn_params.json"
    if os.path.exists(outdated_params_path):
        outdated_params = read_json(outdated_params_path)
    else: outdated_params = {}

    numpy_params_path = "data/numpy_params.json"
    if os.path.exists(numpy_params_path):
        numpy_params = read_json(numpy_params_path)
    else: numpy_params = {}

    # for module in run_dict:
    #     print(module)

    for module in run_dict:
        #print(module)
        params = module["params"]
        if params is None: continue

        component = module["component"].split('.')[-1]
        if component == "Imputer": component = "SimpleImputer"
        if (component == "ConditionalImputer") or (
            component == "ConditionalImputer2") or (
            component == "ColumnTransformer"
            ): continue

        component_cls = globals()[component]
        allowed_params = list(component_cls().get_params().keys())
        
        for param_key in params:
            if param_key not in allowed_params:
                if component in outdated_params:
                    if param_key not in outdated_params[component]:
                        outdated_params[component].append(param_key)
                else:
                    outdated_params[component] = [param_key]
                print(f"\nFound invalid parameter {param_key} for component {component}")

            if isinstance(params[param_key], dict):
                if "value" in params[param_key]:
                    if param_key in numpy_params:
                        if params[param_key]["value"] not in numpy_params[param_key]:
                            numpy_params[param_key].append(params[param_key]["value"])
                    else: 
                        numpy_params[param_key] = [params[param_key]["value"]]
                else:
                    if component == "SVC" and param_key == "params": continue
                    elif component == "SVR" and param_key == "params": continue
                    sys.exit(f"Unseen type of dict param found: {param_key}")
                    


        track_outdated_params(module["steps"])
    
    #save_json(outdated_params_path, outdated_params)
    save_json(numpy_params_path, numpy_params)
    
    return

def find_invalid_flows():

    invalid_flows = {}
    faulty_runs_path = "data/faulty_runs.txt"
    invalid_flows_path = "data/invalid_flows.json"
    run2task_path = "data/run2task.json"
    run2flow_path = "data/run2flow.json"

    run2task = read_json(run2task_path)
    run2flow = read_json(run2flow_path)
    faulty_runs = read_id_file(faulty_runs_path)

    for run_id in faulty_runs:
        task_id = run2task[str(run_id)]
        flow_id = run2flow[str(run_id)]

        if task_id not in invalid_flows: invalid_flows[task_id] = [flow_id]
        else: invalid_flows[task_id].append(flow_id)
        
    save_json(invalid_flows_path, invalid_flows)
    return

def find_valid_flows():
    """Task that identifies valid flows given identified valid runs"""

    valid_flows = {}
    valid_runs_path = "data/valid_runs.txt"
    valid_flows_path = "data/valid_flows.json"
    run2task_path = "data/run2task.json"
    run2flow_path = "data/run2flow.json"

    run2task = read_json(run2task_path)
    run2flow = read_json(run2flow_path)
    valid_runs = read_id_file(valid_runs_path)

    for run_id in valid_runs:
        task_id = run2task[str(run_id)]
        flow_id = run2flow[str(run_id)]

        if task_id not in valid_flows: valid_flows[task_id] = [flow_id]
        else: valid_flows[task_id].append(flow_id)
        
    save_json(valid_flows_path, valid_flows)
    return

def find_regression_datasets():
    """Function that searches for and identifies regression datasets based on associated tasks' type"""

    valid_runs_path = "data/valid_dataset_runs.json"
    run2task_path = "data/run2task.json"
    dataset2task_path = "data/dataset2task.json"
    task2run_path = "data/task2run.json"

    valid_runs = read_json(valid_runs_path)
    run2task = read_json(run2task_path)
    dataset2task = read_json(dataset2task_path)
    task2run = read_json(task2run_path)

    regression_datasets = []
    for dataset_id in dataset2task:
        print(dataset_id)
        for task_id in dataset2task[dataset_id]:

            task = openml.tasks.get_task(task_id)
            if task.evaluation_measure == "mean_absolute_error":
                for run_id in task2run[str(task_id)]:
                    score = debug_run(run_id)
                    if score:
                        run = openml.runs.get_run(run_id)
                        regression_datasets.append((dataset_id, task_id, run.flow_id, run_id))

    print(regression_datasets)

    return



def debug_run(run_id):
    """Debugging function for testing individual runs with different datasets and observing results"""

    # run_id = 10591783 #regression 2
    #run_id = 9917419 #regression test
    #run_id = 9201431 #bin class
    #run_id = 8948421 #multi class
    #run_id = 1860379 #holdout_test
    #run_id = 2081275 #not responding with dataset 18
    #run_id = 2012955 #openml api not responding
    # run_id = 10592575 #Bagging classifier
    print(f"Testing run: {run_id}")
        
    run = openml.runs.get_run(run_id)
    signal.alarm(0)
    print(run)
    task = openml.tasks.get_task(run.task_id)

    dataset = openml.datasets.get_dataset(23, 
                download_qualities=False, download_data=False, 
                download_features_meta_data=False, download_all_files=False)
    target = dataset.default_target_attribute
    X, y, _, _ = dataset.get_data(dataset_format="dataframe")
    print(X, y)

    run_dict = get_pipeline_dict(run)

    print("\nRun dict:")
    pprint.pprint(run_dict)
    print("\nTask estimation procedure:")
    pprint.pprint(task.estimation_procedure)
    print(task.evaluation_measure)

    #track_outdated_params(run_dict)

    if task.evaluation_measure == "mean_absolute_error":
        task_type = "regression"
        score = test_pipeline(X, y, target, run_dict, task.estimation_procedure, task_type, run_id)
    else: 
        task_type = "classification"
        score = test_pipeline(X, y, target, run_dict, task.estimation_procedure, task_type, run_id)
            
    #print(float(sum(score["test_precision"]))/float(len(score["test_precision"])))
    print("Scores:", score)

    return score


if __name__ == "__main__":

    experiments_machine = "local"
    #assert_pipelines(experiments_machine)
    find_regression_datasets()
    # debug_run()

    
