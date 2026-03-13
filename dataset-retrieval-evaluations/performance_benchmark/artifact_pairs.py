import openml 
import os
import pandas as pd
from utils import read_json, save_json

def match_datasets_to_tasks(dataset2tasks_path, 
                            task2dataset_path):
    """Function that retrieves the task-dataset pairs from the exekg benchmark"""

    print("Matching datasets with tasks...")
    task_path = "openml_exekgs/exekgs"
    tasks = [item for item in os.listdir(task_path) if os.path.isdir(os.path.join(task_path, item))]

    task_ids = []
    task_counter = 0
    for task in tasks:
        if "task" not in task:
            continue
        
        task_id = int(task.split('_')[1])
        task_ids.append(task_id)
        task_counter += 1

    print(f"{task_counter} tasks found!")

    tlist = openml.tasks.list_tasks(size = 500000)
    task_df = pd.DataFrame.from_dict(tlist, orient="index").reset_index()[['tid', 'did']]
    task_df_filtered = task_df[task_df["tid"].isin(task_ids)]
    #print(task_df_filtered)

    non_unique_dataset_tasks = task_df_filtered[task_df_filtered['did'].duplicated(keep=False)]
    #print(non_unique_dataset_tasks)

    task_ids_series = task_df_filtered["tid"].to_list()
    dataset_ids_series = task_df_filtered["did"].to_list()
    
    task2dataset = {int(task): [] for task in task_ids_series}
    dataset2task = {int(dataset): [] for dataset in dataset_ids_series}

    for i in range(len(task_ids_series)):
        task2dataset[int(task_ids_series[i])] = dataset_ids_series[i]
        dataset2task[int(dataset_ids_series[i])].append(task_ids_series[i])

    save_json(task2dataset_path, task2dataset)
    save_json(dataset2tasks_path, dataset2task)
        
    return task2dataset, dataset2task

def match_datasets_to_runs(task2run_path, 
                           run2task_path,
                           flow2run_path, 
                           run2flow_path):
    """Function that retrieves the flow-run-dataset pairs from the exekg benchmark"""
    
    print("Matching datasets with flows...")

    task2run = {}
    run2task = {}

    run2flow = {}
    flow2run = {}

    tasks_path = "openml_exekgs/exekgs"
    tasks = [item for item in os.listdir(tasks_path) if os.path.isdir(os.path.join(tasks_path, item))]
    for task in tasks:
        if "task" not in task: continue
        task_path = os.path.join(tasks_path, task)
        task_id = int(task.split('_')[-1])

        for flow in os.listdir(task_path):
            if "flow" not in flow: continue
            flow_id = int(flow.split('_')[-1])
            flow_path = os.path.join(task_path, flow)

            for run in os.listdir(flow_path):
                if "run" not in run: continue
                run_id = int(run.split('_')[-1].split('.')[0])

                if str(task_id) not in task2run: task2run[str(task_id)] = [run_id]
                else: task2run[str(task_id)].append(run_id)
                run2task[str(run_id)] = task_id

                if str(flow_id) not in flow2run: flow2run[str(flow_id)] = [run_id]
                else: flow2run[str(flow_id)].append(run_id)
                run2flow[str(run_id)] = flow_id

    save_json(task2run_path, task2run)
    save_json(run2task_path, run2task)
    save_json(flow2run_path, flow2run)
    save_json(run2flow_path, run2flow)
    
    return

def main():

    dataset2task_path = "data/dataset2task.json"
    task2dataset_path = "data/task2dataset.json"
    task2run_path = "data/task2run.json"
    run2task_path = "data/run2task.json"
    flow2run_path = "data/flow2run.json"
    run2flow_path = "data/run2flow.json"

    #Dataset-Tasks 
    #if not os.path.exists(dataset2task_path): 
    task2dataset, dataset2task = match_datasets_to_tasks(dataset2task_path, task2dataset_path)
    #else: task2dataset, dataset2task = read_json(task2dataset_path), read_json(dataset2task_path)
         
    #Dataset-Flows
    match_datasets_to_runs(task2run_path, run2task_path,
                           flow2run_path, run2flow_path)


    return

if __name__ == "__main__":
    openml.config.apikey = 'eee9181dd538cb1a9daac582a55efd72'
    print("OpenML Version:", openml.__version__)

    main()
    