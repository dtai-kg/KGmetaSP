import pandas as pd
import random
import openml
import os
from collections import defaultdict
import time
from multiprocessing import Pool

from extract_pipeline_dict import get_pipeline_dict
from utils import read_json, save_json
from test_pipeline import test_pipeline


def pipeline_simulations(valid_runs, 
                         pipeline_evaluations, 
                         valid_runs_path, 
                         pipeline_evaluations_path, 
                         max_sims):
    """Function that performs pipeline simulations using OpenML datasets and runs within benchmark"""
    
    random.seed(321)
    datasets = list(valid_runs.keys())
    datasets_len = len(datasets)

    size_limit = 25
    for i in range(datasets_len):

        dataset_id1 = datasets[i]
        # Prioritize small datasets for faster evalutions
        if assess_dataset_size(dataset_id1, size_limit) == False: 
            continue
        
        #Get valid runs sample
        valid_runs, runs1 = get_runs_sample(valid_runs, str(dataset_id1), max_sims)
        #save_json(valid_runs_path, valid_runs)
        if len(runs1) == 0: 
            print("No valid runs for this dataset to simulate! Skipping...")
            continue
        dataset1 = OpenmlDataset(str(dataset_id1))

        #Perform dataset simulations with original runs for dataset 1 
        pipeline_evaluations = simulate_pipelines_for_dataset(dataset1, runs1, 
                                                              pipeline_evaluations, 
                                                              pipeline_evaluations_path)

        for j in range(i + 1, datasets_len):
            
            dataset_id2 = datasets[j]
            # Prioritize small datasets for faster evalutions
            if assess_dataset_size(dataset_id2, size_limit) == False: 
                continue

            print(f"\nSimulating runs for dataset pair {dataset_id1}, {dataset_id2}")

            #Get valid runs sample
            valid_runs, runs2 = get_runs_sample(valid_runs, str(dataset_id2), max_sims)
            #save_json(valid_runs_path, valid_runs)
            if len(runs2) == 0: 
                print("No valid runs for this dataset to simulate! Skipping...")
                continue
            dataset2 = OpenmlDataset(str(dataset_id2))

            #Perform dataset simulations with original runs for dataset 2 
            pipeline_evaluations = simulate_pipelines_for_dataset(dataset2, runs2, 
                                                              pipeline_evaluations, 
                                                              pipeline_evaluations_path)
            
            #Perform inter-dataset pipeline simulations 
            pipeline_evaluations = simulate_pipelines_for_dataset(dataset2, runs1, 
                                                              pipeline_evaluations, 
                                                              pipeline_evaluations_path)
            
            pipeline_evaluations = simulate_pipelines_for_dataset(dataset1, runs2, 
                                                              pipeline_evaluations, 
                                                              pipeline_evaluations_path)



def get_valid_runs(datasets, 
                   dataset2task, 
                   task2run, 
                   run2flow, 
                   invalid_flows):
    """Function that returns a dictionary of runs that have been found to be valid"""

    valid_runs = defaultdict(list)
    pipeline_evaluations = defaultdict(dict)
    # test_counter = 0
    # test_limit = 20

    print("Finding valid and invalid runs...")
    for dataset_id in datasets:

        #test_counter += 1
        print(f"\nInvestingating Dataset {dataset_id}...")
        #Get all dataset tasks
        tasks = dataset2task[str(dataset_id)]
        # task_dict = {str(task):[] for task in tasks}

        #Get all flows
        for task in tasks:
            task_runs = task2run[str(task)]
            #print(task_runs)
            for run in task_runs:
                flow = run2flow[str(run)]

                # Check for invalid pipelines
                if str(task) in invalid_flows and int(flow) in invalid_flows[str(task)]:
                    
                    print(f"Invalid run found: Task: {task}, Flow {flow}, Run: {run}")
                    pipeline_evaluations[str(dataset_id)][str(run)] = {"run_success": False}
                    
                else:
                    print(f"Valid run found: Task: {task}, Flow {flow}, Run: {run}")
                    valid_runs[str(dataset_id)].append(run)

        # print(valid_runs)
        # pprint.pprint(pipeline_evaluations)

        #if test_counter == test_limit: break

    return valid_runs, pipeline_evaluations

# Function to fetch the run
# def fetch_run_in_process(run_id):
#     try:
#         openml.runs.get_run(run_id)
#     except:
#         pass

def process_run(run):
    """Function that evaluates whether a run is accessible using the OpenML API"""
    try:
        openml.runs.get_run(int(run)) 
    except Exception as e:
        print(f"Run {run} cannot be retrieved by API due to: {e}")
        return run  # Return the run to indicate failure
    return None  # Return None for successful processing

def process_run_with_timeout(run, 
                             timeout):
    """Function that queries an OpenML run from OpenML API, applying a tiemout to account for API failures"""
    try:
        with Pool(processes=1) as pool:  # Use a single worker for the timeout
            result = pool.apply_async(process_run, args=(run,))
            return result.get(timeout=timeout)
    except TimeoutError:
        print(f"Run {run} timed out")
        return run  
    except Exception as e:
        print(f"Error processing run {run} {e}")
        return run  
    

def get_runs_sample(valid_runs, 
                    dataset_id, 
                    threshold):
    """Function to get random runs given a list a threshold of maximum returned number"""

    #Make sure that run can be retrieved by OpenML API for simulations
    #Else treat as invalid
    print(f"Run samples size: {len(valid_runs[dataset_id])}")
    failed_runs = []
    runs = valid_runs[dataset_id]
    timeout = 5
    for run in runs:
        result = process_run_with_timeout(int(run), timeout)
        if result is not None:  # Collect failed or timed-out runs
            failed_runs.append(result)
    

    valid_runs[dataset_id] = [run for run in valid_runs[dataset_id] if run not in failed_runs]
    print(f"Run samples after removing non-retrieved runs: {len(valid_runs[dataset_id])}")

    runs = valid_runs[dataset_id]
    random.shuffle(runs)
    runs_sample_limit = min(threshold, len(runs))
    
    return valid_runs, runs[:runs_sample_limit]


def assess_dataset_size(dataset_id, size_limit, range_min = 0):
    """Function to check if an OpenML dataset exceeds a number of features"""

    print("\nDataset:", dataset_id)
    try:
        dataset = openml.datasets.get_dataset(dataset_id)
        features = dataset.features
        size = int(list(features.keys())[-1])
        print("Size:", size)

        if size > size_limit or size <= range_min:
            print("Dataset size out of bounds.")
            return False

        print("Dataset ok size.")
        return True
    
    except:
        print("Cannot retrieve dataset. Skipping...")
        return False


def simulate_pipelines_for_dataset(dataset, 
                                   runs, 
                                   pipeline_evaluations, 
                                   pipeline_evaluations_path):
    """Function that assesses whether a dataset-run pair has already been simulated or else it can be submitted for simulation"""

    for run_id in runs:
        if dataset.id not in pipeline_evaluations:
            pipeline_evaluations[dataset.id] = {}
        
        run = openml.runs.get_run(run_id)

        if str(run_id) not in pipeline_evaluations[dataset.id] or ("LogisticRegression" in run.flow_name and (
            pipeline_evaluations[dataset.id][str(run_id)] == {"run_success": False}
        )): 
            print(f"\n\nSimulating dataset {dataset.id} with pipeline {run_id}")
            
            simulation_scores = simulate_pipeline(dataset, run, pipeline_evaluations)
            print("Simulation scores:", simulation_scores)

            if not simulation_scores:
                pipeline_evaluations[dataset.id][str(run_id)] = {"run_success": False}

            else:
                pipeline_evaluations[dataset.id][str(run_id)] = {"run_success": True, 
                                                              "run_metrics": simulation_scores}
    
            save_json(pipeline_evaluations_path, pipeline_evaluations)
    return pipeline_evaluations


def simulate_pipeline(dataset, 
                      run, 
                      pipeline_evaluations):

    """Function that simulates a dataset-run pair and extracts evaluation results"""

    #Get run openml object
    try:
        #run = openml.runs.get_run(run_id)
        run_dict = get_pipeline_dict(run)
    except:
        print("Unable to extract pipeline dict!")
        return None

    #Calculate a max timeout for the simulation
    sim_timeout = get_sim_max_timeout(dataset.id, run, pipeline_evaluations)
    print(f"Sim_timeout: {sim_timeout}")

    #Get task openml object
    task = openml.tasks.get_task(run.task_id)
    eval_meas = task.evaluation_measure
    task_type = "regression" if eval_meas == "mean_absolute_error" else "classification"

    #Perform simulation
    try:
        scores = test_pipeline(dataset.X, dataset.y, dataset.target, run_dict, 
                            task.estimation_procedure, task_type, run.id, sim_timeout)
        return scores
    except:
        return None
    
    

def get_sim_max_timeout(dataset_id, 
                        run, 
                        pipeline_evaluations):
    """Function that generates a maximum timeout for a simulation, based on original fit time of run in OpenML for the 
    dataset it was designed for"""


    #If simulation takes place with the dataset for which it was created, 
    # impose no timeout
    init_dataset_id = run.dataset_id
    if int(dataset_id) == init_dataset_id: return 60*15

    min_timeout = 200 #10 folds * 10 times allowed overtime * 2 for other fit_model function methods
    timeout_multiplier = 250 #10 folds * 10 times allowed overtime * 2.5 for other fit_model function methods
    #timeout_lim = 60*15

    try:
        init_fit_time = pipeline_evaluations[str(init_dataset_id)][str(run.id)]["run_metrics"]["fit_time"]
        max_timeout = max(min_timeout, init_fit_time * timeout_multiplier)
        #timeout = min(max_timeout, timeout_lim)
        return max_timeout
    except:
        return 1

def main():

    openml.config.apikey = 'eee9181dd538cb1a9daac582a55efd72'

    #Load artifact relationships
    invalid_flows_path = "data/invalid_flows.json"
    dataset2task_path = "data/dataset2task.json"
    task2run_path = "data/task2run.json"
    run2flow_path = "data/run2flow.json"
    pipeline_evaluations_path = "data/pipeline_evaluations.json"
    valid_runs_path =  "data/valid_dataset_runs.json"

    
    # Check if we already have found all valid runs per dataset
    if not os.path.exists(valid_runs_path) or not os.path.exists(pipeline_evaluations_path):

        #Get all datasets
        #Then we don't need the excel anymore, since every
        #dataset is paired with all the datasets with a greater ID
        similarities_path = "openml_exekgs/similarities_full.csv"
        similarities_df = pd.read_csv(similarities_path)
        datasets = similarities_df["Dataset 1"].unique().astype(int)
        del similarities_df

        invalid_flows = read_json(invalid_flows_path)
        dataset2task = read_json(dataset2task_path)
        task2run = read_json(task2run_path)
        run2flow = read_json(run2flow_path)
        valid_runs, pipeline_evaluations = get_valid_runs(datasets, dataset2task, task2run, 
                                                          run2flow, invalid_flows)
        
        save_json(valid_runs_path, valid_runs)
        #save_json(pipeline_evaluations_path, pipeline_evaluations)

    else:
        valid_runs = read_json(valid_runs_path)
        pipeline_evaluations = read_json(pipeline_evaluations_path)

    max_sims = 10
    pipeline_simulations(valid_runs, pipeline_evaluations, valid_runs_path, pipeline_evaluations_path, max_sims)

#OpenML Dataset Class
class OpenmlDataset:

    def __init__(self, 
                 dataset_id):

        openml_object = openml.datasets.get_dataset(dataset_id, 
                                download_qualities=False, download_data=True, 
                                download_features_meta_data=False, download_all_files=False)
        
        self.id = dataset_id
        self.target = openml_object.default_target_attribute
        X, y, _, _ = openml_object.get_data(dataset_format="dataframe")
        self.X = X
        self.y = y

if __name__ == "__main__":

    main()