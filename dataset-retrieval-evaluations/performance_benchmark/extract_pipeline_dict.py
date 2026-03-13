import ast
import openml
from collections import OrderedDict
import pandas as pd

OPENML_PARAMS_TO_NEW_SKLEARN = {
    "min_impurity_split": "min_impurity_decrease",
    "presort": None,
}
OPENML_PARAM_VALUES_TO_SKLEARN = {
    "null": None,
    "true": True,
    "false": False,
}

OPENML_TO_SKLEARN_CLASSES = {
    "Imputer": "SimpleImputer",
    # "openmlstudy14.preprocessing.ConditionalImputer": "sklearn.impute.SimpleImputer",  # custom OpenML imputer replaced with sklearn imputer
    "ConditionalImputer": "SimpleImputer",  # custom OpenML imputer replaced with sklearn imputer
    "ConditionalImputer2": "SimpleImputer",  # custom OpenML imputer replaced with sklearn imputer
}

DETAILED_LOG = True

def snake_to_camel(text: str) -> str:
    """
    Converts camel-case string to snake-case
    Args:
    text: string to convert

    Returns:
    str: converted string"""

    return "".join(x.capitalize() for x in text.lower().split("_"))


def code_param_to_exekg_param(param_name: str) -> str:
    """
    Converts a parameter name from code format to ExeKG format.

    Args:
    param_name (str): The parameter name in code format.

    Returns:
    str: The parameter name in ExeKG format."""
  
    return f"hasParam{snake_to_camel(param_name)}"


def openml_param_value_to_exekg_param_value(param_value: str):
    """
    Converts a parameter value from OpenML format to ExeKG format.

    Args:
        param_value (str): The parameter value in OpenML format.

    Returns:
        Any: The parameter value in ExeKG format.
    """

    if param_value in OPENML_PARAM_VALUES_TO_SKLEARN:
        return OPENML_PARAM_VALUES_TO_SKLEARN[param_value.lower()]
    else:
        try:
            if "class '" in param_value:
                param_value = param_value[8:-2] #keep only datatype from numpy types
            if str(param_value)[0] == "[" and str(param_value)[-1] != "]":
                param_value += "]"
            param_value = ast.literal_eval(param_value)
            if isinstance(param_value, float) and param_value == int(param_value):
                # convert float to int if it's actually an int, to avoid issues with ExeKG validation
                param_value = int(param_value)
            return param_value
        except ValueError:
            return str(param_value)


def get_param_dict_from_openml_run(run, filter_by_component_id=None):
  param_settings = run.parameter_settings
  if filter_by_component_id is not None:
    param_settings = filter(
      lambda s: s["oml:component"] == str(filter_by_component_id), param_settings
      
    )
  

  converted_param_settings = {}
  for s in param_settings:
    param_name = s["oml:name"]
    if param_name in OPENML_PARAMS_TO_NEW_SKLEARN:
      param_name = OPENML_PARAMS_TO_NEW_SKLEARN[param_name]
      if param_name is None:
        continue
    #param_name = code_param_to_exekg_param(param_name)

    param_value = openml_param_value_to_exekg_param_value(s["oml:value"])
    if param_value is None:
      continue

    converted_param_settings[param_name] = param_value

  return converted_param_settings

def add_or_update_tasks(
    components,
    flow,
    run,
    run_dict,
    depth=1,
):
    components_ordered = OrderedDict(
        sorted(components.items(), key=lambda x: flow.name.index(x[1].class_name))
    )
    components_ordered_items = list(components_ordered.items())
 
    # add tasks to the KG based on the flow's details
    for i, (component_name, component_obj) in enumerate(components_ordered_items):
        subcomponents = component_obj.components
        if not subcomponents:  # or estimator_subcomponent is not None:
            # is a standalone step of the pipeline
            if DETAILED_LOG:
              print(
                  ("\t" * depth)
                  + f"Component {i} uses {component_obj.class_name}"
              )
 
            component_param_dict, subcomponent_param_dict = add_or_update_task(
                run,
                component_obj,
                None,
                # estimator_subcomponent,
            )

            component_dict = {}
            component_dict["component"] = component_obj.class_name
            component_dict["rank"] = i 
            component_dict["params"] = component_param_dict
            component_dict["depth"] = depth - 1
            component_dict["steps"] = []
            run_dict.append(component_dict)
            #print(f"Run dict: {run_dict}")

            continue
        
        steps_labels = []
        for label in subcomponents.keys():
            steps_labels.append(label)
        # is ensemble or model selection step that contains meta_estimator
        component_param_dict, subcomponent_param_dict = add_ensemble_or_model_selection_task(
            component_obj,
            subcomponents,
            run,
            steps_labels
        )
 
        # if test_predicted_y:
        #     continue
 
        if DETAILED_LOG:
            print(
                ("\t" * depth)
                + f"Component {i} with class name {component_obj.class_name} is a flow"
            )

        component_dict = {}
        component_dict["component"] = component_obj.class_name
        component_dict["rank"] = i 
        component_dict["depth"] = depth - 1
        component_dict["steps"] = []
        if component_param_dict is None:
            if "numeric" in subcomponents or "num" in subcomponents:
                component_dict["params"] = {"column_type": "numeric"}
            elif "nominal" in subcomponents or "categorical" in subcomponents or "cat" in subcomponents:
                component_dict["params"] = {"column_type": "nominal"}
            else:
                component_dict["params"] = None
        else: component_dict["params"] = component_param_dict
            #print(f"Run dict: {run_dict}")
 
        component_dict["steps"] = add_or_update_tasks(
            subcomponents,
            component_obj,
            run,
            component_dict["steps"],
            depth + 1,
        )
        run_dict.append(component_dict)
        

    return run_dict
 
def add_ensemble_or_model_selection_task(
    flow,
    subcomponents,
    run,
    steps_labels
):
    estimator_subcomponent = None
    if subcomponents:
        estimator_subcomponent = subcomponents.get("estimator", None)
        if estimator_subcomponent is None:
            estimator_subcomponent = subcomponents.get("base_estimator", None)

    if "sklearn.pipeline.Pipeline" in flow.name:
        pos1 = flow.name.find("numeric")
        if pos1 == -1: pos1 = flow.name.find("num")
        pos2 = flow.name.find("nominal")
        if pos2 == -1: pos2 = flow.name.find("categorical")
        if pos2 == -1: pos2 = flow.name.find("cat")

        if pos1 == -1 and pos2 == -1: pass
        elif pos1 == -1: return {"column_type": ["nominal"]}, None
        elif pos2 == -1: return {"column_type": ["numeric"]}, None
        else: return {"column_type": steps_labels}, None
        # elif pos1<pos2: return {"column_type": ["numeric","nominal"]}, None
        # elif pos1>pos2: return {"column_type": ["nominal","numeric"]}, None
 
    if (
        estimator_subcomponent is None
        or "sklearn.pipeline.Pipeline" in estimator_subcomponent.class_name + flow.name
    ):
        return None, None
 
    if "model_selection" in flow.name and estimator_subcomponent:
        if DETAILED_LOG:
            print(
                ("\t")
                + f"Flow {flow.id} uses {flow.class_name} and is a model selection method with estimator {estimator_subcomponent.class_name}"
            )
 
        component_param_dict, subcomponent_param_dict = add_or_update_task(
            run,
            flow,
            estimator_subcomponent,
        )

        return component_param_dict, subcomponent_param_dict

    elif "ensemble" in flow.name and estimator_subcomponent:
        if DETAILED_LOG:
            print(
                ("\t")
                + f"Flow {flow.id} uses {flow.class_name} and is an ensemble method with estimator {estimator_subcomponent.class_name}"
            )
 
        component_param_dict, subcomponent_param_dict = add_or_update_task(
            run,
            flow,
            estimator_subcomponent,
        )
        return component_param_dict, subcomponent_param_dict
    else:
        raise ValueError(
            f"Flow {flow.id} uses {flow.class_name} and is not a model selection or ensemble method but has a subcomponent {estimator_subcomponent.class_name}."
        )
 
 
def add_or_update_task(
    run,
    component_obj,
    subcomponent_obj=None):
    """Function for adding or updating an exekg task, also used for the generation of the pipeline dict"""

    method_params_dict = get_method_name_and_params_dict(run, component_obj)
    submethod_name = None
    submethod_params_dict = None

    if subcomponent_obj:
        submethod_params_dict = get_method_name_and_params_dict(run, subcomponent_obj)

    return method_params_dict, submethod_params_dict

def get_method_name_and_params_dict(run, 
                                    component_obj):
    """Function to extract a parameters dictionary from a component object"""

    component_class = component_obj.class_name.split(".")[-1]
    if component_class in OPENML_TO_SKLEARN_CLASSES:
        component_class = OPENML_TO_SKLEARN_CLASSES[component_class]
        #print(f"Component class updated to {component_class}")
 
    #method_name = code_method_to_exekg_method(component_class)
    method_params_dict = (
        get_param_dict_from_openml_run(run, filter_by_component_id=component_obj.id)
        if run
        else {}
    )
 
    #return method_name, method_params_dict
    return method_params_dict

def get_all_openml_components():
    """Function that finds and saves all components used within an OpenML benchmark"""
    
    run_logs_path = "openml_exekgs/logs/runs_log.csv"
    run_logs_df = pd.read_csv(run_logs_path, encoding='utf-8')
    run_logs_df = run_logs_df[run_logs_df['error'].isna()]
    flows = list(set(run_logs_df["flow_id"].to_list()))

    discrete_components = []
    
    for flow in flows:
    
      flow = openml.flows.get_flow(int(flow))
      print(flow.id)

      # If a flow has a componenets item, it means that it contains multiple components
      if flow.components:
          for component in flow.components:
              if flow.components[component].name not in discrete_components:
                  discrete_components.append(flow.components[component].name)

      # Else, it contains one component, seen in its name
      else:
          if flow.name not in discrete_components:
              discrete_components.append(flow.name)

    components_path = "evaluations/ground_truth/data/components.txt"
    with open(components_path , 'w') as file:
        for componenet in sorted(discrete_components):
            file.write(f"{componenet}\n")

    return

def assert_pipeline_dict_integrity():
    """Function that validates generated pipeline dictionaries"""

    run_logs_path = "openml_exekgs/logs/runs_log.csv"
    run_logs_df = pd.read_csv(run_logs_path, encoding='utf-8')
    run_logs_df = run_logs_df[run_logs_df['error'].isna()]
    runs = run_logs_df["run_id"].to_list()
    continue_loop = True
    faulty_runs = []
    faulty_runs_path = "data/faulty_run_dicts.txt"

    for run in runs:

        run = openml.runs.get_run(int(run))
        flow_id = run.flow_id
        flow = openml.flows.get_flow(flow_id)

        #print(flow_id)
        #print(run.id)
        
        run_dict = []

        # If a flow has a componenets item, it means that it contains multiple components
        if flow.components:
            try:
                run_dict = add_or_update_tasks(flow.components, flow, run, run_dict)
            except: 
                print(f"\nFaulty run found ({run.id}), skipping!")
                faulty_runs.append(run.id)
                with open(faulty_runs_path , 'a') as file:
                    file.write(f"{run.id}\n")

        # Else, it contains one component, seen in its name
        else:
            #print(f"Single Component: {flow.name}")
            try:
                param_dict = get_param_dict_from_openml_run(run)
                #print(param_dict)
                component_dict = {}
                component_dict["component"] = flow.name
                component_dict["rank"] = 0 
                component_dict["depth"] = 0 
                component_dict["params"] = param_dict
                run_dict.append(component_dict)
            except:
                print(f"\nFaulty run found ({run.id}), skipping!")
                faulty_runs.append(run.id)
                with open(faulty_runs_path , 'a') as file:
                    file.write(f"{run.id}\n")

        #print(f"\n\n\nRun dict:")
        #pprint.pprint(run_dict)

    print(f"\nFaulty runs:{faulty_runs}")
    
    # with open(faulty_runs_path , 'w') as file:
    #     for run in sorted(faulty_runs):
    #         file.write(f"{run}\n")

    return 


def get_pipeline_dict(run):
    """Function that extracts a pipeline dictionary given the OpenML run ID"""
    
    flow_id = run.flow_id
    flow = openml.flows.get_flow(flow_id)
    
    run_dict = []
    
    #print("\nFlow sub-steps:")
    # If a flow has a componenets item, it means that it contains multiple components
    if flow.components:
        run_dict = add_or_update_tasks(flow.components, flow, run, run_dict)

    # Else, it contains one component, seen in its name
    else:
        param_dict = get_param_dict_from_openml_run(run)
        component_dict = {}
        component_dict["component"] = flow.name
        component_dict["rank"] = 0 
        component_dict["depth"] = 0 
        component_dict["params"] = param_dict
        component_dict["steps"] = []
        run_dict.append(component_dict)

    return run_dict







