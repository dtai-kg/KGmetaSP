import openml
import math 

from utils.file_helpers import * 
from metafeatures.metafeature_groups import *
from config.config import Config

def get_metafeatures(dataset_id: int, extraction_settings: str) -> dict:

    """
    Get metafeatures for a given dataset ID and extraction settings.
    
    Parameters
    ----------
    dataset_id : int
        OpenML dataset ID.
    extraction_settings : str
        Extraction settings. 
    
    Returns
    -------
    dict
        Dictionary containing the extracted metafeatures.
    """
    
    # Load the dataset from OpenML
    dataset = openml.datasets.get_dataset(dataset_id)
    dataset_qualities = dataset.qualities
    
    # Extract the metafeatures based on the specified settings
    if extraction_settings == "all":
        return dataset_qualities
    elif extraction_settings == "simple":
        included_qualities = simple_metafeatures
    elif extraction_settings == "statistical":  
        included_qualities = statistical_metafeatures
    elif extraction_settings == "information_theory":
        included_qualities = information_theory_metafeatures
    elif extraction_settings == "landmarkers":
        included_qualities = landmarker_metafeatures
    elif extraction_settings == "mlsea":
        included_qualities = mlsea_metafeatures    
    
    dataset_qualities_filtered = {quality: dataset_qualities[quality] 
                                  for quality in included_qualities}
    
    return dataset_qualities_filtered
    


def extract_metafeatures(dataset_ids: list[int], extraction_settings: list[str]):
    """
    Extract metafeatures for a list of dataset IDs and save them to a CSV file.
    
    Parameters
    ----------
    dataset_ids : list[int]
        List of OpenML dataset IDs.
    output_path : str
        Path to save the extracted metafeatures.
    extraction_settings : list[str]
        List of extraction settings. Each setting corresponds to a different set of metafeatures to extract.
    """

    for setting in extraction_settings:
        metafeature_dict = {}
        for dataset_id in dataset_ids:
            metafeature_values = list(get_metafeatures(int(dataset_id), setting).values())
            for val in range(len(metafeature_values)):
                if math.isnan(metafeature_values[val]):
                    metafeature_values[val] = 0
            metafeature_dict[int(dataset_id)] = metafeature_values
            

        save_path = Config.DIR_PREFIX + f"metafeatures/metafeature_store/metafeatures_{setting}.pkl"

        #save(save_path, metafeature_dict)
                                    
    return

def main():

    # Load the dataset IDs from the file
    dataset_ids_path = Config.DIR_PREFIX + "metafeatures/dataset_ids.pkl"
    dataset_ids = load(dataset_ids_path)    

    extraction_settings = Config.METAFEATURE_SETTINGS

    extract_metafeatures(dataset_ids, extraction_settings)

    return


if __name__ == "__main__":

    main()