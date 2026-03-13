import itertools
import subprocess

from config.config import Config

def main():
    """
    Main function to run the experiment simulator.
    """
    
    #Set experiment parameters
    target_metrics = Config.TARGET_METRICS
    metafeature_settings = Config.METAFEATURE_SETTINGS
    meta_model_labels = Config.CLASS_META_MODEL_LABELS + Config.REG_META_MODEL_LABELS
    n_meta_models = 1500 #All samples
    split_mode = "global" #Global split mode

    # Generate all combinations
    combinations = itertools.product(target_metrics, metafeature_settings, meta_model_labels)
    for target_metric, meta_feature, meta_model in combinations:
        print(f"\n\nRunning experiment with target metric: {target_metric}, metafeature setting: {meta_feature}, meta model: {meta_model}\n\n")
        
        # Run the experiment
        subprocess.run([
            "python", "-m", "baselines.model_specific.meta_training",
            "-t", target_metric,
            "-m", meta_model,
            "-e", meta_feature,
            "-n", str(n_meta_models),
            "-sm", split_mode
        ])

    return


if __name__ == "__main__":

    main()
