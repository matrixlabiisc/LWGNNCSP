import os
import numpy as np
import pandas as pd

def get_metric1_metric2(csv_dir):
    '''
    calculate metric1 and metric2 for the pred_results.csv in csv_dir

    metric 1 : number of structures such that pred energy < threshold, 
        where threshold is minimum formation energy in the training dataset (=-5)
    metric 2: number of structures such that pred_energy < relaxed_counter_part_energy

    Args:
    csv_dir: str, directory containing pred_results.csv

    Returns:
    metric1: int, number of structures such that pred energy < threshold
    metric2: int, number of structures such that pred_energy < relaxed_counter_part_energy
    '''
    df = pd.read_csv(os.path.join(csv_dir, "pred_results.csv"))

    pred = df['predicted_formation_energy'].values
    relaxed_counter_part = df['formation_energy'].values

    # metric 1 : number of structures such that pred energy < threshold (-5)
    threshold = [-5]
    metric1 = np.sum(pred < threshold)

    # metric 2: number of structures such that pred_energy < relaxed_counter_part_energy
    metric2 = np.sum(pred < relaxed_counter_part)    

    return metric1, metric2

if __name__ == '__main__':

    path_to_results = "./holdout_unrelaxed_predictions"
    model_type_trial_list = ["M1_trial1", "M1_trial2", "M1_trial3", "M1_trial4", "M1_trial5",
                             "M2_trial1", "M2_trial2", "M2_trial3", "M2_trial4", "M2_trial5",]

    output_file = "holdout_unrelaxed_metrics_schnet_kgcnn.csv"

    df = pd.DataFrame(columns=["model_type_trial", "metric1", "metric2"])

    for i,model_type_trial in enumerate(model_type_trial_list):
        metric1, metric2 = get_metric1_metric2(f"{path_to_results}/schnet_kgcnn_{model_type_trial}")
        row = {"model_type_trial": model_type_trial, "metric1": metric1, "metric2": metric2}
        df.loc[i] = row
    
    df.to_csv(output_file, index=False, header=True)
    print(f"Metrics saved to {output_file}")
