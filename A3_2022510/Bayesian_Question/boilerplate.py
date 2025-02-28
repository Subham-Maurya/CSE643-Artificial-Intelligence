#############
## Imports ##
#############

import pickle
import pandas as pd
import numpy as np
import bnlearn as bn
from test_model import test_model

######################
## Boilerplate Code ##
######################

def load_data():
    train_df = pd.read_csv("train_data.csv")
    val_df = pd.read_csv("validation_data.csv")
    return train_df, val_df

def make_network(df):
    DAG = [
        ('Start_Stop_ID', 'End_Stop_ID'),
        ('Start_Stop_ID', 'Distance'),
        ('Start_Stop_ID', 'Zones_Crossed'),
        ('Start_Stop_ID', 'Route_Type'),
        ('Start_Stop_ID', 'Fare_Category'),
        ('End_Stop_ID', 'Distance'),
        ('End_Stop_ID', 'Zones_Crossed'),
        ('End_Stop_ID', 'Route_Type'),
        ('End_Stop_ID', 'Fare_Category'),
        ('Distance', 'Zones_Crossed'),
        ('Distance', 'Route_Type'),
        ('Distance', 'Fare_Category'),
        ('Zones_Crossed', 'Route_Type'),
        ('Zones_Crossed', 'Fare_Category'),
        ('Route_Type', 'Fare_Category')
    ]

    DAG = bn.make_DAG(DAG)
    model = bn.parameter_learning.fit(DAG, df)
    return model

def make_pruned_network(df):
    DAG = [
        ('Start_Stop_ID', 'Distance'),
        ('Start_Stop_ID', 'Zones_Crossed'),
        ('End_Stop_ID', 'Distance'),
        ('End_Stop_ID', 'Zones_Crossed'),
        ('Distance', 'Fare_Category'),
        ('Zones_Crossed', 'Fare_Category'),
    ]

    DAG = bn.make_DAG(DAG)
    pruned_model = bn.parameter_learning.fit(DAG, df)
    return pruned_model

def make_optimized_network(df):
    optimized_model = bn.structure_learning.fit(df, methodtype='hc')  # hc = Hill Climbing method
    optimized_model = bn.parameter_learning.fit(optimized_model, df)
    return optimized_model

def save_model(fname, model):
    with open(fname, 'wb') as f:
        pickle.dump(model, f)

def evaluate(model_name, val_df):
    with open(f"{model_name}.pkl", 'rb') as f:
        model = pickle.load(f)
        correct_predictions, total_cases, accuracy = test_model(model, val_df)
        print(f"Total Test Cases: {total_cases}")
        print(f"Total Correct Predictions: {correct_predictions} out of {total_cases}")
        print(f"Model accuracy on filtered test cases: {accuracy:.2f}%")

############
## Driver ##
############

def main():
    # Load data
    train_df, val_df = load_data()

    # Create and save base model
    base_model = make_network(train_df.copy())
    save_model("base_model.pkl", base_model)
    # bn.plot(base_model)

    # Create and save pruned model
    pruned_network = make_pruned_network(train_df.copy())
    save_model("pruned_model.pkl", pruned_network)
    # bn.plot(pruned_network)

    # # Create and save optimized model
    optimized_network = make_optimized_network(train_df.copy())
    save_model("optimized_model.pkl", optimized_network)
    # bn.plot(optimized_network)

    # Evaluate all models on the validation set
    evaluate("base_model", val_df)
    evaluate("pruned_model", val_df)
    evaluate("optimized_model", val_df)

    print("[+] Done")

if __name__ == "__main__":
    main()
