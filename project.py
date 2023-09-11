import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from xgboost import XGBRegressor
import numpy as np
from joblib import dump, load

sns.set_theme(style="darkgrid")

class RegressionModelFactory:
    @staticmethod
    def create_model(model_type):
        if model_type == 'linear':
            return LinearRegression()
        elif model_type == 'svm':
            return SVR()
        elif model_type == 'random_forest':
            return RandomForestRegressor()
        elif model_type == 'gradient_boosting':
            return GradientBoostingRegressor()
        elif model_type == 'xgb':
            return XGBRegressor()
        elif model_type == 'lasso':
            return Lasso()
        elif model_type == 'ridge':
            return Ridge()
        else:
            raise ValueError("Unsupported model type")
def loadData():
    # Import excel file
    excel_file = "Net_Worth_Data.xlsx"
    # Read excel file with pandas
    df = pd.read_excel(excel_file)
    return df

def pairPlot():
    df = loadData()
    
    sns.pairplot(df)
    plt.show()

def preprocessData():
    df = loadData()
    
    # Assuming df is your DataFrame
    columns_to_select = ["Gender", "Age", "Income", "Credit Card Debt", "Inherited Amount", "Stocks", "Bonds", "Mutual Funds", "ETFs", "REITs"]
    input_df = df[columns_to_select].copy()

    # Extract the "Purchase" column and reshape it into a 2D array
    output_df = df[["Net Worth"]].copy()
    output_df = output_df.values.reshape(-1, 1)

    # Scale input_df and output_df
    scalerInput = MinMaxScaler()
    scaled_input_df = scalerInput.fit_transform(input_df)
    scalerOutput = MinMaxScaler()
    scaled_output_df = scalerOutput.fit_transform(output_df)
    
    return input_df, output_df, scaled_input_df,scaled_output_df, scalerInput, scalerOutput

# Transform and split data
# Transform scales the data in relationship of 0 to 1
# Test split seperated data for testing purposes with relationship of test_size being 20%
def splitData():
    input_df, output_df, scaled_input_df,scaled_output_df, scalerInput, scalerOutput = preprocessData()
    print("Splitting in progress...")
    
    # Create train/test with scaled dfs
    x = scaled_input_df
    y = scaled_output_df

    # using the train/test split function
    x_train, x_test, y_train, y_test = train_test_split(x ,y ,random_state=42 ,test_size=0.20 ,shuffle=True)
    
    print("Splitting complete")
    
    return x_train, x_test, y_train, y_test

# Evaluate Fastest Model
# Regressiong Models differ depending on relationship of graph eg; random ploted graph would be terrible for linear model
def trainModels():
    
    x_train, x_test, y_train, y_test = splitData()
    
    print("Training in progress...")
    
    model_types = [
            'linear','svm', 'random_forest', 'gradient_boosting', 'xgb', 'lasso', 'ridge'
    ]
    
    trained_models = {}
    
    for model_type in model_types:
        print(f"Training {model_type}")
        model = RegressionModelFactory.create_model(model_type)
        model.fit(x_train, y_train.ravel())
        trained_models[model_type] = model
        print(f"{model_type} training complete")
        
    print("Training complete")
        
    return trained_models

def evaluateModels():
    x_train, x_test, y_train, y_test = splitData()
    trained_models = trainModels()
    
    rmse_values = {}
    preds_reshaped = {}  # Create a dictionary to store reshaped predictions
    
    for name, model in trained_models.items():
        preds = model.predict(x_test)
        rmse_values[name] = mean_squared_error(y_test, preds, squared=False)
        
    #     # Reshape the preds array
    #     preds_reshaped[name] = preds.reshape(-1, 1)
        
    # y_test_reshaped = y_test.reshape(-1, 1)  # Reshape y_test
    
    # for name, preds in preds_reshaped.items():
    #     comparison_df = pd.DataFrame({'Actual': y_test_reshaped.flatten(), 'Predicted': preds.flatten()})
    #     print(f"Comparison DataFrame for {name}:")
    #     print(f"RMSE for {name}: {rmse_values[name]}")
    #     print(comparison_df)
    #     # print(rmse_values[name])
        
    return rmse_values

def plotPerformance():
    
    rmse_values = evaluateModels()
    
    plt.figure(figsize=(10,7))
    models = list(rmse_values.keys())
    rmse = list(rmse_values.values())
    bars = plt.bar(models, rmse, color=['blue', 'green', 'red', 'purple', 'orange', 'grey', 'yellow'])

    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.00001, round(yval, 5), ha='center', va='bottom', fontsize=10)

    plt.xlabel('Models')
    plt.ylabel('Root Mean Squared Error (RMSE)')
    plt.title('Model RMSE Comparison')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    
    return rmse_values

# Train best model using the whole data set
def train_entireData():
    
    input_df, output_df, scaled_input_df,scaled_output_df, scalerInput, scalerOutput = preprocessData()
    models = trainModels()
    rmse_values = plotPerformance()
    
    best_model_name = min(rmse_values, key=rmse_values.get)
    best_model = models[best_model_name]
    
    # Create instance
    xg = best_model
    
    print(f"Best model: {best_model}")
    
    # Train Model
    xg.fit(scaled_input_df, scaled_output_df)
    
    # Predict
    xg_preds = xg.predict(scaled_input_df)
    
    # Evaluate
    xg_rmse = mean_squared_error(scaled_output_df, xg_preds, squared=False)
    result = xg_rmse.reshape(-1,1)
    
    print(f"Prediction is: {scalerOutput.inverse_transform(result)}")
    
    return xg

def saveModel():
    xg = train_entireData()
    # Save
    filename = "purchseFinal.joblib"
    dump(xg, open(filename, "wb"))
    
def loadModel():
    input_df, output_df, scaled_input_df,scaled_output_df, scalerInput, scalerOutput = preprocessData()
    filename = "purchseFinal.joblib"
    # Load the trained model
    loaded_model = load(open(filename, "rb"))

    print("---Model loaded---")
    user_input = []
    for column in ["Gender", "Age", "Income", "Credit Card Debt", "Inherited Amount", "Stocks", "Bonds", "Mutual Funds", "ETFs", "REITs"]:
        while True:
            value = input(f"Enter value for {column}: ")
            if value.strip() == "":
                print("Value cannot be empty. Please try again.")
            else:
                try:
                    value = float(value)
                    user_input.append(value)
                    break
                except ValueError:
                    print("Invalid input. Please enter a valid number.")

    # Convert user inputs to a NumPy array and preprocess
    user_input = np.array([user_input])
    scaled_user_input = scalerInput.transform(user_input)
    
    # Use the loaded model to make a prediction
    predicted_amount = loaded_model.predict(scaled_user_input)
    
    predicted_amount = predicted_amount.reshape(1,-1)
    
    # Display the prediction to the user
    print("Predicted Car Purchase Amount:", scalerOutput.inverse_transform(predicted_amount))  # Access the single prediction value directly
    



if __name__ == "__main__":
    try: #add try except to handle missing value error
        pairPlot()
        plotPerformance()
        saveModel()
        loadModel()
        train_entireData()
    except ValueError as ve:
        print(f"Error: {ve}")