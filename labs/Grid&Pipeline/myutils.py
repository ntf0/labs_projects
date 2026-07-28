import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import PowerTransformer

from sklearn.metrics import mean_squared_error, r2_score


def skew_calc(df):
    """
    Diagnoses skewness for every numeric column in a DataFrame and recommends a transformation based on the column's skewness and
    minimum value. Binary, encoded, and ID columns are excluded, since skewness isn't a meaningful for them.
    It returns a DataFrame with the following columns:
    Feature, Skewness, Degree, Direction, Recommended Transformation
    """
    # Your code here 
    results = []

    # Select numeric columns only
    numeric_columns = df.select_dtypes(include=np.number).columns

    for column in numeric_columns:

        clean_column = df[column].dropna()
        unique_count = clean_column.nunique()

        # Skip empty or constant columns
        if unique_count <= 1:
            continue

        # Skip binary columns
        if unique_count == 2:
            continue

        # Skip common ID/index columns
        column_lower = column.lower()

        if (
            column_lower == "id"
            or column_lower.endswith("_id")
            or column_lower.startswith("id_")
        ):
            continue

        # Calculate skewness
        skewness = clean_column.skew()
        absolute_skewness = abs(skewness)
        minimum_value = clean_column.min()

        # Determine direction
        if skewness > 0:
            direction = "Right"
        elif skewness < 0:
            direction = "Left"
        else:
            direction = "Symmetric"

        # Determine degree and recommended transformation
        if absolute_skewness < 0.5:
            degree = "Approximately symmetric"
            transformation = "None"

        elif absolute_skewness < 1:
            degree = "Moderately skewed"

            #
            if skewness > 0 and minimum_value >= 0:
                transformation = "Square root"
            elif skewness < 0 and minimum_value >= 0:
                transformation = "Square"
            #if min<0 we should use Yeo
            else:
                transformation = "Yeo-Johnson"


        else:
            degree = "Highly skewed"

            #if there is not negative values do: 
            if skewness > 0 and minimum_value >= 0:
                transformation = "Try Log1p first; if skewness remains high, try Box-Cox"
            #if there is negative values do yeo:
            else:
                transformation = "Yeo-Johnson"

        #
        results.append({
            "Feature": column,
            "Skewness": skewness,
            "Degree": degree,
            "Direction": direction,
            "Recommended Transformation": transformation
        })

    skew_table = pd.DataFrame(results)

    if not skew_table.empty:
        skew_table["Skewness"] = skew_table["Skewness"].round(3)

    return skew_table



def plot_transformations(df, skew_table):
    """
    Applies the recommended transformation to each column and plots
    the original and transformed distributions side by side.

    Returns a transformed copy of the DataFrame.
    """

    transformed_df = pd.DataFrame()

    # Go through every row in the skewness table
    for index, row in skew_table.iterrows():

        feature = row["Feature"]
        transformation = row["Recommended Transformation"]
        valid_mask = df[feature].notna()
        original = df.loc[valid_mask, feature].astype(float)
        
        # Apply the recommended transformation
        if transformation == "None":
            transformed = original.copy()

        elif transformation == "Square root":
            transformed = np.sqrt(original)

        elif transformation == "Square":
            transformed = np.square(original)

        elif transformation == "Log1p":
            transformed = np.log1p(original)
        
        elif transformation == "Yeo-Johnson":
            pt = PowerTransformer(
                method="yeo-johnson", #choose the method
                standardize=False #Don't use standarlize 
            )
            # do the calculations 
            transformed_values = pt.fit_transform(
                #to_numpy(): Convert our data to NumPy array
                #reshape(-1,1): -1,1 make the NumPy array in two dimenionas to make it readable by PowerTransformation
                original.to_numpy().reshape(-1, 1) 
            ).flatten() # Recovert to NumPy array to make it easier to store in a pandas Series

            transformed = pd.Series(
                transformed_values,
                index=original.index
            )

            transformation = "Box-Cox"

        
        elif transformation == "Try Log1p first; if skewness remains high, try Box-Cox":

            # Try Log1p first
            log_transformed = np.log1p(original)
            log_skewness = log_transformed.skew()

            # If Log1p is still highly skewed, try Box-Cox
            if abs(log_skewness) > 0.5:

                # Box-Cox requires values greater than zero
                boxcox_original = original + 1

                pt = PowerTransformer(
                method="box-cox", #choose the method
                standardize=False #Don't use standarlize 
                )
                
                # do the calculations 
                transformed_values = pt.fit_transform(
                    #to_numpy(): Convert our data to NumPy array
                    #reshape(-1,1): -1,1 make the NumPy array in two dimenionas to make it readable by PowerTransformation
                    boxcox_original.to_numpy().reshape(-1, 1) 
                ).flatten() # Recovert to NumPy array to make it easier to store in a pandas Series

                transformed = pd.Series(
                    transformed_values,
                    index=original.index
                )

                transformation = "Box-Cox"

            else:
                transformed = log_transformed
                transformation = "Log1p"


        
        else:
            transformed = original.copy()

        # Store the transformed values
        transformed_df[feature+' transform'] = transformed

        # Calculate skewness before and after
        before_skew = original.skew()
        after_skew = transformed.skew()

        # Plot both distributions
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        sns.histplot(original, kde=True, ax=axes[0])
        axes[0].set_title(
            f"{feature} — Before\nSkewness = {before_skew:.3f}"
        )
        axes[0].set_xlabel(feature)

        sns.histplot(transformed, kde=True, ax=axes[1])
        axes[1].set_title(
            f"{feature} — After {transformation}\n"
            f"Skewness = {after_skew:.3f}"
        )
        axes[1].set_xlabel(feature)

        plt.tight_layout()
        plt.show()

    return transformed_df

    
def evaluate_model(model, X_train, X_test, y_train, y_test, model_name="Model"):
    """
    Predicts and calculates R2 and RMSE for both train and test sets.
    Then prints the Train R2 and RMSE along with the Test R2 and RMSE
    """
    print(f"--- {model_name} Performance ---")
    
    # Generate predictions for training and testing data
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # Calculate R² for training and testing data
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)

    # Calculate RMSE for training and testing data
    train_rmse = np.sqrt(
        mean_squared_error(y_train, y_train_pred)
    )

    test_rmse = np.sqrt(
        mean_squared_error(y_test, y_test_pred)
    )

    # Print results
    print(f"Train R²:   {train_r2:.4f}")
    print(f"Test R²:    {test_r2 :.4f}")
    print(f"Train RMSE: {train_rmse:.4f}")
    print(f"Test RMSE:  {test_rmse:.4f}")    