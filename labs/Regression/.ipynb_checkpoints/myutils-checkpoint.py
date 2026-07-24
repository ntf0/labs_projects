import numpy as np
import pandas as pd



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

                
        elif transformation == "Try Log1p first; if skewness remains high, try Box-Cox":

            # Try Log1p first
            log_transformed = np.log1p(original)
            log_skewness = log_transformed.skew()

            # If Log1p is still highly skewed, try Box-Cox
            if abs(log_skewness) >= 1:

                # Box-Cox requires values greater than zero
                boxcox_original = original + 1

                pt = PowerTransformer(
                    method="box-cox",
                    standardize=False
                )

                transformed_values = pt.fit_transform(
                    boxcox_original.to_numpy().reshape(-1, 1)
                ).flatten()

                transformed = pd.Series(
                    transformed_values,
                    index=original.index
                )

                transformation = "Box-Cox"

            else:
                transformed = log_transformed
                transformation = "Log1p"

        elif transformation == "Yeo-Johnson":
            pt = PowerTransformer(
                method="yeo-johnson",
                standardize=False
            )

            transformed_values = pt.fit_transform(
                original.to_numpy().reshape(-1, 1)
            ).flatten()

            transformed = pd.Series(
                transformed_values,
                index=original.index
            )


        
        
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
