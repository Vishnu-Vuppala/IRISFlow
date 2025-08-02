from sklearn.datasets import load_iris
iris = load_iris()

import pandas as pd

# Convert iris data to DataFrame
df_iris = pd.DataFrame(iris.data, columns=iris.feature_names)
df_iris['target'] = iris.target

# Print three rows for each iris type
for target_value in df_iris['target'].unique():
    print(f"Iris type {target_value}:")
    print(df_iris[df_iris['target'] == target_value].head(3))
    print()