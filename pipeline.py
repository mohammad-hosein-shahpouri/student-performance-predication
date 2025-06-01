from sklearn.compose import ColumnTransformer
from sklearn.discriminant_analysis import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from ucimlrepo import fetch_ucirepo,dotdict # https://github.com/uci-ml-repo/ucimlrepo

true = True
false = False
__all__ = ['fetch_dataset', 'prepare_pipeline']

def fetch_dataset(dataset_id: int, quiet: bool = false) -> dotdict:
    try:
        dataset = fetch_ucirepo(id=dataset_id)        
        if not quiet:
            print("Dataset Info:")
            print(f"\tName: {dataset.metadata.name}")
            print(f"\tDescription: {dataset.metadata.abstract}")
            print(f"\tDonated on: {dataset.metadata.year_of_dataset_creation}")
            print ("-------------------------- Dataset fetched successfully. --------------------------")
        return dataset
    except Exception as e:
        print(f"Failed to fetch the dataset({dataset_id}): {e}")
        exit(32)

def prepare_pipeline(dataset: dotdict):
    features = dataset.data.features

    numeric_cols = features.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = features.select_dtypes(include=['object', 'category']).columns.tolist()
    
    transformer = ColumnTransformer([
        ("num", StandardScaler(), numeric_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)
    ])

    transformed = transformer.fit_transform(features)
    return transformed