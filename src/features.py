from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import FunctionTransformer

def build_numeric_preprocess():
    """
    Prétraitement minimal (baseline) :
    - imputation médiane
    - standardisation
    """
    return Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

def _clip(X): 
    return X.clip(-3, 3)

#add a comment just for testing purpose
def build_numeric_preprocess(): 
    return Pipeline(steps=[ 
        ("imputer", SimpleImputer(strategy="median")), 
        ("scaler", StandardScaler()), 
        ("clip", FunctionTransformer(_clip)), 
    ]) 