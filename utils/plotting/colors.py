from collections import OrderedDict

INFERENCE_METHODS = OrderedDict(
    {
        "Standard": "gray",
        "Temperature Scaling": "C0",
        "Laplace (Last-layer, GS)": "#fdbf6f",
        "Laplace (Last-layer, ML)": "C1",
        "Weight-space VI (Mean-field)": "C3",
        "Implicit Bias VI (Low-rank)": "C2",
        "Generalized VI": "#7A85B6",
        "Ensemble": "C4",
        "SWAG": "C6",
        "Generalized VI (Low-rank)": "#7A85B6",
        # "Implicit Bias VI (Kronecker)": "#b2df8a",
    }
)
