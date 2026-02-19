import lime
import lime.lime_tabular
import numpy as np

def create_lime_explainer(X_train, feature_names, class_names):
    explainer = lime.lime_tabular.LimeTabularExplainer(
        X_train,
        feature_names=feature_names,
        class_names=class_names,
        mode="classification"
    )
    return explainer
