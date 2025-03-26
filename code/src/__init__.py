# customer_financial_ai_hyperpersonalisation4/__init__.py

from .data_loader import load_data
from .preprocessing import preprocess_data
from .model import train_model
from .recommendation import generate_recommendations

__all__ = ["load_data", "preprocess_data", "train_model", "generate_recommendations"]
