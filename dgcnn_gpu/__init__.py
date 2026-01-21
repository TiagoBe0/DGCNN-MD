"""DGCNN GPU-optimized package"""
from .model import get_model, DGCNN_Classifier, DGCNN_Segmentation

__all__ = ['get_model', 'DGCNN_Classifier', 'DGCNN_Segmentation']
