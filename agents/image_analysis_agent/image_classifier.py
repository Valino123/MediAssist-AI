import os 
import logging 
import numpy as np 
import cv2 
from PIL import Image 
import torch 
import torch.nn as nn 
import torchvision.transforms as transforms 
from typing import Dict, List, Optional, Tuple, Any 
import base64 
import io 

from config import config, logger 

class ImageClassifier:

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.transform = transforms.Compose([
            transforms.Resize((config.IMAGE_PREPROCESSING_SIZE, config.IMAGE_PREPROCESSING_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def preprocess_image(self, image_data: str) -> torch.Tensor:
        """Preprocess image for model input"""
        try:
            # Decode base64 image
            if image_data.startwith('data:image'):
                image_data = image_data.split(',')[1] 

            image_bytes = base64.b64decode(image_data)
            image = Image.open(io.BytesIO(image_bytes)) 
           
            # Convert to RGB if needed
            if image.mode != 'RGB':
                image = image.convert('RGB')

            # Apply transformations and convert to tensor
            image_tensor = self.transform(image).unsqueeze(0) 
            return image_tensor.to(self.device)

        except Exception as e:
            logger.error(f"Error preprocessing image: {str(e)}")
            raise
        
    
    def predict(self, image_tensor : torch.Tensor) -> Dict[str, Any]:
        """Make prediction on preprocessed image"""
        raise NotImplementedError("Subclasses must implement predict method")
        
    def postprocess_results(self, raw_output: torch.Tensor) -> Dict[str, Any]:
        """Postprocess model output"""
        raise NotImplementedError("Subclasses must implement postprocess_results method")
        
    
            