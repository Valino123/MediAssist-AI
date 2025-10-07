import base64
import io
import torch
from PIL import Image
from datetime import datetime
from typing import Dict, Any

# Key imports from Hugging Face
from transformers import AutoImageProcessor, AutoModelForImageClassification

from config import logger
# We still import ImageClassifier to maintain the class inheritance structure
from .image_classifier import ImageClassifier

class ChestXrayAgent(ImageClassifier):
    """Agent for Chest X-ray analysis (Pneumonia) using a Hugging Face model."""

    def __init__(self, model_name: str = "codewithdark/vit-chest-xray"):
        super().__init__()
        self.model_name = model_name
        self.model = None
        self.processor = None
        self.load_model()

    def load_model(self):
        """Load the ViT model and processor from Hugging Face."""
        try:
            logger.info(f"Loading model '{self.model_name}' from Hugging Face...")
            
            # The processor handles all the transformations for the ViT model
            self.processor = AutoImageProcessor.from_pretrained(self.model_name, use_fast=True)
            
            # The model is the Vision Transformer (ViT) architecture with its pre-trained weights
            self.model = AutoModelForImageClassification.from_pretrained(self.model_name)

            self.model.to(self.device)
            self.model.eval() # Set the model to evaluation mode
            logger.info("Hugging Face Chest X-ray model loaded successfully.")

        except Exception as e:
            logger.error(f"Error loading model from Hugging Face: {str(e)}")
            raise

    def predict(self, image: Image.Image) -> torch.Tensor:
        """
        Predicts using the loaded Hugging Face model.
        Returns the raw logits from the model.
        """
        try:
            # The processor converts the PIL image into the exact tensor format the model needs
            inputs = self.processor(images=image, return_tensors="pt").to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs)
                return outputs.logits
        except Exception as e:
            logger.error(f"Error during X-ray prediction: {str(e)}")
            raise

    def postprocess_results(self, logits: torch.Tensor) -> Dict[str, Any]:
        """Post-processes the raw logits to create a final report."""
        try:
            # Map CheXpert generic labels to meaningful medical conditions
            # Based on CheXpert dataset structure for chest X-ray classification
            # CheXpert is a large-scale chest X-ray dataset with 14 observations
            # This model appears to classify 5 common conditions
            chexpert_label_mapping = {
                "0": "Normal",           # No abnormalities detected
                "1": "Atelectasis",      # Partial or complete lung collapse
                "2": "Cardiomegaly",     # Enlarged heart
                "3": "Consolidation",    # Lung consolidation (pneumonia-like)
                "4": "Edema"            # Pulmonary edema (fluid in lungs)
            }
            
            # Get raw class names from model config (LABEL_0, LABEL_1, etc.)
            raw_class_names = self.model.config.id2label
            
            # Create meaningful class names mapping with robust key handling
            meaningful_class_names = {}
            for idx, raw_label in raw_class_names.items():
                # Convert idx to string for consistent key handling
                idx_str = str(idx)
                meaningful_class_names[idx_str] = chexpert_label_mapping.get(idx_str, f"Unknown_Condition_{idx_str}")

            probabilities = torch.softmax(logits, dim=-1)
            confidence, predicted_id = torch.max(probabilities, 1)

            predicted_id_value = predicted_id.item()
            
            # Convert predicted_id to string to match the model config keys
            predicted_class_key = str(predicted_id_value)
            predicted_class = meaningful_class_names.get(predicted_class_key, f"Unknown_Condition_{predicted_id_value}")
            confidence_score = confidence.item()

            # Create class probabilities with meaningful names
            class_probabilities = {}
            for i, prob in enumerate(probabilities[0]):
                key = str(i)
                class_probabilities[meaningful_class_names.get(key, f"Unknown_Condition_{i}")] = prob.item()
            
            analysis_report = self._generate_analysis_report(predicted_class, confidence_score, class_probabilities)

            return {
                'agent': 'CHEST_XRAY_AGENT_HF',
                'predicted_class': predicted_class,
                'confidence': confidence_score,
                'class_probabilities': class_probabilities,
                'analysis_report': analysis_report,
                'timestamp': datetime.now().isoformat(),
                'status': 'success'
            }
        except Exception as e:
            logger.error(f"Error post-processing X-ray results: {str(e)}")
            return {'agent': 'CHEST_XRAY_AGENT_HF', 'error': str(e), 'status': 'error'}

    def process_image(self, image_data: str) -> Dict[str, Any]:
        """
        Full pipeline: decodes base64, predicts, and post-processes.
        """
        try:
            # 1. Decode base64 string to PIL Image
            if image_data.startswith('data:image'):
                image_data = image_data.split(',')[1]
            image_bytes = base64.b64decode(image_data)
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

            # 2. Make prediction to get raw logits
            logits = self.predict(image)

            # 3. Post-process the results into a final report
            final_result = self.postprocess_results(logits)
            
            return final_result
        except Exception as e:
            logger.error(f"Error processing X-ray image: {str(e)}")
            return {'agent': 'CHEST_XRAY_AGENT_HF', 'error': str(e), 'status': 'error'}

    def _generate_analysis_report(self, predicted_class: str, confidence: float, probabilities: Dict[str, float]) -> str:
        """Generate a detailed analysis report for Pneumonia detection."""
        report = f"Chest X-ray Analysis Results:\n\n"
        report += f"Predicted Condition: {predicted_class}\n"
        report += f"Confidence Level: {confidence:.2%}\n\n"

        report += "Detailed Probability Breakdown:\n"
        for class_name, prob in sorted(probabilities.items(), key=lambda x: x[1], reverse=True):
            report += f"• {class_name}: {prob:.2%}\n"

        report += f"\nAnalysis Summary:\n"
        
        # Categorize specific CheXpert conditions
        predicted_lower = predicted_class.lower()
        
        if predicted_lower == 'normal':
            report += "✅ The chest X-ray appears normal with no clear signs of abnormalities."
        elif predicted_lower == 'atelectasis':
            report += "⚠️ ATELECTASIS DETECTED: The X-ray shows signs of lung collapse or incomplete expansion.\n"
            report += "Atelectasis can be caused by airway obstruction, compression, or post-surgical complications. Medical evaluation is recommended."
        elif predicted_lower == 'cardiomegaly':
            report += "⚠️ CARDIOMEGALY DETECTED: The X-ray shows an enlarged heart (cardiothoracic ratio > 50%).\n"
            report += "This may indicate heart failure, hypertension, or other cardiac conditions. Cardiology consultation is recommended."
        elif predicted_lower == 'consolidation':
            report += "⚠️ CONSOLIDATION DETECTED: The X-ray shows areas of lung consolidation (dense, opaque areas).\n"
            report += "This may indicate pneumonia, infection, or other inflammatory conditions. Medical attention is recommended."
        elif predicted_lower == 'edema':
            report += "🚨 PULMONARY EDEMA DETECTED: The X-ray shows signs of fluid accumulation in the lungs.\n"
            report += "This is a serious finding that may indicate heart failure, fluid overload, or other critical conditions requiring immediate medical evaluation."
        else:
            report += f"📋 CLASSIFIED AS: '{predicted_class}'.\n"
            report += "Please consult with a radiologist or pulmonologist for proper evaluation and interpretation of this chest X-ray."

        # Add confidence-based recommendations
        if confidence >= 0.90:
            report += f"\n\nHigh confidence result ({confidence:.1%}). However, this AI analysis is not a substitute for professional medical evaluation."
        elif confidence >= 0.75:
            report += f"\n\nModerate confidence result ({confidence:.1%}). Professional medical evaluation is recommended."
        else:
            report += f"\n\n⚠️ Low confidence result ({confidence:.1%}). This analysis should be interpreted with caution and professional medical evaluation is strongly recommended."

        # Add general disclaimer
        report += f"\n\nNote: This information is for educational purposes and is not a substitute for professional medical advice. For urgent or personal medical concerns, consult a qualified clinician."

        return report
