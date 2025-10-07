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

class SkinLesionAgent(ImageClassifier):
    """Agent for Skin Lesion analysis using a Hugging Face model."""

    def __init__(self, model_name: str = "Anwarkh1/Skin_Cancer-Image_Classification"):
        super().__init__()
        self.model_name = model_name
        self.model = None
        self.processor = None
        self.load_model()

    def load_model(self):
        """Load the ViT model and processor from Hugging Face."""
        try:
            logger.info(f"Loading model '{self.model_name}' from Hugging Face...")
            
            # This processor knows how to correctly resize, normalize, and prepare images for the model
            self.processor = AutoImageProcessor.from_pretrained(self.model_name, use_fast=True)
            
            # This is the Vision Transformer model with its pre-trained weights
            self.model = AutoModelForImageClassification.from_pretrained(self.model_name)

            self.model.to(self.device)
            self.model.eval() # Set the model to evaluation mode
            logger.info("Hugging Face skin lesion model loaded successfully.")

        except Exception as e:
            logger.error(f"Error loading model from Hugging Face: {str(e)}")
            raise

    def predict(self, image: Image.Image) -> torch.Tensor:
        """
        Predicts using the loaded Hugging Face model.
        Returns the raw logits from the model.
        """
        try:
            # The processor converts the PIL image into the exact tensor format the model expects
            inputs = self.processor(images=image, return_tensors="pt").to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs)
                return outputs.logits
        except Exception as e:
            logger.error(f"Error during skin lesion prediction: {str(e)}")
            raise

    def postprocess_results(self, logits: torch.Tensor) -> Dict[str, Any]:
        """Post-processes the raw logits to create a final, structured report."""
        try:
            # Get class names directly from the model's configuration
            class_names = self.model.config.id2label

            probabilities = torch.softmax(logits, dim=-1)
            confidence, predicted_id = torch.max(probabilities, 1)

            predicted_class = class_names[predicted_id.item()]
            confidence_score = confidence.item()

            class_probabilities = {class_names[i]: prob.item() for i, prob in enumerate(probabilities[0])}
            
            analysis_report = self._generate_analysis_report(predicted_class, confidence_score, class_probabilities)

            return {
                'agent': 'SKIN_LESION_AGENT_HF',
                'predicted_class': predicted_class,
                'confidence': confidence_score,
                'class_probabilities': class_probabilities,
                'analysis_report': analysis_report,
                'timestamp': datetime.now().isoformat(),
                'status': 'success'
            }
        except Exception as e:
            logger.error(f"Error post-processing skin lesion results: {str(e)}")
            return {'agent': 'SKIN_LESION_AGENT_HF', 'error': str(e), 'status': 'error'}

    def process_image(self, image_data: str) -> Dict[str, Any]:
        """
        Full pipeline: decodes a base64 image, runs prediction, and creates a report.
        """
        try:
            # 1. Decode base64 string to a PIL Image
            if image_data.startswith('data:image'):
                image_data = image_data.split(',')[1]
            image_bytes = base64.b64decode(image_data)
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

            # 2. Get raw prediction scores (logits) from the model
            logits = self.predict(image)

            # 3. Format the scores into a final report
            final_result = self.postprocess_results(logits)
            
            return final_result
        except Exception as e:
            logger.error(f"Error processing skin lesion image: {str(e)}")
            return {'agent': 'SKIN_LESION_AGENT_HF', 'error': str(e), 'status': 'error'}

    def _generate_analysis_report(self, predicted_class: str, confidence: float, probabilities: Dict[str, float]) -> str:
        """Generate a detailed analysis report for the skin lesion classification."""
        report = f"Skin Lesion Analysis Results:\n\n"
        report += f"Predicted Condition: {predicted_class}\n"
        report += f"Confidence Level: {confidence:.2%}\n\n"

        report += "Detailed Probability Breakdown:\n"
        for class_name, prob in sorted(probabilities.items(), key=lambda x: x[1], reverse=True):
            report += f"• {class_name}: {prob:.2%}\n"

        report += f"\nAnalysis Summary:\n"

        # More robust categorization using case-insensitive matching and partial matching
        predicted_lower = predicted_class.lower().replace('_', ' ').replace('-', ' ')
        
        # Define condition categories with various possible naming formats
        malignant_keywords = ['melanoma', 'basal cell carcinoma', 'basal_cell_carcinoma']
        pre_malignant_keywords = ['actinic keratoses', 'actinic_keratoses']
        benign_keywords = ['benign keratosis', 'dermatofibroma', 'melanocytic nevi', 'melanocytic_nevi', 'vascular lesions', 'vascular_lesions']

        # Check for malignant conditions
        is_malignant = any(keyword in predicted_lower for keyword in malignant_keywords)
        is_pre_malignant = any(keyword in predicted_lower for keyword in pre_malignant_keywords)
        is_benign = any(keyword in predicted_lower for keyword in benign_keywords)

        if is_malignant:
            report += f"🚨 POTENTIALLY MALIGNANT: The model identified the lesion as '{predicted_class}'.\n"
            report += "This finding is significant and requires immediate consultation with a dermatologist for a definitive diagnosis and potential biopsy."
        elif is_pre_malignant:
            report += f"⚠️ PRE-MALIGNANT CONDITION: The model identified the lesion as '{predicted_class}'.\n"
            report += "Actinic keratoses are considered pre-cancerous lesions that should be evaluated by a dermatologist. Early treatment can prevent progression to skin cancer."
        elif is_benign:
            report += f"✅ LIKELY BENIGN: The model identified the lesion as '{predicted_class}'.\n"
            report += "While likely benign, any changing, growing, or concerning lesion should be monitored and shown to a doctor for proper evaluation."
        else:
            report += f"📋 CLASSIFIED AS: '{predicted_class}'.\n"
            report += "Please consult with a dermatologist for proper evaluation and diagnosis of this skin lesion."

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
        
# # Global skin lesion agent instance
# skin_lesion_agent = SkinLesionAgent()