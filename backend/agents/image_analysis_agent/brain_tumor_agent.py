import base64
import io
import os
import torch
from PIL import Image
from datetime import datetime
from typing import Dict, Any

# Key imports from Hugging Face
from transformers import AutoImageProcessor, AutoModelForImageClassification

from config import config, logger
from .image_classifier import ImageClassifier # Assuming your base class is here

class BrainTumorAgent(ImageClassifier):
    """Agent for brain tumor analysis using a Hugging Face model."""

    def __init__(self, model_name: str = "Devarshi/Brain_Tumor_Classification"):
        super().__init__()
        self.model_name = model_name
        self.model = None
        self.processor = None
        self.load_model()
        # The class names are now loaded directly from the model's configuration

    def load_model(self):
        """Load the model and processor from Hugging Face.

        We explicitly cache weights under the configured models/ directory so that
        the first download is reused across runs and environments.
        """
        try:
            logger.info(f"Loading model '{self.model_name}' from Hugging Face...")

            # Ensure a dedicated cache directory for this model under models/
            model_cache_dir = os.path.join(config.MODELS_PATH, "brain_tumor")
            os.makedirs(model_cache_dir, exist_ok=True)

            # The processor handles all the transformations (resize, normalize, etc.)
            self.processor = AutoImageProcessor.from_pretrained(
                self.model_name,
                use_fast=True,
                cache_dir=model_cache_dir,
            )
            
            # The model is the neural network architecture with its pre-trained weights
            self.model = AutoModelForImageClassification.from_pretrained(
                self.model_name,
                cache_dir=model_cache_dir,
            )

            self.model.to(self.device)
            self.model.eval() # Set the model to evaluation mode
            logger.info("Hugging Face model loaded successfully.")

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
                # The model returns an object with logits, loss, etc. We only need the logits.
                outputs = self.model(**inputs)
                return outputs.logits
        except Exception as e:
            logger.error(f"Error during prediction: {str(e)}")
            raise

    def postprocess_results(self, logits: torch.Tensor) -> Dict[str, Any]:
        """Post-processes the raw logits to create a final report."""
        try:
            # Get class names from the model's configuration (e.g., {0: 'glioma', 1: 'meningioma', ...})
            class_names = self.model.config.id2label

            probabilities = torch.softmax(logits, dim=-1)
            confidence, predicted_id = torch.max(probabilities, 1)

            # Get the predicted class name using the ID
            predicted_class = class_names[predicted_id.item()]
            confidence_score = confidence.item()

            # Create a dictionary of all class probabilities
            class_probabilities = {class_names[i]: prob.item() for i, prob in enumerate(probabilities[0])}
            
            # You can re-use your existing report generation logic
            analysis_report = self._generate_analysis_report(predicted_class, confidence_score, class_probabilities)

            return {
                'agent': 'BRAIN_TUMOR_AGENT_HF',
                'predicted_class': predicted_class,
                'confidence': confidence_score,
                'class_probabilities': class_probabilities,
                'analysis_report': analysis_report,
                'timestamp': datetime.now().isoformat(),
                'status': 'success'
            }
        except Exception as e:
            logger.error(f"Error post-processing results: {str(e)}")
            # Return your standard error dictionary
            return {'agent': 'BRAIN_TUMOR_AGENT_HF', 'error': str(e), 'status': 'error'}
    
    def process_image(self, image_data: str) -> Dict[str, Any]:
        """
        Full pipeline: decodes base64, predicts, and post-processes.
        This is the main method you will call.
        """
        try:
            # 1. Decode base64 string to PIL Image
            if image_data.startswith('data:image'):
                image_data = image_data.split(',')[1]
            image_bytes = base64.b64decode(image_data)
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

            # 2. Make prediction (get raw logits)
            logits = self.predict(image)

            # 3. Post-process the results into a final report
            final_result = self.postprocess_results(logits)
            
            return final_result
        except Exception as e:
            logger.error(f"Error processing image: {str(e)}")
            return {'agent': 'BRAIN_TUMOR_AGENT_HF', 'error': str(e), 'status': 'error'}

    def _generate_analysis_report(self, predicted_class: str, confidence: float, probabilities: Dict[str, float]) -> str:
        """Generate a detailed analysis report for the brain tumor classification."""
        report = f"Brain MRI Analysis Results:\n\n"
        report += f"Predicted Condition: {predicted_class}\n"
        report += f"Confidence Level: {confidence:.2%}\n\n"

        report += "Detailed Probability Breakdown:\n"
        for class_name, prob in sorted(probabilities.items(), key=lambda x: x[1], reverse=True):
            report += f"• {class_name}: {prob:.2%}\n"

        report += f"\nAnalysis Summary:\n"

        # More robust categorization using case-insensitive matching
        predicted_lower = predicted_class.lower().replace('_', ' ').replace('-', ' ')
        
        # Define condition categories with various possible naming formats
        abnormal_keywords = ['tumor', 'brain tumor', 'brain_tumor', 'glioma', 'meningioma', 'abnormal', 'lesion']
        normal_keywords = ['no tumor', 'no_tumor', 'normal', 'healthy', 'benign']

        is_abnormal = any(keyword in predicted_lower for keyword in abnormal_keywords)
        is_normal = any(keyword in predicted_lower for keyword in normal_keywords)

        if is_abnormal:
            report += f"🚨 POTENTIALLY ABNORMAL: The model identified the scan as '{predicted_class}'.\n"
            report += "This finding requires immediate consultation with a neurologist or neurosurgeon for further evaluation, additional imaging, and potential biopsy."
        elif is_normal:
            report += f"✅ LIKELY NORMAL: The model identified the scan as '{predicted_class}'.\n"
            report += "No obvious signs of brain tumors were detected. However, this AI analysis is not a substitute for professional medical evaluation."
        else:
            report += f"📋 CLASSIFIED AS: '{predicted_class}'.\n"
            report += "Please consult with a neurologist for proper evaluation and interpretation of this brain scan."

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
