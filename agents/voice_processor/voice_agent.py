import logging
from typing import Dict, Any, Optional
from datetime import datetime

from config import config, logger
from .speech_to_text import SpeechToTextProcessor
from .text_to_speech import TextToSpeechProcessor

class VoiceAgent:
    """Agent for handling voice interactions"""

    def __init__(self):
        """Initialize the voice agent"""
        self.speech_to_text = SpeechToTextProcessor()
        self.text_to_speech = TextToSpeechProcessor()

    def process_voice_input(self, audio_data: str) -> Dict[str, Any]:
        """Process voice input and return text"""
        try:
            # Convert speech to text
            stt_result = self.speech_to_text.process_audio_file(audio_data)

            if stt_result['status'] == 'success':
                return {
                    'agent': 'VOICE_AGENT',
                    'status': 'success',
                    'text': stt_result['text'],
                    'confidence': stt_result['confidence'],
                    'language': stt_result['language'],
                    'timestamp': datetime.now().isoformat()
                }
            else:
                return {
                    'agent': 'VOICE_AGENT',
                    'status': 'error',
                    'error': stt_result['error'],
                    'text': '',
                    'timestamp': datetime.now().isoformat()
                }

        except Exception as e:
            logger.error(f"Error in voice input processing: {str(e)}")
            return {
                'agent': 'VOICE_AGENT',
                'status': 'error',
                'error': str(e),
                'text': '',
                'timestamp': datetime.now().isoformat()
            }

    def generate_voice_response(self, text: str, voice_id: Optional[str] = None) -> Dict[str, Any]:
        """Generate voice response from text"""
        try:
            # Convert text to speech
            tts_result = self.text_to_speech.generate_speech(text, voice_id)

            if tts_result['status'] == 'success':
                return {
                    'agent': 'VOICE_AGENT',
                    'status': 'success',
                    'audio_data': tts_result['audio_data'],
                    'voice_id': tts_result['voice_id'],
                    'text_length': tts_result['text_length'],
                    'timestamp': datetime.now().isoformat()
                }
            else:
                return {
                    'agent': 'VOICE_AGENT',
                    'status': 'error',
                    'error': tts_result['error'],
                    'audio_data': None,
                    'timestamp': datetime.now().isoformat()
                }

        except Exception as e:
            logger.error(f"Error in voice response generation: {str(e)}")
            return {
                'agent': 'VOICE_AGENT',
                'status': 'error',
                'error': str(e),
                'audio_data': None,
                'timestamp': datetime.now().isoformat()
            }

    def get_voice_info(self) -> Dict[str, Any]:
        """Get information about available voices"""
        return self.text_to_speech.get_available_voices()
