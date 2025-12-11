import base64
from typing import Dict, Any, Optional

import azure.cognitiveservices.speech as speechsdk

from config import config, logger

class TextToSpeechProcessor:
    """Handles text-to-speech conversion using Azure Speech Services"""

    def __init__(self):
        """Initialize the text-to-speech processor"""
        if config.AZURE_SPEECH_KEY and config.AZURE_SPEECH_REGION:
            self.speech_config = speechsdk.SpeechConfig(
                subscription=config.AZURE_SPEECH_KEY, 
                region=config.AZURE_SPEECH_REGION
            )
            self.available = True
            logger.info("Azure TTS initialized successfully")
        else:
            self.speech_config = None
            self.available = False
            logger.warning("Azure Speech credentials not provided, TTS disabled")

    def generate_speech(self, text: str, voice_name: Optional[str] = None) -> Dict[str, Any]:
        """Generate speech from text"""
        if not self.available:
            return {
                'status': 'error',
                'error': 'Text-to-speech not available - Azure credentials not configured',
                'audio_data': None
            }

        try:
            # Use provided voice_name or default from config
            voice = voice_name or config.AZURE_TTS_VOICE_NAME
            if not voice:
                return {'status': 'error', 'error': 'No default voice name provided in config', 'audio_data': None}
            
            self.speech_config.speech_synthesis_voice_name = voice

            # Synthesize to an in-memory stream
            audio_config = speechsdk.audio.AudioOutputConfig(use_default_speaker=False)
            synthesizer = speechsdk.SpeechSynthesizer(speech_config=self.speech_config, audio_config=audio_config)

            result = synthesizer.speak_text_async(text).get()

            if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
                audio_bytes = result.audio_data
                # Convert to base64 data URL
                audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')
                audio_data_url = f"data:audio/mpeg;base64,{audio_base64}"

                return {
                    'status': 'success',
                    'audio_data': audio_data_url,
                    'voice_id': voice, # Kept key as 'voice_id' for consistency
                    'text_length': len(text)
                }
            elif result.reason == speechsdk.ResultReason.Canceled:
                cancellation_details = result.cancellation_details
                error_message = f"Speech synthesis canceled: {cancellation_details.reason}. Error: {cancellation_details.error_details}"
                return {'status': 'error', 'error': error_message, 'audio_data': None}
            else:
                 return {'status': 'error', 'error': f'Unknown synthesis error: {result.reason}', 'audio_data': None}

        except Exception as e:
            logger.error(f"Error in text-to-speech generation: {e}")
            return {'status': 'error', 'error': str(e), 'audio_data': None}


    def get_available_voices(self) -> Dict[str, Any]:
        """Get list of available voices"""
        if not self.available:
            return {'status': 'error', 'error': 'TTS not available', 'voices': []}

        try:
            synthesizer = speechsdk.SpeechSynthesizer(speech_config=self.speech_config)
            result = synthesizer.get_voices_async().get()
            
            if result.reason == speechsdk.ResultReason.VoicesListRetrieved:
                voices_info = []
                for voice in result.voices:
                    voices_info.append({
                        'voice_id': voice.name,
                        'name': voice.local_name,
                        'category': 'Azure Standard' if 'Standard' in voice.voice_type.name else 'Azure Neural',
                        'description': f"{voice.locale}, Gender: {voice.gender.name}"
                    })
                return {'status': 'success', 'voices': voices_info}
            else:
                return {'status': 'error', 'error': 'Failed to retrieve voice list', 'voices': []}

        except Exception as e:
            logger.error(f"Error getting voices: {e}")
            return {'status': 'error', 'error': str(e), 'voices': []}