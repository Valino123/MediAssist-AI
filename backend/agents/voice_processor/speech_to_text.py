import base64
import io
from typing import Dict, Any

import azure.cognitiveservices.speech as speechsdk
from pydub import AudioSegment

from config import config, logger

class SpeechToTextProcessor:
    """Handles speech-to-text conversion using Azure Speech Services"""

    def __init__(self):
        """Initialize the speech-to-text processor"""
        try:
            self.speech_config = speechsdk.SpeechConfig(
                subscription=config.AZURE_SPEECH_KEY, 
                region=config.AZURE_SPEECH_REGION
            )
            # Set the language from your config
            self.speech_config.speech_recognition_language = config.SPEECH_RECOGNITION_LANGUAGE
            logger.info("Azure STT initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize Azure SpeechConfig: {e}")
            self.speech_config = None

    def process_audio_file(self, audio_data: str) -> Dict[str, Any]:
        """Process a base64 encoded audio string and convert to text"""
        if not self.speech_config:
            return {'status': 'error', 'error': 'Azure Speech Service not configured', 'text': ''}

        try:
            # Decode base64 audio data
            if audio_data.startswith('data:audio'):
                audio_data = audio_data.split(',')[1]
            audio_bytes = base64.b64decode(audio_data)

            # Use pydub to ensure the audio is in the correct WAV format for Azure
            audio_segment = AudioSegment.from_file(io.BytesIO(audio_bytes))
            # Azure works best with 16-bit, 16kHz mono PCM WAV
            audio_segment = audio_segment.set_frame_rate(16000).set_sample_width(2).set_channels(1)
            
            wav_data = audio_segment.export(format="wav").read()

            # Create an audio configuration from the in-memory WAV data
            audio_config = speechsdk.audio.AudioConfig(stream=speechsdk.audio.PushAudioInputStream())
            recognizer = speechsdk.SpeechRecognizer(speech_config=self.speech_config, audio_config=audio_config)
            
            # Write the audio data to the stream and close it
            audio_config.stream.write(wav_data)
            audio_config.stream.close()

            # Perform recognition
            result = recognizer.recognize_once()

            # Process the result
            if result.reason == speechsdk.ResultReason.RecognizedSpeech:
                return {
                    'status': 'success',
                    'text': result.text,
                    'confidence': result.confidence,
                    'language': config.SPEECH_RECOGNITION_LANGUAGE
                }
            elif result.reason == speechsdk.ResultReason.NoMatch:
                return {'status': 'error', 'error': 'Could not understand audio', 'text': ''}
            elif result.reason == speechsdk.ResultReason.Canceled:
                cancellation_details = result.cancellation_details
                error_message = f"Speech recognition canceled: {cancellation_details.reason}. Error: {cancellation_details.error_details}"
                return {'status': 'error', 'error': error_message, 'text': ''}
            else:
                return {'status': 'error', 'error': f'Unknown recognition error: {result.reason}', 'text': ''}

        except Exception as e:
            logger.error(f"Error in speech-to-text processing: {e}")
            return {'status': 'error', 'error': str(e), 'text': ''}


    def process_microphone_input(self) -> Dict[str, Any]:
        """Process live microphone input"""
        if not self.speech_config:
            return {'status': 'error', 'error': 'Azure Speech Service not configured', 'text': ''}

        audio_config = speechsdk.audio.AudioConfig(use_default_microphone=True)
        recognizer = speechsdk.SpeechRecognizer(speech_config=self.speech_config, audio_config=audio_config)

        try:
            logger.info("Listening...")
            result = recognizer.recognize_once()

            if result.reason == speechsdk.ResultReason.RecognizedSpeech:
                return {
                    'status': 'success',
                    'text': result.text,
                    'confidence': result.confidence,
                    'language': config.SPEECH_RECOGNITION_LANGUAGE
                }
            elif result.reason == speechsdk.ResultReason.NoMatch:
                return {'status': 'error', 'error': 'No speech detected or could not understand audio', 'text': ''}
            elif result.reason == speechsdk.ResultReason.Canceled:
                cancellation_details = result.cancellation_details
                error_message = f"Speech recognition canceled: {cancellation_details.reason}"
                if cancellation_details.reason == speechsdk.CancellationReason.Error:
                    error_message += f". Error details: {cancellation_details.error_details}"
                return {'status': 'error', 'error': error_message, 'text': ''}
            else:
                return {'status': 'error', 'error': f'Unknown recognition error: {result.reason}', 'text': ''}
        
        except Exception as e:
            logger.error(f"Error in microphone processing: {e}")
            return {'status': 'error', 'error': str(e), 'text': ''}