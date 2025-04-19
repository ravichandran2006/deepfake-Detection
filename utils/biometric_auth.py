import cv2
import numpy as np
from deepface import DeepFace
import os
import json
from datetime import datetime
import wave
import numpy as np
from scipy.io import wavfile
import librosa
from pymongo import MongoClient
import base64
import logging
import speech_recognition as sr
from pydub import AudioSegment
import sys

class BiometricAuth:
    def __init__(self):
        try:
            # Suppress DeepFace model download messages
            original_stdout = sys.stdout
            sys.stdout = open(os.devnull, 'w')

            # Check if VGG-Face weights already exist
            weights_path = os.path.expanduser('~/.deepface/weights/vgg_face_weights.h5')
            if not os.path.exists(weights_path):
                # Restore stdout temporarily for first-time download message
                sys.stdout = original_stdout
                print("\nDownloading VGG-Face weights for the first time (this will only happen once)...")
                sys.stdout = open(os.devnull, 'w')
            
            # Pre-load the model to avoid lazy loading during verification
            DeepFace.build_model("VGG-Face")
            
            # Restore stdout
            sys.stdout = original_stdout

            # MongoDB connection with minimal logging
            connection_string = "mongodb://localhost:27017/?directConnection=true"
            self.client = MongoClient(connection_string)
            
            # Test the connection silently
            self.client.server_info()
            
            # Setup database and collections
            self.db = self.client['deepfake_detection']
            if 'users' not in self.db.list_collection_names():
                self.db.create_collection('users')
            self.users_collection = self.db['users']
            
            # Create an index on username
            self.users_collection.create_index('username', unique=True)
            
            # Create necessary directories
            self.biometric_data_dir = 'data/biometric_data'
            os.makedirs(self.biometric_data_dir, exist_ok=True)
            
            # Initialize speech recognizer
            self.recognizer = sr.Recognizer()
            
        except Exception as e:
            if 'original_stdout' in locals():
                sys.stdout = original_stdout
            raise Exception(f"Failed to initialize BiometricAuth: {str(e)}")
        
    def _convert_wav_to_text(self, audio_path):
        """Convert WAV file to text using speech recognition"""
        try:
            print(f"Processing audio file: {audio_path}")
            print(f"File exists: {os.path.exists(audio_path)}")
            print(f"File size: {os.path.getsize(audio_path)} bytes")
            
            # Configure recognizer with basic settings
            self.recognizer.energy_threshold = 300
            self.recognizer.dynamic_energy_threshold = True
            self.recognizer.pause_threshold = 0.5
            
            # Simple recognition attempt
            with sr.AudioFile(audio_path) as source:
                print("Reading audio file...")
                # Adjust for ambient noise
                self.recognizer.adjust_for_ambient_noise(source, duration=0.2)
                print("Adjusted for ambient noise")
                
                # Record the audio
                print("Recording audio...")
                audio_data = self.recognizer.record(source)
                print("Audio recorded successfully")
                print("Audio data length:", len(audio_data.frame_data))
                
                # Try to recognize the speech
                print("Recognizing speech...")
                text = self.recognizer.recognize_google(audio_data)
                print(f"Recognized text: {text}")
                
                if text and len(text.strip()) > 0:
                    return text.lower().strip()
                else:
                    raise Exception("No text recognized from audio")
            
        except sr.UnknownValueError:
            print("Speech not understood")
            raise Exception("Could not understand speech. Please speak clearly.")
        except sr.RequestError as e:
            print(f"Request error: {e}")
            raise Exception(f"Error with speech recognition service: {str(e)}")
        except Exception as e:
            print(f"Error in _convert_wav_to_text: {str(e)}")
            import traceback
            print(traceback.format_exc())
            raise Exception(f"Failed to process voice: {str(e)}")

    def create_user(self, user_data, face_path, voice_path):
        """Create a new user with biometric data"""
        try:
            username = user_data['username']
            print(f"Creating user: {username}")
            
            # Check if user already exists
            existing_user = self.users_collection.find_one({'username': username})
            if existing_user:
                print(f"User {username} already exists")
                return False
                
            # Get voice password from speech recognition
            voice_password = self._convert_wav_to_text(voice_path)
            if not voice_password:
                print("Could not recognize voice password")
                return False
            
            print(f"Voice password set as: {voice_password}")
            
            # Extract face embedding using DeepFace
            print("Extracting face embedding...")
            face_embedding = DeepFace.represent(face_path, model_name="VGG-Face", enforce_detection=True)[0]["embedding"]
            face_embedding = np.array(face_embedding).tolist()  # Convert to list for MongoDB storage
            
            # Extract voice features for voice verification
            voice_features = self._extract_voice_features(voice_path)
            if voice_features is not None:
                voice_features = voice_features.tolist()
            
            # Save biometric data
            user_biometric_dir = os.path.join(self.biometric_data_dir, username)
            os.makedirs(user_biometric_dir, exist_ok=True)
            
            # Save face embedding
            np.save(os.path.join(user_biometric_dir, 'face_embedding.npy'), np.array(face_embedding))
            
            # Save voice features
            if voice_features:
                np.save(os.path.join(user_biometric_dir, 'voice_features.npy'), np.array(voice_features))
            
            # Create user entry in MongoDB with biometric data
            user_doc = {
                'username': username,
                'email': user_data.get('email', ''),
                'created_at': datetime.now(),
                'face_path': face_path,
                'voice_path': voice_path,
                'voice_password': voice_password,  # Store the voice password
                'face_embedding': face_embedding,
                'voice_features': voice_features,
                'last_login': None
            }
            
            print(f"Inserting user document: {user_doc}")
            result = self.users_collection.insert_one(user_doc)
            print(f"User created with ID: {result.inserted_id}")
            
            # Verify the data was stored
            stored_user = self.users_collection.find_one({'_id': result.inserted_id})
            if stored_user:
                print(f"Successfully verified user data in MongoDB: {stored_user['username']}")
                return True
            else:
                print("Failed to verify user data in MongoDB")
                return False
                
        except Exception as e:
            print(f"Error creating user: {str(e)}")
            import traceback
            print(traceback.format_exc())
            return False
        
    def verify_user(self, username, face_path, voice_path):
        """Verify user identity using face, voice, and voice password"""
        try:
            print(f"\n=== Starting verification for user: {username} ===")
            # Check if user exists
            user = self.users_collection.find_one({'username': username})
            if not user:
                print(f"❌ User {username} not found in database")
                return False
                
            print(f"✓ Found user in database: {username}")
            
            # Step 1: Face Verification
            try:
                print("\n=== Face Verification ===")
                stored_face_path = user['face_path']
                if not os.path.exists(stored_face_path):
                    print(f"❌ Stored face image not found at: {stored_face_path}")
                    return False
                    
                print("1. Comparing face images...")
                print(f"   - Login image: {face_path}")
                print(f"   - Stored image: {stored_face_path}")
                
                face_result = DeepFace.verify(
                    img1_path=face_path,
                    img2_path=stored_face_path,
                    model_name="VGG-Face",
                    enforce_detection=True
                )
                
                print("\n2. Face Verification Results:")
                print(f"   - Verified: {face_result['verified']}")
                print(f"   - Distance: {face_result.get('distance', 'N/A')}")
                print(f"   - Threshold: {face_result.get('threshold', 'N/A')}")
                print(f"   - Model: {face_result.get('model', 'VGG-Face')}")
                
                if not face_result['verified']:
                    print("\n❌ Face verification failed - faces don't match")
                    return False
                    
                print("\n✓ Face verification successful!")
                
                # Step 2: Voice Password Verification
                print("\n=== Voice Password Verification ===")
                spoken_text = self._convert_wav_to_text(voice_path)
                if not spoken_text:
                    print("❌ Could not recognize spoken text")
                    return False
                
                stored_password = user['voice_password']
                print(f"Comparing voice passwords:")
                print(f"Spoken: {spoken_text}")
                print(f"Stored: {stored_password}")
                
                if spoken_text != stored_password:
                    print("❌ Voice password does not match")
                    return False
                
                print("✓ Voice password verified!")
                
                # Step 3: Voice Biometric Verification
                print("\n=== Voice Biometric Verification ===")
                current_voice_features = self._extract_voice_features(voice_path)
                stored_voice_features = np.array(user['voice_features'])
                
                if current_voice_features is None:
                    print("❌ Could not extract voice features")
                    return False
                
                # Compare voice features
                voice_match = self._compare_voice_features(current_voice_features, stored_voice_features)
                
                if not voice_match:
                    print("❌ Voice biometrics do not match")
                    return False
                
                print("✓ Voice biometrics verified!")
                
                print("\n=== Overall Verification Result ===")
                print("✓ All verifications passed!")
                return True
                
            except Exception as e:
                print(f"\n❌ Error during verification: {e}")
                import traceback
                print(traceback.format_exc())
                return False
                
        except Exception as e:
            print(f"\n❌ Error verifying user: {str(e)}")
            import traceback
            print(traceback.format_exc())
            return False
        
    def verify_biometric_data(self, face_path, voice_path):
        """Verify that biometric data is real (not deepfake)"""
        try:
            print("Verifying biometric data...")
            # Analyze face using DeepFace
            face_analysis = DeepFace.analyze(face_path, actions=['emotion', 'age', 'gender'], enforce_detection=True)
            
            # Check if face detection was successful
            if not face_analysis:
                print("Face analysis failed")
                return False
            
            print("Face analysis successful")
            
            # Verify voice authenticity
            voice_features = self._extract_voice_features(voice_path)
            if voice_features is None:
                print("Voice feature extraction failed")
                return False
            
            print("Voice feature extraction successful")
            return True
        except Exception as e:
            print(f"Error verifying biometric data: {str(e)}")
            import traceback
            print(traceback.format_exc())
            return False
        
    def _extract_voice_features(self, voice_path):
        """Extract features from voice recording"""
        try:
            print(f"   Extracting features from: {voice_path}")
            # Load audio file
            y, sr = librosa.load(voice_path)
            
            # Extract MFCC features
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            
            # Extract pitch
            pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
            
            # Calculate statistics
            mfcc_mean = np.mean(mfcc, axis=1)
            mfcc_std = np.std(mfcc, axis=1)
            pitch_mean = np.mean(pitches[magnitudes > np.max(magnitudes)/2])
            
            features = np.concatenate([mfcc_mean, mfcc_std, [pitch_mean]])
            print(f"   ✓ Successfully extracted {len(features)} features")
            return features
            
        except Exception as e:
            print(f"   ❌ Error extracting voice features: {e}")
            import traceback
            print(traceback.format_exc())
            return None
            
    def _compare_voice_features(self, current_features, stored_features):
        """Compare voice features for authentication"""
        try:
            if current_features is None or stored_features is None:
                print("❌ Missing voice features for comparison")
                return False
                
            # Calculate Euclidean distance
            distance = np.linalg.norm(current_features - stored_features)
            print("\n2. Voice Comparison Results:")
            print(f"   - Distance: {distance}")
            
            # Threshold for voice matching (adjust this value based on testing)
            threshold = 2.0  # Increased threshold for better matching
            print(f"   - Threshold: {threshold}")
            
            result = distance < threshold
            print(f"   - Match: {result}")
            return result
            
        except Exception as e:
            print(f"❌ Error comparing voice features: {e}")
            import traceback
            print(traceback.format_exc())
            return False

    def verify_user_detailed(self, username, face_image_path, voice_sample_path):
        """
        Verify a user's identity with detailed results.
        Focuses on username check, deepfake detection, and voice password word.
        """
        try:
            # Get stored user data
            user_data = self.users_collection.find_one({'username': username})
            if not user_data:
                return {
                    'overall_success': False,
                    'error': 'Username not found. Please check your username and try again.'
                }

            stored_voice_password = user_data.get('voice_password')

            # Initialize results
            results = {
                'overall_success': False,
                'username_valid': True,
                'deepfake_detected': False,
                'voice_word_matched': False,
                'error': None
            }

            # Check for deepfake
            try:
                img = cv2.imread(face_image_path)
                if img is None:
                    results['error'] = 'Could not process the image. Please try again with a clear photo.'
                    return results

                # 1. Check for unrealistic color patterns
                hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
                saturation = hsv[:, :, 1].mean()
                value = hsv[:, :, 2].mean()
                
                if saturation > 120 or value > 200:
                    results['deepfake_detected'] = True
                    results['error'] = 'Potential deepfake detected: Unusual color patterns in the image. Please use a natural photo.'
                    return results

                # 2. Check face detection and proportions
                face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
                faces = face_cascade.detectMultiScale(img, 1.1, 4)
                
                if len(faces) == 0:
                    results['error'] = 'No face detected in the image. Please ensure your face is clearly visible.'
                    return results
                elif len(faces) > 1:
                    results['deepfake_detected'] = True
                    results['error'] = 'Multiple faces detected. Please provide a photo with only your face.'
                    return results

                # 3. Check face proportions
                for (x, y, w, h) in faces:
                    aspect_ratio = float(w) / h
                    if aspect_ratio < 0.5 or aspect_ratio > 1.5:
                        results['deepfake_detected'] = True
                        results['error'] = 'Unusual face proportions detected. Please provide a natural front-facing photo.'
                        return results

            except Exception as e:
                print(f"Error in image processing: {str(e)}")
                results['error'] = 'Error processing the image. Please try again with a different photo.'
                return results

            # Verify voice password word
            try:
                print(f"Processing voice sample: {voice_sample_path}")
                print(f"Voice sample exists: {os.path.exists(voice_sample_path)}")
                print(f"Voice sample size: {os.path.getsize(voice_sample_path)} bytes")
                
                # Use our improved voice processing method
                voice_text = self._convert_wav_to_text(voice_sample_path)
                
                if voice_text is None:
                    results['error'] = 'Could not understand the spoken word. Please speak clearly and try again.'
                    return results
                
                print(f"Recognized voice text: {voice_text}")
                print(f"Stored voice password: {stored_voice_password}")
                
                # Compare the spoken word with stored password
                results['voice_word_matched'] = voice_text == stored_voice_password.lower()
                if not results['voice_word_matched']:
                    results['error'] = f'Voice password did not match. You said "{voice_text}" but the stored password is "{stored_voice_password}". Please say the correct password word clearly.'
                    return results

            except sr.UnknownValueError:
                results['error'] = 'Could not understand the spoken word. Please speak clearly and try again.'
                return results
            except Exception as e:
                print(f"Error in voice processing: {str(e)}")
                import traceback
                print(traceback.format_exc())
                results['error'] = f'Error processing voice: {str(e)}. Please try speaking again.'
                return results

            # Set overall success if everything passes
            results['overall_success'] = (
                results['username_valid'] and 
                not results['deepfake_detected'] and 
                results['voice_word_matched']
            )

            if results['overall_success']:
                results['error'] = None

            return results

        except Exception as e:
            print(f"Error in verify_user_detailed: {str(e)}")
            return {
                'overall_success': False,
                'error': 'An unexpected error occurred. Please try again.'
            } 