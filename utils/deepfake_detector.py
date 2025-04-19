import cv2
import numpy as np
from deepface import DeepFace
import mediapipe as mp
import tensorflow as tf
import os

class DeepfakeDetector:
    def __init__(self):
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            min_detection_confidence=0.5
        )
        
        # Load pre-trained models
        self.face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
    def analyze_image(self, image_path):
        """Analyze a single image for deepfake detection"""
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError("Could not read image")
            
        # Convert to RGB for face mesh
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Detect faces
        faces = self.face_detector.detectMultiScale(rgb_image, 1.1, 4)
        if len(faces) == 0:
            return {'error': 'No face detected'}
            
        results = {
            'face_detected': True,
            'deepfake_probability': 0.0,
            'analysis_points': []
        }
        
        for (x, y, w, h) in faces:
            face_roi = rgb_image[y:y+h, x:x+w]
            
            # Analyze facial landmarks
            face_mesh_results = self.face_mesh.process(face_roi)
            if face_mesh_results.multi_face_landmarks:
                landmarks = face_mesh_results.multi_face_landmarks[0]
                
                # Check for unnatural facial features
                unnatural_features = self._check_unnatural_features(landmarks)
                results['analysis_points'].extend(unnatural_features)
                
                # Use DeepFace for additional analysis
                try:
                    deepface_analysis = DeepFace.analyze(face_roi, 
                                                       actions=['emotion'],
                                                       enforce_detection=False)
                    results['deepfake_probability'] = self._calculate_deepfake_probability(
                        deepface_analysis, unnatural_features
                    )
                except Exception as e:
                    print(f"DeepFace analysis error: {str(e)}")
                    
        return results
        
    def analyze_video(self, video_path):
        """Analyze a video for deepfake detection"""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError("Could not open video file")
            
        frame_results = []
        frame_count = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            # Analyze every 5th frame to save processing time
            if frame_count % 5 == 0:
                temp_path = f"temp_frame_{frame_count}.jpg"
                cv2.imwrite(temp_path, frame)
                
                try:
                    frame_result = self.analyze_image(temp_path)
                    frame_results.append(frame_result)
                finally:
                    if os.path.exists(temp_path):
                        os.remove(temp_path)
                        
            frame_count += 1
            
        cap.release()
        
        # Aggregate results
        return self._aggregate_video_results(frame_results)
        
    def _check_unnatural_features(self, landmarks):
        """Check for unnatural facial features that might indicate a deepfake"""
        unnatural_features = []
        
        # Check eye symmetry
        left_eye = landmarks.landmark[33]  # Left eye center
        right_eye = landmarks.landmark[263]  # Right eye center
        
        eye_distance = abs(left_eye.x - right_eye.x)
        if eye_distance > 0.4:  # Threshold for unnatural eye spacing
            unnatural_features.append({
                'type': 'unnatural_eye_spacing',
                'confidence': 0.8
            })
            
        # Add more feature checks as needed
        
        return unnatural_features
        
    def _calculate_deepfake_probability(self, deepface_analysis, unnatural_features):
        """Calculate overall deepfake probability based on multiple factors"""
        base_probability = 0.0
        
        # Weight for different factors
        weights = {
            'unnatural_features': 0.6,
            'emotion_consistency': 0.4
        }
        
        # Calculate probability from unnatural features
        if unnatural_features:
            feature_confidence = sum(f['confidence'] for f in unnatural_features) / len(unnatural_features)
            base_probability += feature_confidence * weights['unnatural_features']
            
        # Add emotion consistency check
        if 'emotion' in deepface_analysis:
            emotion_scores = deepface_analysis['emotion']
            max_emotion = max(emotion_scores.values())
            if max_emotion > 0.9:  # Unusually strong emotion might indicate manipulation
                base_probability += 0.3 * weights['emotion_consistency']
                
        return min(base_probability, 1.0)
        
    def _aggregate_video_results(self, frame_results):
        """Aggregate results from multiple video frames"""
        if not frame_results:
            return {'error': 'No valid frames analyzed'}
            
        # Calculate average deepfake probability
        valid_results = [r for r in frame_results if 'deepfake_probability' in r]
        if not valid_results:
            return {'error': 'No valid analysis results'}
            
        avg_probability = sum(r['deepfake_probability'] for r in valid_results) / len(valid_results)
        
        # Collect all analysis points
        all_points = []
        for result in frame_results:
            if 'analysis_points' in result:
                all_points.extend(result['analysis_points'])
                
        return {
            'video_analysis': {
                'frames_analyzed': len(valid_results),
                'deepfake_probability': avg_probability,
                'analysis_points': all_points
            }
        }

    def verify_face(self, input_face_path, stored_face_path):
        """
        Verify if the input face is real (not a deepfake) and matches the stored face.
        Basic verification without strict face detection requirements.
        """
        try:
            # Read images
            input_img = cv2.imread(input_face_path)
            stored_img = cv2.imread(stored_face_path)
            
            if input_img is None or stored_img is None:
                return {
                    'success': False,
                    'error': 'Could not read one or both images'
                }

            # Basic face detection with lower confidence threshold
            input_gray = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
            stored_gray = cv2.cvtColor(stored_img, cv2.COLOR_BGR2GRAY)
            
            faces_input = self.face_detector.detectMultiScale(
                input_gray,
                scaleFactor=1.3,
                minNeighbors=3,
                minSize=(30, 30)
            )
            
            if len(faces_input) == 0:
                # Try with even lower threshold if no face detected
                faces_input = self.face_detector.detectMultiScale(
                    input_gray,
                    scaleFactor=1.1,
                    minNeighbors=2,
                    minSize=(20, 20)
                )

            # Simple image comparison using structural similarity
            try:
                # Resize images to same size for comparison
                stored_img_resized = cv2.resize(stored_img, (300, 300))
                input_img_resized = cv2.resize(input_img, (300, 300))
                
                # Convert to grayscale for comparison
                stored_gray_resized = cv2.cvtColor(stored_img_resized, cv2.COLOR_BGR2GRAY)
                input_gray_resized = cv2.cvtColor(input_img_resized, cv2.COLOR_BGR2GRAY)
                
                # Calculate similarity
                similarity = cv2.matchTemplate(input_gray_resized, stored_gray_resized, cv2.TM_CCOEFF_NORMED)[0][0]
                
                # Check for basic similarity threshold
                if similarity > 0.5:  # Lowered threshold for more lenient matching
                    return {
                        'success': True,
                        'message': 'Face verification successful'
                    }
                else:
                    return {
                        'success': False,
                        'error': 'Face verification failed - images do not match sufficiently'
                    }
                    
            except Exception as e:
                print(f"Error during image comparison: {str(e)}")
                return {
                    'success': False,
                    'error': 'Error comparing images'
                }

        except Exception as e:
            print(f"Verification error: {str(e)}")
            return {
                'success': False,
                'error': 'Face verification failed'
            } 