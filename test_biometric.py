from utils.biometric_auth import BiometricAuth
from utils.deepfake_detector import DeepfakeDetector
import os

def main():
    # Create test directories if they don't exist
    os.makedirs('data/test_images', exist_ok=True)
    os.makedirs('data/test_audio', exist_ok=True)
    
    # Initialize the classes
    bio_auth = BiometricAuth()
    deepfake_detector = DeepfakeDetector()
    
    print("Biometric Authentication System Test")
    print("-----------------------------------")
    
    # Test image path (you need to provide an actual image)
    test_image = 'data/test_images/test_face.jpg'
    test_audio = 'data/test_audio/test_voice.wav'
    
    if not os.path.exists(test_image):
        print(f"Please place a test face image at: {test_image}")
        return
        
    if not os.path.exists(test_audio):
        print(f"Please place a test voice recording at: {test_audio}")
        return
    
    # Test deepfake detection
    print("\nTesting Deepfake Detection...")
    try:
        results = deepfake_detector.analyze_image(test_image)
        print("Deepfake Analysis Results:")
        print(f"Face Detected: {results.get('face_detected', False)}")
        print(f"Deepfake Probability: {results.get('deepfake_probability', 0.0):.2%}")
        if 'analysis_points' in results:
            print("\nAnalysis Points:")
            for point in results['analysis_points']:
                print(f"- {point['type']}: {point['confidence']:.2%}")
    except Exception as e:
        print(f"Error in deepfake detection: {str(e)}")
    
    # Test user creation
    print("\nTesting User Creation...")
    try:
        user_data = {
            'username': 'test_user'
        }
        creation_success = bio_auth.create_user(user_data, test_image, test_audio)
        print(f"User creation {'successful' if creation_success else 'failed'}")
        
        # Test user verification
        if creation_success:
            print("\nTesting User Verification...")
            verification_success = bio_auth.verify_user('test_user', test_image, test_audio)
            print(f"User verification {'successful' if verification_success else 'failed'}")
    except Exception as e:
        print(f"Error in biometric authentication: {str(e)}")

if __name__ == "__main__":
    main() 