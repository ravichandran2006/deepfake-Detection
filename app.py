from flask import Flask, request, jsonify, render_template, send_file, redirect, url_for, flash, session
from flask_cors import CORS
import os
from datetime import datetime
import cv2
import numpy as np
from deepface import DeepFace
import mediapipe as mp
from utils.deepfake_detector import DeepfakeDetector
from utils.biometric_auth import BiometricAuth
from utils.pdf_generator import PDFGenerator
import json
import base64
import traceback
from pymongo import MongoClient
from flask_login import login_user
import logging
import sys
import speech_recognition as sr
from bson.objectid import ObjectId

# Suppress INFO and DEBUG logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Redirect stdout to devnull during model loading
original_stdout = sys.stdout
sys.stdout = open(os.devnull, 'w')

app = Flask(__name__)
CORS(app)
app.secret_key = 'deepfake_detection_secret_key'

# MongoDB setup
try:
    mongo_client = MongoClient('mongodb://localhost:27017/')
    db = mongo_client['deepfake_detection']
    users_collection = db['users']
    # Restore stdout before printing connection message
    sys.stdout = original_stdout
    print("\nMongoDB connected successfully!")
except Exception as e:
    sys.stdout = original_stdout
    print(f"\nMongoDB connection error: {e}")

# Initialize components silently
deepfake_detector = DeepfakeDetector()
biometric_auth = BiometricAuth()
pdf_generator = PDFGenerator()

# Configure upload folder
UPLOAD_FOLDER = 'static/uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

@app.route('/')
def index():
    if 'user_id' in session:
        return redirect(url_for('dashboard'))
    return render_template('index.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        try:
            # Get user data from JSON
            data = request.get_json()
            if not data:
                return jsonify({'error': 'No data provided'}), 400

            username = data.get('username')
            email = data.get('email')
            face_image_data = data.get('face_image')
            voice_password = data.get('voice_password')

            # Validate required fields
            if not all([username, email, face_image_data, voice_password]):
                return jsonify({'error': 'Missing required fields'}), 400
            
            # Check if username already exists
            if users_collection.find_one({'username': username}):
                return jsonify({'error': 'Username already exists'}), 400

            # Save face image
            try:
                face_path = os.path.join(UPLOAD_FOLDER, f"{username}_face.jpg")
                face_image_data = face_image_data.split(',')[1]  # Remove data URL prefix
                with open(face_path, 'wb') as f:
                    f.write(base64.b64decode(face_image_data))
            except Exception as e:
                print(f"Error saving face image: {e}")
                return jsonify({'error': 'Invalid face image data'}), 400

            # Create user document
            user_doc = {
                'username': username,
                'email': email,
                'created_at': datetime.now(),
                'face_path': face_path,
                'voice_password': voice_password.lower()  # Store password in lowercase
            }
            
            # Insert into MongoDB
            try:
                result = users_collection.insert_one(user_doc)
                print(f"User created with ID: {result.inserted_id}")
                
                return jsonify({
                    'status': 'success',
                    'message': 'User created successfully',
                    'redirect': url_for('login')
                }), 201
            except Exception as e:
                print(f"Error inserting into MongoDB: {e}")
                return jsonify({'error': 'Database error'}), 500
                
        except Exception as e:
            print(f"Error during signup: {str(e)}")
            print(traceback.format_exc())
            return jsonify({'error': f'Server error: {str(e)}'}), 500
            
    return render_template('signup.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        try:
            # Get data from JSON request
            data = request.get_json()
            if not data:
                return jsonify({'error': 'No data provided'}), 400

            username = data.get('username')
            face_image = data.get('face_image')
            voice_password = data.get('voice_password')

            # Validate required fields
            if not all([username, face_image, voice_password]):
                return jsonify({
                    'error': 'Missing required fields. Please provide username, face image, and voice password.'
                }), 400

            # Get user from database
            user = users_collection.find_one({'username': username})
            if not user:
                return jsonify({
                    'error': 'Username not found. Please check your username and try again.'
                }), 401

            # Save face image to temporary file
            temp_face_path = f'temp_face_{username}.jpg'
            try:
                face_data = face_image.split(',')[1]  # Remove data URL prefix
                with open(temp_face_path, 'wb') as f:
                    f.write(base64.b64decode(face_data))
            except Exception as e:
                print(f"Error saving face image: {e}")
                return jsonify({'error': 'Invalid face image data'}), 400

            try:
                # Verify face
                face_verification = deepfake_detector.verify_face(temp_face_path, user['face_path'])
                
                # Check if face verification passed
                if not face_verification['success']:
                    os.remove(temp_face_path)
                    return jsonify({
                        'error': face_verification.get('error', 'Face verification failed')
                    }), 401

                # Check if the voice password matches (case-insensitive)
                if voice_password.lower() != user['voice_password'].lower():
                    os.remove(temp_face_path)
                    return jsonify({
                        'error': 'Incorrect voice password'
                    }), 401

                # If we get here, all verifications passed
                os.remove(temp_face_path)
                session['user_id'] = str(user['_id'])
                return jsonify({
                    'message': 'Login successful',
                    'redirect': url_for('dashboard')
                })

            except Exception as e:
                print(f"Error during verification: {e}")
                if os.path.exists(temp_face_path):
                    os.remove(temp_face_path)
                return jsonify({'error': f'Verification error: {str(e)}'}), 500

        except Exception as e:
            print(f"Login error: {str(e)}")
            print(traceback.format_exc())
            return jsonify({'error': f'An error occurred during login: {str(e)}'}), 500

    return render_template('login.html')

@app.route('/dashboard')
def dashboard():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    # Get user information from database
    user = users_collection.find_one({'_id': ObjectId(session['user_id'])})
    if not user:
        session.clear()
        return redirect(url_for('login'))
    
    return render_template('dashboard.html', 
                         username=user['username'],
                         email=user['email'])

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

@app.route('/analyze_video', methods=['POST'])
def analyze_video():
    # Check if user is logged in
    if 'user_id' not in session:
        return jsonify({'error': 'Please login to analyze videos'}), 401
        
    if 'video' not in request.files:
        return jsonify({'error': 'No video file provided'}), 400
        
    video_file = request.files['video']
    video_path = os.path.join(UPLOAD_FOLDER, 'temp_video.mp4')
    video_file.save(video_path)
    
    # Analyze video for deepfake
    results = deepfake_detector.analyze_video(video_path)
    
    # Generate PDF report
    report_path = pdf_generator.generate_report(results)
    
    # Clean up video file
    os.remove(video_path)
    
    return jsonify({
        'results': results,
        'report_url': f'/download_report/{os.path.basename(report_path)}'
    })

@app.route('/download_report/<filename>')
def download_report(filename):
    return send_file(
        os.path.join(UPLOAD_FOLDER, filename),
        as_attachment=True,
        download_name=f"deepfake_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
    )

@app.route('/process_voice', methods=['POST'])
def process_voice():
    try:
        print("Processing voice request...")
        print("Request method:", request.method)
        print("Request content type:", request.content_type)
        print("Form data keys:", request.form.keys())
        
        if 'voice_sample' not in request.form:
            print("No voice sample in form data")
            return jsonify({'success': False, 'error': 'No voice sample provided'})
            
        voice_data = request.form['voice_sample']
        if not voice_data:
            print("Empty voice sample data")
            return jsonify({'success': False, 'error': 'Empty voice sample'})
            
        print("Voice sample data length:", len(voice_data))
        
        # Create a unique filename
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        temp_filename = f'temp_voice_{timestamp}.wav'
        temp_path = os.path.join(UPLOAD_FOLDER, temp_filename)
        
        # Ensure upload folder exists
        os.makedirs(UPLOAD_FOLDER, exist_ok=True)
        
        try:
            # Decode base64 data
            voice_bytes = base64.b64decode(voice_data.split(',')[1])
            
            # Save the audio file
            with open(temp_path, 'wb') as f:
                f.write(voice_bytes)
                
            print(f"Saved voice file to: {temp_path}")
            print(f"File exists: {os.path.exists(temp_path)}")
            print(f"File size: {os.path.getsize(temp_path)} bytes")
            
            # Initialize speech recognizer
            recognizer = sr.Recognizer()
            
            # Read the audio file
            with sr.AudioFile(temp_path) as source:
                print("Reading audio file...")
                audio = recognizer.record(source)
                
            print("Attempting speech recognition...")
            # Perform speech recognition
            text = recognizer.recognize_google(audio)
            print(f"Recognized text: {text}")
            
            # Clean up the temporary file
            os.remove(temp_path)
            print("Cleaned up temporary file")
            
            return jsonify({
                'success': True,
                'voice_password': text.lower()
            })
            
        except base64.binascii.Error as e:
            print(f"Base64 decoding error: {str(e)}")
            return jsonify({'success': False, 'error': 'Invalid audio data format'})
        except sr.UnknownValueError:
            print("Speech recognition could not understand audio")
            return jsonify({'success': False, 'error': 'Could not understand audio. Please speak more clearly.'})
        except sr.RequestError as e:
            print(f"Speech recognition service error: {str(e)}")
            return jsonify({'success': False, 'error': 'Speech recognition service error. Please try again.'})
        except Exception as e:
            print(f"Error processing voice: {str(e)}")
            return jsonify({'success': False, 'error': f'Error processing voice: {str(e)}'})
            
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        return jsonify({'success': False, 'error': f'Unexpected error: {str(e)}'})
    finally:
        # Clean up temporary file if it exists
        if 'temp_path' in locals() and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
                print("Cleaned up temporary file in finally block")
            except Exception as e:
                print(f"Error cleaning up file: {str(e)}")

if __name__ == '__main__':
    # Restore stdout for Flask messages
    sys.stdout = original_stdout
    print("\nStarting Flask server...")
    app.run(debug=True, port=5000) 