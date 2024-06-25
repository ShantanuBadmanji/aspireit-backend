import os
import openai
import sounddevice as sd
import scipy.io.wavfile as wav
from sentence_transformers import SentenceTransformer, util
import torch
from faster_whisper import WhisperModel
from collections import defaultdict
import pickle
import pyttsx3
from flask import Flask, request, jsonify, send_file
import requests
from werkzeug.utils import secure_filename
# Load your OpenAI API Key from an environment variable for security
openai.api_key = os.getenv('OPENAI_API_KEY')

# Ensure the directory containing ffmpeg is in the PATH environment variable
os.environ['PATH'] += os.pathsep + '/usr/local/bin'  # Update this path for macOS

# Load Sentence Transformer Model for semantic similarity
semantic_model = SentenceTransformer('all-MiniLM-L6-v2')

app = Flask(__name__)

# Set the directory for storing audio files
audio_file_directory = './audio_files'

# Create the directory if it does not exist
if not os.path.exists(audio_file_directory):
    os.makedirs(audio_file_directory)

# Initialize pyttsx3 for TTS (Windows)
tts_engine = pyttsx3.init()

# Set up Whisper model for TTS
tts_model = WhisperModel("distil-large-v3", device="cpu", compute_type="int8")

device = "cuda" if torch.cuda.is_available() else "cpu"
model = WhisperModel("base", device=device, compute_type="float16" if device == "cuda" else "int8")

def speech_to_text(audio_file_path):
    if not os.path.isfile(audio_file_path):
        print(f"Error: audio file {audio_file_path} not found.")
        return ""
    try:
        segments, info = model.transcribe(audio_file_path)
        # Extract text from each segment if segments are not strings
        text_output = ' '.join(segment.text for segment in segments)
        return text_output
    except Exception as e:
        print(f"Error transcribing audio file {audio_file_path}: {e}")
        return ""
    
# Record audio function
def record_audio(filename, duration=60, sample_rate=16000):
    print(f"Recording for {duration} seconds. Please answer the question...")
    recording = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1)
    sd.wait()  # Wait until recording is finished
    wav.write(filename, sample_rate, recording)
    print(f"Recording completed and saved as {filename}.")
    return filename  # Return the path of the recorded file

# Speak function using pyttsx3
def speak(text):
    tts_engine.say(text)
    tts_engine.runAndWait()

# Generate interview question using OpenAI
def generate_question(topic):
    prompt = f"Generate a concise interview question (max 20 words) about {topic}."
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=100,
        temperature=0.6
    )
    return response['choices'][0]['message']['content']



def generate_followup_question(topic, context):
    prompt = f"Generate a concise follow-up question (max 20 words) about {topic} based on the candidate's previous response: {context}"
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=100,
        temperature=0.6
    )
    return response['choices'][0]['message']['content']


# Generate reference answer using OpenAI
def generate_reference_answer(question):
    prompt = f"Provide a comprehensive answer for the following AI engineering interview question: {question}"
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a knowledgeable assistant."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=150,
        temperature=0.6
    )
    return response['choices'][0]['message']['content']

# Compare candidate answer with reference answer
def compare_answers(candidate_answer, reference_answer):
    candidate_embedding = semantic_model.encode(candidate_answer)
    reference_embedding = semantic_model.encode(reference_answer)
    similarity_score = util.pytorch_cos_sim(candidate_embedding, reference_embedding).item()
    return similarity_score * 100  # Normalize to 100

# Evaluate candidate response
def evaluate_candidate_response(candidate_response, reference_answer):
    relevance_score = compare_answers(candidate_response, reference_answer)  # Normalize to 100
    return {
        "Relevance Score": relevance_score
    }

# Check if response is meaningful
def is_meaningful_response(response):
    if response is None:
        return False
    response = response.lower().strip()
    meaningless_responses = [
        "i don't know", "i do not know", "i'm not sure", "i cannot answer",
        "i don't understand", "thank you", "no comment"
    ]
    for meaningless in meaningless_responses:
        if meaningless in response:
            return False
    return len(response.split()) > 5  # Check if the response has more than 5 words

# Conduct the interview process
def conduct_interview(topics, recording_duration=60):
    results = []
    report_data = defaultdict(list)
    total_questions_asked = 0

    for idx, topic in enumerate(topics):
        question = generate_question(topic)
        print(f"Question {idx + 1}: {question}")
        total_questions_asked += 1
        
        # Speak the question using pyttsx3
        speak(question)

        audio_filename = os.path.join(audio_file_directory, f'recorded_response_{idx + 1}.wav')
        audio_file_path = record_audio(audio_filename, duration=recording_duration)

        try:
            candidate_response = speech_to_text(audio_file_path)
            if not is_meaningful_response(candidate_response):
                print("No meaningful response detected. Moving to the next question.")
                candidate_response = None

            if candidate_response:
                reference_answer = generate_reference_answer(question)

                scores = evaluate_candidate_response(candidate_response, reference_answer)
                print(f"Candidate's Response: {candidate_response}")

                if scores["Relevance Score"] > 50:  # Threshold for correct/partial answers
                    followup_question = generate_followup_question(topic, candidate_response)
                    print(f"Follow-up Question: {followup_question}")
                    total_questions_asked += 1

                    # Speak the follow-up question using pyttsx3
                    speak(followup_question)

                    # Record the follow-up response
                    followup_audio_filename = os.path.join(audio_file_directory, f'followup_response_{idx + 1}.wav')
                    followup_audio_file_path = record_audio(followup_audio_filename, duration=recording_duration)
                    candidate_followup_response = speech_to_text(followup_audio_file_path)
                    if is_meaningful_response(candidate_followup_response):
                        followup_scores = evaluate_candidate_response(candidate_followup_response, reference_answer)
                        scores["Follow-up Relevance Score"] = followup_scores["Relevance Score"]

                report_data["Question"].append(question)
                report_data["Response"].append(candidate_response)
                report_data["Relevance Score"].append(scores["Relevance Score"])

                if "Follow-up Relevance Score" in scores:   
                    report_data["Follow-up Question"].append(followup_question)
                    report_data["Follow-up Response"].append(candidate_followup_response)
                    report_data["Follow-up Relevance Score"].append(scores["Follow-up Relevance Score"])
                else:
                    report_data["Follow-up Question"].append(None)
                    report_data["Follow-up Response"].append(None)
                    report_data["Follow-up Relevance Score"].append(None)

                results.append({
                    "Question": question,
                    "Response": candidate_response,
                    "Relevance Score": scores["Relevance Score"],
                    "Follow-up Question": followup_question if "Follow-up Relevance Score" in scores else None,
                    "Follow-up Response": candidate_followup_response if "Follow-up Relevance Score" in scores else None,
                    "Follow-up Relevance Score": scores["Follow-up Relevance Score"] if "Follow-up Relevance Score" in scores else None
                })
        except Exception as e:
            print(f"Error during interview process for question '{question}': {e}")

    # Save the interim results
    with open('interim_results.pkl', 'wb') as f:
        pickle.dump(results, f)

    return results, report_data, total_questions_asked

@app.route('/generate_questions', methods=['POST'])
def generate_questions_endpoint():
    data = request.get_json()
    topics_input = data.get('topics')
    
    if not topics_input:
        return jsonify({'error': 'Topics are required'}), 400
    
    try:
        questions = {}
        for topic in topics_input:
            questions[topic] = generate_question(topic)
        
        return jsonify({'questions': questions})
    except Exception as e:
        return jsonify({'error': f"Failed to generate questions: {str(e)}"}), 500
    
@app.route('/record_responses', methods=['POST'])
def record_responses_endpoint():
    try:
        # Extract form data
        topics = request.form.getlist('topics[]')
        questions = request.form.getlist('questions[]')
        audio_files = request.files.getlist('audio_files[]')
        recording_duration = int(request.form.get('recording_duration', 60))

        if not topics or not questions or not audio_files:
            return jsonify({'error': 'Topics, questions, and audio files must be provided'}), 400

        report_data = []
        for idx, (topic, question) in enumerate(zip(topics, questions)):
            # Ensure question and topic are not empty
            if not question.strip() or not topic.strip():
                print(f'Skipping invalid input for topic "{topic}"')
                continue

            reference_answer = generate_reference_answer(question)

            # Save audio file
            audio_file = audio_files[idx]
            if audio_file.filename == '':
                print(f'No selected file for topic "{topic}"')
                continue
            
            audio_filename = secure_filename(audio_file.filename)
            audio_file_path = os.path.join(audio_file_directory, audio_filename)
            audio_file.save(audio_file_path)

            # Convert audio to text
            candidate_response = speech_to_text(audio_file_path)
            if not is_meaningful_response(candidate_response):
                print(f"No meaningful response detected for topic '{topic}'. Skipping.")
                continue

            # Evaluate response
            scores = evaluate_candidate_response(candidate_response, reference_answer)
            print(f"Candidate's Response to '{topic}': {candidate_response}")

            # Collect report data
            report_data.append({
                "Question": question,
                "Topic": topic,
                "Response": candidate_response,
                "Relevance Score": scores["Relevance Score"]
            })

        # Save the report data
        with open('final_report.pkl', 'wb') as f:
            pickle.dump(report_data, f)

        return jsonify({'report_data': report_data})
    except Exception as e:
        return jsonify({'error': f"Failed to record responses: {str(e)}"}), 500
if __name__ == '__main__':
    app.run(debug=True)
