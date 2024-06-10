import os
import whisper
import openai
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from difflib import SequenceMatcher
from collections import defaultdict
from flask import Flask, request, jsonify
from dotenv import load_dotenv

load_dotenv()

GPT_VERSION_4O = "gpt-4o"
GPT_VERSION_3_5_TURBO = "gpt-3.5-turbo-0125"  # also called gpt-3.5-turbo
gpt_version = GPT_VERSION_4O


# Set OpenAI API Key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
assert OPENAI_API_KEY, "Please set your OpenAI API Key in a .env file"
openai.api_key = OPENAI_API_KEY


print("start loading model")
whisper_model = whisper.load_model("base")
print("model loaded")

app = Flask(__name__)


# Function to Transcribe Audio Files
def speech_to_text(audio_file):
    result = whisper_model.transcribe(audio_file)
    return result["text"]


# Function to Generate GPT Questions
def generate_question(topic):
    prompt = f"Generate a complex interview question for an AI engineer about {topic}."
    response = openai.ChatCompletion.create(
        model=gpt_version,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ],
        max_tokens=100,
        temperature=0.6,
    )
    return response["choices"][0]["message"]["content"]


# Function to Generate Reference Answers
def generate_reference_answer(question):
    prompt = f"Provide a comprehensive answer for the following AI engineering interview question: {question}"
    response = openai.ChatCompletion.create(
        model=gpt_version,
        messages=[
            {"role": "system", "content": "You are a knowledgeable assistant."},
            {"role": "user", "content": prompt},
        ],
        max_tokens=150,
        temperature=0.6,
    )
    return response["choices"][0]["message"]["content"]


# Function to Categorize Answers
def categorize_response(candidate_response, reference_answer):
    # Keyword Extraction (Basic Example)
    def extract_keywords(text):
        stop_words = set(
            [
                "the",
                "is",
                "in",
                "and",
                "to",
                "with",
                "for",
                "on",
                "by",
                "an",
                "of",
                "a",
                "as",
            ]
        )
        keywords = [word for word in text.lower().split() if word not in stop_words]
        return set(keywords)

    important_keywords = extract_keywords(reference_answer)
    response_keywords = extract_keywords(candidate_response)
    keyword_match_ratio = len(
        set(response_keywords).intersection(important_keywords)
    ) / len(important_keywords)
    category = (
        "Accurate"
        if keyword_match_ratio > 0.7
        else "General" if keyword_match_ratio > 0.4 else "Inaccurate"
    )
    return category, keyword_match_ratio


# Fine-tuned Evaluation Factors
def evaluate_candidate_response(candidate_response, reference_answer):
    # Function to Compare and Score Candidate Answers
    def compare_answers(candidate_answer, reference_answer):
        matcher = SequenceMatcher(None, candidate_answer, reference_answer)
        similarity_score = matcher.ratio()
        return similarity_score

    relevance_score = compare_answers(candidate_response, reference_answer)
    critical_thinking_score = relevance_score * 10  # Enhanced calculation
    communication_score = min(
        len(candidate_response.split()) / len(reference_answer.split()), 1.0
    )  # Normalize to [0, 1]
    depth_of_answer_score = min(
        len(candidate_response) / len(reference_answer), 1.0
    )  # Normalize to [0, 1]
    coherence_score = (
        1 if candidate_response.count(".") >= reference_answer.count(".") else 0.5
    )

    return {
        "Relevance Score": relevance_score,
        "Critical Thinking": critical_thinking_score,
        "Communication Skills": communication_score,
        "Depth of Answer": depth_of_answer_score,
        "Coherence": coherence_score,
    }


# Generate Report and Visualization
def generate_report(report_data, results):
    df = pd.DataFrame(report_data)

    # Filter out non-numeric columns for calculating overall score
    numeric_cols = df.select_dtypes(include=[np.number])

    # Overall Interview Score
    if not numeric_cols.empty:
        overall_score = numeric_cols.mean().mean()
    else:
        overall_score = None

    # Visual Representation (e.g., Box Plot)
    numeric_cols.plot(kind="box", figsize=(10, 6))
    plt.title("Overall Interview Evaluation Report")
    plt.savefig("overall_evaluation_report.png")

    # Structure the final report
    final_report = {
        "Overall Interview Score": overall_score,
        "Question-wise Scores": numeric_cols.to_dict(orient="list"),
        "Overall Visual Representation": "overall_evaluation_report.png",
        "Detailed Results": results,
    }

    return final_report


def evaluate_audio_response_for_given_question(audio_filename, question):
    results = []
    report_data = defaultdict(list)

    candidate_response = speech_to_text(audio_filename)
    print("\n", "candidate_response: ", candidate_response, "\n")
    reference_answer = generate_reference_answer(question)
    print("\n", "reference_answer: ", reference_answer, "\n")

    # Analyze Response
    category, keyword_match_ratio = categorize_response(
        candidate_response, reference_answer
    )
    scores = evaluate_candidate_response(candidate_response, reference_answer)
    scores["Category"] = category
    scores["Keyword Match Ratio"] = keyword_match_ratio
    results.append(
        {
            "Question": question,
            "Transcribed Answer": candidate_response,
            "Reference Answer": reference_answer,
            "Scores": scores,
        }
    )

    # Add scores to report data
    for key, value in scores.items():
        report_data[key].append(value)

    # Generate the Final Report
    final_report = generate_report(report_data, results)
    return final_report


# Example Usage
default_topics = [
    "dsa",
    "oops",
]


@app.get("/")
def home():
    return "Welcome to the AI Interview Assistant!"


@app.post("/gererate_questions")
def generate_questions():
    data = request.json
    topics = data.get("topics", default_topics)
    print("topics: ", topics)
    questions = [
        {"question": generate_question(topic), "topic": topic} for topic in topics
    ]

    return jsonify(questions)


@app.post("/evaluate_candidate__audio_response_for_given_question")
def evaluate_candidate_audio_response_for_given_question():
    audio_file = request.files.get("audio_file")
    print("audio_file: ", audio_file)
    question = request.form.get("question")
    print("question: ", question)

    if not audio_file or not question:
        return (
            jsonify({"error": "Please provide both the audio file and the question."}),
            400,
        )

    audio_filename = "audio_file.wav"
    audio_file.save(audio_filename)
    # Generate the Final Report
    try:
        final_report = evaluate_audio_response_for_given_question(
            audio_filename, question
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    return jsonify(final_report)


if __name__ == "__main__":
    print("file_name: ", __file__)
    app.run(debug=True, host="0.0.0.0", port=5050)
