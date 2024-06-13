""""
To run the FastAPI application, execute the following command in the terminal:
uvicorn fastapi_app:app --host 0.0.0.0 --port 5050 --reload
"""

import os
from pydantic import BaseModel
import whisper
import openai
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from difflib import SequenceMatcher
from collections import defaultdict
from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor
import asyncio
from typing import List  # rushabh
import requests
import json

load_dotenv()

# set the GPT version to use
GPT_VERSION_4O = "gpt-4o"
GPT_VERSION_3_5_TURBO = "gpt-3.5-turbo-0125"
gpt_version = GPT_VERSION_4O

# Set OpenAI API Key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# keep this line to ensure the API key is set in the .env file
assert OPENAI_API_KEY, "Please set your OpenAI API Key in a .env file"

openai.api_key = OPENAI_API_KEY

print("start loading model")
whisper_model = whisper.load_model("base")
print("model loaded")

app = FastAPI()

# Allow CORS (Optional)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust as necessary
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create a thread pool executor for CPU-bound tasks
executor = ThreadPoolExecutor(max_workers=4)


# Asynchronous Function to Transcribe Audio Files
async def speech_to_text(audio_file):
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(executor, whisper_model.transcribe, audio_file)
    return result["text"]


# Asynchronous Function to Generate GPT Questions
async def generate_question(topic):
    prompt = f"Generate a complex interview question for an AI engineer about {topic}."
    response = await openai.ChatCompletion.acreate(
        model=gpt_version,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ],
        max_tokens=100,
        temperature=0.6,
    )
    return response["choices"][0]["message"]["content"]


# Asynchronous Function to Generate Reference Answers
async def generate_reference_answer(question):
    prompt = f"Provide a comprehensive answer for the following AI engineering interview question: {question}"
    response = await openai.ChatCompletion.acreate(
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
    def compare_answers(candidate_answer, reference_answer):
        matcher = SequenceMatcher(None, candidate_answer, reference_answer)
        similarity_score = matcher.ratio()
        return similarity_score

    relevance_score = compare_answers(candidate_response, reference_answer)
    critical_thinking_score = relevance_score * 10
    communication_score = min(
        len(candidate_response.split()) / len(reference_answer.split()), 1.0
    )
    depth_of_answer_score = min(len(candidate_response) / len(reference_answer), 1.0)
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
    numeric_cols = df.select_dtypes(include=[np.number])
    overall_score = numeric_cols.mean().mean() if not numeric_cols.empty else None

    numeric_cols.plot(kind="box", figsize=(10, 6))
    plt.title("Overall Interview Evaluation Report")
    plt.savefig("overall_evaluation_report.png")

    final_report = {
        "Overall Interview Score": overall_score,
        "Question-wise Scores": numeric_cols.to_dict(orient="list"),
        "Overall Visual Representation": "overall_evaluation_report.png",
        "Detailed Results": results,
    }

    return final_report


async def evaluate_audio_response_for_given_question(audio_filename, question):
    results = []
    report_data = defaultdict(list)

    candidate_response, reference_answer = await asyncio.gather(
        speech_to_text(audio_filename), generate_reference_answer(question)
    )

    print("\n", "candidate_response: ", candidate_response, "\n")
    print("\n", "reference_answer: ", reference_answer, "\n")

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

    for key, value in scores.items():
        report_data[key].append(value)

    final_report = generate_report(report_data, results)
    return final_report


default_topics = ["dsa", "oops"]


@app.get("/")
async def home():
    return "Welcome to the AI Interview Assistant!"


# Define request model
class TopicsRequest(BaseModel):
    topics: List[str] = ["dsa", "oops"]


@app.post("/generate_questions")
async def generate_questions(request: TopicsRequest):
    topics = request.topics
    print("topics: ", topics)

    question_promises = [generate_question(topic) for topic in topics]
    answers = await asyncio.gather(*question_promises)

    questions = [
        {"question": question, "topic": topic}
        for topic, question in zip(topics, answers)
    ]

    return JSONResponse(content=questions)


# Define request model
class ReferenceAnswersRequest(BaseModel):
    questions: List[str] = ["What is a linked list?", "What is polymorphism?"]


@app.post("/generate_reference_answers")
async def generate_reference_answers(request: ReferenceAnswersRequest):
    questions = request.questions
    print("questions: ", questions)

    answer_promises = [generate_reference_answer(question) for question in questions]
    answers = await asyncio.gather(*answer_promises)

    reference_answers = [
        {"question": question, "answer": answer}
        for question, answer in zip(questions, answers)
    ]

    return JSONResponse(content=reference_answers)


# @app.post("/evaluate_candidate_audio_response_for_given_question")
# async def evaluate_candidate_audio_response_for_given_question(
#     audio_file: UploadFile = File(...), question: str = Form(...)
# ):
#     print("audio_file: ", audio_file)
#     print("question: ", question)

#     if not audio_file or not question:
#         raise HTTPException(
#             status_code=400,
#             detail="Please provide both the audio file and the question.",
#         )

#     audio_filename = "audio_file.wav"
#     with open(audio_filename, "wb") as f:
#         f.write(await audio_file.read())

#     try:
#         final_report = await evaluate_audio_response_for_given_question(
#             audio_filename, question
#         )
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))

#     return JSONResponse(content=final_report)


@app.post("/evaluate_candidate_audio_response_for_given_question")
async def evaluate_candidate_audio_response_for_given_question(
    audio_file: UploadFile = File(...), question: str = Form(...)
):
    print("audio_file: ", audio_file)
    print("question: ", question)

    # Check if audio file and question are provided
    if not audio_file or not question:
        raise HTTPException(
            status_code=400,
            detail="Please provide both the audio file and the question.",
        )

    # Save audio file locally
    audio_filename = "audio_file.wav"
    with open(audio_filename, "wb") as f:
        f.write(await audio_file.read())

    try:
        # Evaluate audio response for given question
        final_report = await evaluate_audio_response_for_given_question(
            audio_filename, question
        )

        # Serialize final_report to JSON
        final_report_json = json.dumps(final_report)

        # Prepare data to send to Node.js server
        data_to_send = {
            "question": question,
            "audio_file_name": audio_filename,
            "evaluation_report": final_report_json,
        }

        # URL of Node.js server endpoint
        node_server_url = "http://localhost:5002/data/receive-data"

        # Send data to Node.js server
        response = requests.post(node_server_url, json=data_to_send)

        # Print response from Node.js server
        print("Response from Node.js server:", response.text)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return JSONResponse(content=final_report)


print("done loading application")
