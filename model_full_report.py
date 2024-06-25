import pandas as pd
import matplotlib.pyplot as plt
from textblob import TextBlob
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
import language_tool_python
import openai
import os
import pickle

# Load your OpenAI API Key from an environment variable for security
openai.api_key = os.getenv('OPENAI_API_KEY')

# Ensure NLTK resources are downloaded
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

# Load interim results
with open('interim_results.pkl', 'rb') as f:
    results = pickle.load(f)

# Load total questions asked
with open('total_questions_asked.pkl', 'rb') as f:
    total_questions_asked = pickle.load(f)

# Initialize language tool for grammar checking
tool = language_tool_python.LanguageTool('en-US')

# Evaluate communication skills
def evaluate_communication_skills(response):
    if response is None:
        return {"Readability Score": 0, "Grammar Score": 0}

    # Readability score (using TextBlob's sentence parsing)
    sentences = TextBlob(response).sentences
    readability_score = sum(len(sentence.words) for sentence in sentences) / len(sentences) if sentences else 0

    # Grammar score (using language_tool_python)
    matches = tool.check(response)
    grammar_score = max(0, 100 - len(matches))  # Deduct points for each grammar issue

    return {
        "Readability Score": readability_score,
        "Grammar Score": grammar_score
    }

# Extract keywords with TF-IDF
def extract_keywords(texts):
    vectorizer = TfidfVectorizer(stop_words='english', max_features=10)
    tfidf_matrix = vectorizer.fit_transform(texts)
    feature_names = vectorizer.get_feature_names_out()
    return feature_names

# Calculate average sentence complexity and sentiment
def analyze_sentences(texts):
    complexity_scores = []
    sentiment_scores = []
    for text in texts:
        if text:
            sentences = TextBlob(text).sentences
            complexity = sum(len(sentence.words) for sentence in sentences) / len(sentences) if sentences else 0
            sentiment = TextBlob(text).sentiment.polarity
            complexity_scores.append(complexity)
            sentiment_scores.append(sentiment)
    avg_complexity = sum(complexity_scores) / len(complexity_scores) if complexity_scores else 0
    avg_sentiment = sum(sentiment_scores) / len(sentiment_scores) if sentiment_scores else 0
    return avg_complexity, avg_sentiment

# Load candidate responses
candidate_responses = [result["Response"] for result in results]
followup_responses = [result["Follow-up Response"] for result in results if result["Follow-up Response"]]

# Calculate communication scores
communication_scores = [evaluate_communication_skills(response) for response in candidate_responses]
if followup_responses:
    followup_communication_scores = [evaluate_communication_skills(response) for response in followup_responses]
else:
    followup_communication_scores = []

# Extract keywords
all_responses = candidate_responses + followup_responses
keywords = extract_keywords(all_responses)

# Analyze sentence complexity and sentiment
avg_complexity, avg_sentiment = analyze_sentences(all_responses)

# Generate final report
report = {
    "Total Questions Asked": total_questions_asked,
    "Keywords": keywords.tolist(),
    "Average Sentence Complexity": avg_complexity,
    "Average Sentiment": avg_sentiment,
    "Candidate Responses": candidate_responses,
    "Follow-up Responses": followup_responses,
    "Communication Scores": communication_scores,
    "Follow-up Communication Scores": followup_communication_scores
}

# Create a DataFrame for easier manipulation and visualization
df_report = pd.DataFrame(report)

# Save the report as a CSV file
df_report.to_csv('interview_report.csv', index=False)

# Generate plots for the report
plt.figure(figsize=(10, 6))

# Plot Average Sentence Complexity
plt.subplot(2, 1, 1)
plt.plot([i for i in range(len(candidate_responses))], [score["Readability Score"] for score in communication_scores], label='Readability Score')
if followup_responses:
    plt.plot([i for i in range(len(followup_responses))], [score["Readability Score"] for score in followup_communication_scores], label='Follow-up Readability Score')
plt.xlabel('Question Index')
plt.ylabel('Readability Score')
plt.title('Readability Scores')
plt.legend()

# Plot Average Sentiment
plt.subplot(2, 1, 2)
plt.plot([i for i in range(len(candidate_responses))], [TextBlob(response).sentiment.polarity for response in candidate_responses], label='Sentiment Score')
if followup_responses:
    plt.plot([i for i in range(len(followup_responses))], [TextBlob(response).sentiment.polarity for response in followup_responses], label='Follow-up Sentiment Score')
plt.xlabel('Question Index')
plt.ylabel('Sentiment Score')
plt.title('Sentiment Scores')
plt.legend()

plt.tight_layout()
plt.savefig('interview_report_plots.png')
plt.show()
