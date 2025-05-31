import streamlit as st
import os
from linkedin_api import Linkedin
import spacy
from spacy.matcher import Matcher
import openai
from dotenv import load_dotenv
load_dotenv()

nlp = spacy.load("en_core_web_sm")

def authenticate_linkedin():
    """Authenticate LinkedIn using API keys set as environment variables."""
    client_id = os.getenv('LINKEDIN_CLIENT_ID')
    client_secret = os.getenv('LINKEDIN_CLIENT_SECRET')
    if not client_id or not client_secret:
        st.error("LinkedIn API keys are not set. Please configure them as environment variables.")
        return None
    try:
        return Linkedin(client_id, client_secret)
    except Exception as e:
        st.error(f"Failed to authenticate with LinkedIn: {e}")
        return None

def analyze_text(text):
    """Analyze input text using SpaCy patterns."""
    doc = nlp(text)
    matcher = Matcher(nlp.vocab)
    matcher.add('Improvement', [[{"LOWER": "improve"}, {"POS": "ADP"}, {"POS": "NOUN"}]])
    matches = matcher(doc)
    suggestions = [doc[start:end].text for _, start, end in matches]
    return suggestions

def generate_summary_and_recommendations(text):
    """Use OpenAI to generate summary and recommendations."""
    openai.api_key = os.getenv("OPENAI_API_KEY")
    try:
        summary_resp = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": f"Summarize this LinkedIn profile:\n{text}"}]
        )
        summary = summary_resp.choices[0].message.content.strip()

        recommend_resp = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": f"Provide suggestions to improve this LinkedIn profile:\n{text}"}]
        )
        recommendations = recommend_resp.choices[0].message.content.strip()
        return summary, recommendations
    except Exception as e:
        return "Summary generation failed.", f"Error from OpenAI: {e}"

def show_linkedin_analyzer():
    st.title("🔗 LinkedIn Profile Analyzer")
    st.write("Enter your public LinkedIn profile URL to get AI-powered insights.")

    profile_url = st.text_input("Enter LinkedIn Profile URL")

    if profile_url:
        api = authenticate_linkedin()
        if not api:
            return

        with st.spinner("Fetching profile..."):
            try:
                profile = api.get_profile(profile_url=profile_url)
                summary_text = profile.get("summary", "")

                if not summary_text:
                    st.warning("No summary text found in this profile.")
                    return

                suggestions = analyze_text(summary_text)
                summary, recommendations = generate_summary_and_recommendations(summary_text)

                st.subheader("📝 AI Summary")
                st.write(summary)

                st.subheader("💡 AI Recommendations")
                st.write(recommendations)

                if suggestions:
                    st.subheader("🔍 Pattern-based Suggestions")
                    for s in suggestions:
                        st.markdown(f"- {s}")
            except Exception as e:
                st.error(f"Error fetching profile: {e}")
