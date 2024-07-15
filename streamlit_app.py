import streamlit as st
import requests
import json
import datetime
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import smtplib
import os
from openai import OpenAI

# Fetch secrets
GMAIL_PASSWORD = st.secrets["GMAIL_PASSWORD"]
GMAIL_USER = st.secrets["GMAIL_USER"]
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]

# Ensure the API key is loaded
if not OPENAI_API_KEY:
    st.error("OpenAI API key is not set in the environment variables.")
else:
    client = OpenAI(api_key=OPENAI_API_KEY)

def fetch_transcripts(api_key):
    url = 'https://api.fireflies.ai/graphql'
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {api_key}'
    }
    query = """
    query Transcripts($limit: Int) {
        transcripts(limit: $limit) {
            id
            title
            date
            sentences {
                text
                speaker_name
            }
        }
    }
    """
    variables = {"limit": 10}
    response = requests.post(url, json={'query': query, 'variables': variables}, headers=headers)
    if response.status_code != 200:
        raise Exception(f"API request failed with status code {response.status_code}")
    return response.json()['data']['transcripts']

def get_transcript_content(transcript):
    return '\n'.join([f"{sentence['speaker_name']}: {sentence['text']}" for sentence in transcript['sentences']])

def gpt4o_json_prompt(transcript_content, prompt_type):
    prompts = {
        'financial': "Analyze this meeting transcript and provide a JSON summary focusing on financial discussions, key figures, and monetary decisions. Include any mentioned budgets, costs, revenues, or financial projections.",
        'action_items': "Analyze this meeting transcript and provide a JSON summary of all action items, tasks, and deadlines discussed. Include who is responsible for each item and any mentioned due dates.",
        'risk_assessment': "Analyze this meeting transcript and provide a JSON summary of any discussed risks, potential issues, or areas of concern. Include any mitigation strategies or risk assessments mentioned.",
        'tax_info': "Analyze this meeting transcript and provide a JSON summary of all tax-related information discussed. Include any mentions of tax planning, changes in tax laws, or specific tax concerns of the client.",
        'client_concerns': "Analyze this meeting transcript and provide a JSON summary of all client questions, concerns, or areas where the client expressed confusion or needed clarification.",
        'compliance': "Analyze this meeting transcript and provide a JSON summary of all compliance and regulatory matters discussed. Include any mentions of legal requirements, industry standards, or regulatory changes."
    }

    try:
        with st.spinner('Analyzing transcript...'):
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are an AI assistant specialized in analyzing meeting transcripts for accounting professionals. You must only reply with JSON. [no prose]"},
                    {"role": "user", "content": f"""{prompts[prompt_type]}

                    Provide the summary in the following JSON structure:
                    {{
                        "summary": "A concise summary of the requested information",
                        "key_points": ["point1", "point2", "point3"],
                        "details": [
                            {{
                                "topic": "Specific topic or item",
                                "description": "Detailed description",
                                "relevance": "Why this is important for the accountant"
                            }},
                            ...
                        ],
                        "follow_up_suggestions": ["suggestion1", "suggestion2"]
                    }}

                    Transcript:
                    {transcript_content}

                    [Output only JSON]"""}
                ],
                temperature=0.7,
                max_tokens=3000,
                response_format={"type": "json_object"},
                logit_bias={123: 100}  # Increase likelihood of `{` to start the response
            )
        
        # Check if the response content is valid JSON
        try:
            return json.loads(response.choices[0].message.content)
        except json.JSONDecodeError as json_error:
            st.error(f"Error decoding JSON: {json_error}")
            st.error(f"Raw response: {response.choices[0].message.content}")
            return None
    except Exception as e:
        st.error(f"An error occurred while analyzing the transcript: {e}")
        return None

def generate_follow_up_email(transcript_content):
    try:
        with st.spinner('Generating follow-up email...'):
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are an AI assistant helping an accountant draft a follow-up email after a client meeting. The email should sound authentic, professional, and as if it's coming directly from the accountant who organized the meeting. You must only reply with JSON containing the email content."},
                    {"role": "user", "content": f"""Based on the following meeting transcript, create a follow-up email to the client. The email should:

                    1. Briefly summarize the key points discussed in the meeting
                    2. Confirm any responsibilities or action items for the client
                    3. Mention any deadlines discussed or set reasonable deadlines if none were specified
                    4. Sound authentic and professional, as if written by the accountant who organized the meeting
                    5. End with a polite closing and offer for further assistance

                    Format the email with appropriate HTML tags, including <p> for paragraphs, <br> for line breaks, and any other relevant HTML formatting.

                    Provide the email content in the following JSON structure:
                    {{
                        "subject": "Meeting Follow-up: [Brief Description]",
                        "body": "HTML formatted email content"
                    }}

                    Transcript:
                    {transcript_content}

                    [Output only JSON]"""}
                ],
                temperature=0.7,
                max_tokens=2000,
                response_format={"type": "json_object"},
                logit_bias={123: 100}  # Increase likelihood of `{` to start the response
            )
        
        # Check if the response content is valid JSON
        try:
            return json.loads(response.choices[0].message.content)
        except json.JSONDecodeError as json_error:
            st.error(f"Error decoding JSON for follow-up email: {json_error}")
            st.error(f"Raw response: {response.choices[0].message.content}")
            return None
    except Exception as e:
        st.error(f"An error occurred while generating the follow-up email: {e}")
        return None

def format_analysis_to_html(analysis_type, gpt4o_response, follow_up_email):
    html_content = f"""
    <html>
    <head>
        <style>
            body {{
                font-family: Arial, sans-serif;
                line-height: 1.6;
                color: #333;
                max-width: 800px;
                margin: 0 auto;
                padding: 20px;
            }}
            .header {{
                background-color: #007bff;
                color: white;
                padding: 20px;
                text-align: center;
            }}
            h1, h2 {{
                margin: 0;
            }}
            h2 {{
                color: #007bff;
                border-bottom: 2px solid #007bff;
                padding-bottom: 10px;
            }}
            .section {{
                margin-bottom: 30px;
            }}
            ul {{
                padding-left: 20px;
            }}
            .detail {{
                background-color: #f8f9fa;
                border-left: 4px solid #007bff;
                padding: 15px;
                margin-bottom: 20px;
            }}
            .footer {{
                text-align: center;
                margin-top: 40px;
                font-size: 0.9em;
                color: #666;
            }}
            .follow-up {{
                margin-top: 40px;
                border-top: 2px solid #007bff;
                padding-top: 20px;
            }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>{analysis_type.replace('_', ' ').title()} Analysis</h1>
        </div>

        <div class="section">
            <h2>Summary</h2>
            <p>{gpt4o_response['summary']}</p>
        </div>

        <div class="section">
            <h2>Key Points</h2>
            <ul>
    """

    for point in gpt4o_response['key_points']:
        html_content += f"<li>{point}</li>"

    html_content += """
            </ul>
        </div>

        <div class="section">
            <h2>Details</h2>
    """

    for detail in gpt4o_response['details']:
        html_content += f"""
            <div class="detail">
                <h3>{detail['topic']}</h3>
                <p><strong>Description:</strong> {detail['description']}</p>
                <p><strong>Relevance:</strong> {detail['relevance']}</p>
            </div>
        """

    html_content += """
        </div>

        <div class="section">
            <h2>Follow-up Suggestions</h2>
            <ul>
    """

    for suggestion in gpt4o_response['follow_up_suggestions']:
        html_content += f"<li>{suggestion}</li>"

    html_content += f"""
            </ul>
        </div>

        <div class="follow-up">
            <h2>Follow-up Email</h2>
            {follow_up_email}
        </div>

        <div class="footer">
            <p>This analysis was generated automatically. Please review for accuracy.</p>
        </div>
    </body>
    </html>
    """
    return html_content

def send_email(recipient_email, subject, html_content):
    try:
        msg = MIMEMultipart('alternative')
        msg['Subject'] = subject
        msg['From'] = GMAIL_USER
        msg['To'] = recipient_email

        html_part = MIMEText(html_content, 'html')
        msg.attach(html_part)

        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp_server:
            smtp_server.login(GMAIL_USER, GMAIL_PASSWORD)
            smtp_server.sendmail(GMAIL_USER, recipient_email, msg.as_string())
        st.success("Email sent successfully!")
    except Exception as e:
        st.error(f"Error sending email: {e}")

# Streamlit UI
st.title('Fireflies Meeting Prototype')

# User inputs
fireflies_api_key = st.text_input('Fireflies API Key', type='password')
email = st.text_input('Email Address')

auto_email = st.checkbox('Auto-email analysis')

# Fetch transcripts button
refresh_button = st.button('Refresh Transcripts')

# Use session state to store transcripts and selected transcript ID
if 'transcripts' not in st.session_state:
    st.session_state.transcripts = []
if 'selected_transcript_id' not in st.session_state:
    st.session_state.selected_transcript_id = None

if fireflies_api_key and refresh_button:
    with st.spinner('Refreshing transcripts...'):
        try:
            st.session_state.transcripts = fetch_transcripts(fireflies_api_key)
            st.success("Transcripts refreshed successfully.")
        except Exception as e:
            st.error(f"Error refreshing transcripts: {e}")

if st.session_state.transcripts:
    transcript_options = [t['id'] for t in st.session_state.transcripts]
    transcript_titles = [t['title'] for t in st.session_state.transcripts]

    transcript_id = st.selectbox(
        'Select Transcript',
        options=transcript_options,
        format_func=lambda x: transcript_titles[transcript_options.index(x)] if x in transcript_options else ""
    )

    if transcript_id:
        st.session_state.selected_transcript_id = transcript_id

if st.session_state.selected_transcript_id:
    selected_transcript = next(
        t for t in st.session_state.transcripts if t['id'] == st.session_state.selected_transcript_id
    )
    transcript_content = get_transcript_content(selected_transcript)
    st.text_area('Transcript', transcript_content, height=200, disabled=True)

analyze_buttons = {
    'financial': st.button('Financial Analysis'),
    'action_items': st.button('Action Items'),
    'risk_assessment': st.button('Risk Assessment'),
    'tax_info': st.button('Tax Information'),
    'client_concerns': st.button('Client Concerns'),
    'compliance': st.button('Compliance Matters')
}

def handle_analysis(prompt_type):
    if st.session_state.selected_transcript_id:
        selected_transcript = next(
            t for t in st.session_state.transcripts if t['id'] == st.session_state.selected_transcript_id
        )
        transcript_content = get_transcript_content(selected_transcript)
        st.write(f"Analyzing with GPT-4o: {prompt_type.replace('_', ' ').title()}...")
        gpt4o_response = gpt4o_json_prompt(transcript_content, prompt_type)
        follow_up_email = generate_follow_up_email(transcript_content)
        if gpt4o_response and follow_up_email:
            st.write(f"## {prompt_type.replace('_', ' ').title()} Analysis")
            st.write(f"### Summary\n{gpt4o_response['summary']}")
            st.write("### Key Points")
            for point in gpt4o_response['key_points']:
                st.write(f"- {point}")
            st.write("### Details")
            for detail in gpt4o_response['details']:
                st.write(f"#### {detail['topic']}")
                st.write(f"Description: {detail['description']}")
                st.write(f"Relevance: {detail['relevance']}")
                st.write("")
            st.write("### Follow-up Suggestions")
            for suggestion in gpt4o_response['follow_up_suggestions']:
                st.write(f"- {suggestion}")
            st.write("### Follow-up Email")
            st.markdown(follow_up_email['body'], unsafe_allow_html=True)

            if auto_email and email:
                with st.spinner('Sending email...'):
                    html_content = format_analysis_to_html(prompt_type, gpt4o_response, follow_up_email['body'])

                    # Create a set of unique speakers
                    speakers = set(sentence['speaker_name'] for sentence in selected_transcript['sentences'])

                    # Format the date
                    date = datetime.datetime.fromtimestamp(selected_transcript['date'] / 1000).strftime('%Y-%m-%d')

                    # Create the email subject
                    subject = f"Meeting Analysis: {prompt_type.replace('_', ' ').title()} - {selected_transcript['title']} - {date} - Speakers: {', '.join(speakers)}"

                    send_email(email, subject, html_content)
        else:
            st.error("Failed to generate analysis or follow-up email. Please check the error messages above and try again.")

# Trigger analysis based on button clicks
if analyze_buttons['financial']:
    handle_analysis('financial')
if analyze_buttons['action_items']:
    handle_analysis('action_items')
if analyze_buttons['risk_assessment']:
    handle_analysis('risk_assessment')
if analyze_buttons['tax_info']:
    handle_analysis('tax_info')
if analyze_buttons['client_concerns']:
    handle_analysis('client_concerns')
if analyze_buttons['compliance']:
    handle_analysis('compliance')

# Add a footer
st.markdown("---")
st.markdown("Powered by Fireflies.ai and OpenAI GPT-4o")