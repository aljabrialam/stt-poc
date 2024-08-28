import os
from dotenv import load_dotenv
import json 
import boto3
import streamlit as st
import sounddevice as sd
import numpy as np
import wavio
from io import BytesIO
import uuid
import requests
from transformers import pipeline
import subprocess
from botocore.exceptions import ClientError
import streamlit.components.v1 as components

# client = boto3.client("bedrock-runtime", region_name="ap-southeast-1")

# Set the model ID, e.g., Titan Text Premier.
model_id = "anthropic.claude-v2"

# Load environment variables
load_dotenv()
subprocess.run(["pip", "install", "--upgrade", "pip"])


# # Initialize AWS clients
# s3_client = boto3.client(
#     's3',
#     aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
#     aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
#     region_name=os.getenv('AWS_REGION')
# )

# transcribe_client = boto3.client('transcribe', region_name=os.getenv('AWS_REGION'))
# bucket_name = os.getenv('AWS_S3_BUCKET_NAME')

# # Initialize SageMaker runtime client
# sagemaker_runtime = boto3.client(
#     'sagemaker-runtime',
#     aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
#     aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
#     region_name=os.getenv('AWS_REGION')
# )

# sagemaker_endpoint_name = os.getenv('SAGEMAKER_ENDPOINT_NAME')


# Initialize AWS clients
s3_client = boto3.client(
    's3',
    aws_access_key_id=st.secrets["AWS_ACCESS_KEY_ID"],
    aws_secret_access_key=st.secrets["AWS_SECRET_ACCESS_KEY"],
    region_name=st.secrets["AWS_REGION"]
)

transcribe_client = boto3.client('transcribe', region_name=st.secrets["AWS_REGION"])
bucket_name = st.secrets["AWS_S3_BUCKET_NAME"]

bedrock_client = boto3.client(
    'bedrock-runtime',
    aws_access_key_id=st.secrets["AWS_ACCESS_KEY_ID"],
    aws_secret_access_key=st.secrets["AWS_SECRET_ACCESS_KEY"],
    region_name=st.secrets["AWS_REGION"]
)


# Function to upload audio to S3
def upload_to_s3(audio_data, file_name):
    s3_client.upload_fileobj(audio_data, bucket_name, file_name)

# Function to transcribe audio
def transcribe_audio(file_name):
    job_name = f"transcribe-job-{uuid.uuid4()}"
    s3_uri = f"s3://{bucket_name}/{file_name}"

    transcribe_client.start_transcription_job(
        TranscriptionJobName=job_name,
        Media={'MediaFileUri': s3_uri},
        MediaFormat='wav',
        LanguageCode='en-US'
    )

    while True:
        result = transcribe_client.get_transcription_job(TranscriptionJobName=job_name)
        status = result['TranscriptionJob']['TranscriptionJobStatus']
        if status in ['COMPLETED', 'FAILED']:
            break

    if status == 'COMPLETED':
        transcript_file_uri = result['TranscriptionJob']['Transcript']['TranscriptFileUri']
        transcript = requests.get(transcript_file_uri).json()
        return transcript['results']['transcripts'][0]['transcript']
    else:
        return "Transcription failed."

def summarize_text(text):
    conversation = [
        {
            "role": "user",
            "content": [{"text": text}],
        }
    ]

    try:
        # Send the message to the model, using a basic inference configuration.
        response = bedrock_client.converse(
            modelId="anthropic.claude-v2",
            messages=conversation,
            inferenceConfig={"maxTokens":2048,"stopSequences":["\n\nHuman:"],"temperature":0.5,"topP":1},
            additionalModelRequestFields={"top_k":250}
        )

        # Extract and print the response text.
        response_text = response["output"]["message"]["content"][0]["text"]
        print(response_text)
        return response_text

    except (ClientError, Exception) as e:
        print(f"ERROR: Can't invoke '{model_id}'. Reason: {e}")
        return f"ERROR: Can't invoke '{model_id}'. Reason: {e}"
        # exit(1)
    

# Function to record audio using sounddevice
def record_audio(duration=30, fs=44100):
    try:
        st.write("Recording...")
        device_index = None  # Set to None to use the default device
        with sd.InputStream(samplerate=fs, channels=1, dtype='int16', device=device_index) as stream:
            frames = []
            for _ in range(int(duration * fs / 1024)):
                data, overflowed = stream.read(1024)
                frames.append(data)
        
        audio_data = np.concatenate(frames, axis=0)
        st.write("Recording finished.")
        return audio_data, fs

    except Exception as e:
        st.error(f"An error occurred while recording audio: {e}")
        return None, None

# Streamlit UI
st.title("Speech-to-Text and Summarization App")

# Record audio
if st.button("Record Audio"):
    duration = st.slider("Select duration (seconds)", 1, 30, 30)
    audio, fs = record_audio(duration)

    if audio is not None:
        # Convert the numpy array to bytes
        audio_bytes = BytesIO()
        wavio.write(audio_bytes, audio, fs, sampwidth=2)  # sampwidth=2 for 16-bit PCM
        audio_bytes.seek(0)

        st.audio(audio_bytes, format='audio/wav')

        file_name = f"audio/{uuid.uuid4()}.wav"
        upload_to_s3(audio_bytes, file_name)

        with st.spinner("Transcribing audio..."):
            transcript = transcribe_audio(file_name)

        st.subheader("Transcript")
        st.write(transcript)

        with st.spinner("Summarizing text..."):
            summary = summarize_text(transcript)

        st.subheader("Summary")
        st.write(summary)

# Upload audio file
uploaded_file = st.file_uploader("Or upload an audio file", type=["mp3", "wav", "m4a"])

if uploaded_file:
    st.audio(uploaded_file, format='audio/wav')

    # Save audio to a BytesIO object
    audio_bytes = BytesIO(uploaded_file.read())
    file_name = f"audio/{uuid.uuid4()}.wav"
    upload_to_s3(audio_bytes, file_name)

    with st.spinner("Transcribing audio..."):
        transcript = transcribe_audio(file_name)

    st.subheader("Transcript")
    st.write(transcript)

    with st.spinner("Summarizing text..."):
        summary = summarize_text(transcript)

    st.subheader("Summary")
    st.write(summary)
