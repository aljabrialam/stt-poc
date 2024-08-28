import streamlit as st
import streamlit.components.v1 as components
import uuid
from io import BytesIO
import base64
import boto3
import json
import requests
from dotenv import load_dotenv
import threading
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import boto3
from botocore.exceptions import ClientError
from io import BytesIO
import uuid

# Load environment variables
load_dotenv()

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
    region_name="us-east-1"
)

model_id = "anthropic.claude-3-sonnet-20240229-v1:0"

# FastAPI app
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://stt-poc.streamlit.app/", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class AudioRequest(BaseModel):
    audio: str

def upload_to_s3(audio_data, file_name):
    s3_client.upload_fileobj(audio_data, bucket_name, file_name)

def transcribe_audio(file_name):
    job_name = f"transcribe-job-{uuid.uuid4()}"
    s3_uri = f"s3://{bucket_name}/{file_name}"

    try:
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

    except Exception as e:
        return str(e)

def summarize_text(text):
    init_sum = "Below provided are some meeting notes. Read through the notes, understand key take aways and summarize the meeting notes: "
    conversation = [
        {
            "role": "user",
            "content": [{"text": init_sum + text}],
        }
    ]

    try:
        # Send the message to the model, using a basic inference configuration.
        response = bedrock_client.converse(
            modelId="amazon.titan-text-express-v1",
            messages=conversation,
            inferenceConfig={"maxTokens":4096,"stopSequences":["User:"],"temperature":0,"topP":1},
            additionalModelRequestFields={}
        )

        # Extract and print the response text.
        response_text = response["output"]["message"]["content"][0]["text"]
        print(response_text)
        
        return response_text
    except (ClientError, Exception) as e:
        print(f"ERROR: Can't invoke '{model_id}'. Reason: {e}")



@app.post("/transcribe")
async def transcribe(request: AudioRequest):
    audio_base64 = request.audio
    audio_bytes = BytesIO(base64.b64decode(audio_base64))
    file_name = f"audio/{uuid.uuid4()}.wav"
    upload_to_s3(audio_bytes, file_name)
    
    transcript = transcribe_audio(file_name)
    summary = summarize_text(transcript)
    
    return JSONResponse(content={
        "transcript": transcript,
        "summary": summary
    })

def run_fastapi():
    uvicorn.run(app, host="0.0.0.0", port=8000)

# Start FastAPI server in a separate thread
threading.Thread(target=run_fastapi, daemon=True).start()

# Streamlit UI
st.markdown(
    "<h1 style='text-align: center;'>Speech-to-Text and Summarization App</h1>",
    unsafe_allow_html=True
)

# JavaScript-based audio recorder
st.markdown("## Record Audio Using Browser")

# Embed JavaScript directly in the HTML component
html_code = """
<style>
    button {
        font-size: 16px; /* Increase font size */
        padding: 10px 20px; /* Add padding for larger button */
        margin: 5px; /* Add margin for spacing */
        cursor: pointer; /* Change cursor on hover */
        border: none; /* Remove default border */
        border-radius: 5px; /* Rounded corners */
        background-color: #4CAF50; /* Green background for Start Recording */
        color: white; /* White text color */
    }

    #stopButton {
        background-color: red; /* Red background for Stop Recording */
    }

    button:disabled {
        background-color: #ccc; /* Gray background for disabled button */
        cursor: not-allowed; /* Change cursor for disabled button */
    }
</style>

<button id="recordButton">Start Recording</button>
<button id="stopButton" disabled>Stop Recording</button>
<div></div>
<audio id="audioPlayback" controls></audio>
<div></div>
<div id="transcript"></div>

<script>
    let mediaRecorder;
    let audioChunks = [];

    document.getElementById('recordButton').addEventListener('click', () => {
        navigator.mediaDevices.getUserMedia({ audio: true })
            .then(stream => {
                console.log("Microphone access granted");
                mediaRecorder = new MediaRecorder(stream);
                mediaRecorder.start();
                console.log("Recording started");
                audioChunks = [];

                mediaRecorder.addEventListener('dataavailable', event => {
                    audioChunks.push(event.data);
                    console.log("Audio chunk received");
                });

                mediaRecorder.addEventListener('stop', () => {
                    console.log("Recording stopped");
                    const audioBlob = new Blob(audioChunks, { type: 'audio/wav' });
                    const audioUrl = URL.createObjectURL(audioBlob);
                    document.getElementById('audioPlayback').src = audioUrl;
                    console.log("Audio URL created");

                    const reader = new FileReader();
                    reader.readAsDataURL(audioBlob);
                    reader.onloadend = function() {
                        const base64data = reader.result.split(',')[1];
                        console.log("Audio Base64 data prepared");
                        fetch('https://stt-poc.streamlit.app/transcribe', {
                            method: 'POST',
                            headers: {
                                'Content-Type': 'application/json'
                            },
                            body: JSON.stringify({ audio: base64data })
                        })
                        .then(response => response.json())
                        .then(data => {
                            console.log("Received transcript and summary", data);
                            const transcriptElement = document.getElementById('transcript');
                            transcriptElement.innerHTML = "<h3>Transcript:</h3><p>" + data.transcript + "</p>";

                            if (data.transcript) {
                                const summaryElement = document.createElement('div');
                                summaryElement.innerHTML = "<h3>Summary:</h3><p>" + data.summary + "</p>";
                                document.body.appendChild(summaryElement);
                            }
                        })
                        .catch(error => {
                            console.error('Error:', error);
                            const transcriptElement = document.getElementById('transcript');
                            # transcriptElement.innerHTML = "Error sending audio.";
                        });
                    };
                });

                document.getElementById('stopButton').disabled = false;
                document.getElementById('recordButton').disabled = true;
            })
            .catch(error => {
                console.error('Error accessing media devices:', error);
                const transcriptElement = document.getElementById('transcript');
                transcriptElement.innerHTML = "Error accessing media devices.";
            });
    });

    document.getElementById('stopButton').addEventListener('click', () => {
        if (mediaRecorder) {
            mediaRecorder.stop();
            document.getElementById('stopButton').disabled = true;
            document.getElementById('recordButton').disabled = false;
        }
    });
</script>
"""

st.components.v1.html(html_code, height=150)

# Update Streamlit UI with results from JavaScript
st.components.v1.html("""
<script>
    window.addEventListener('message', function(event) {
        if (event.data.type === 'audioResult') {
            // Handle the audio result
            const transcript = event.data.transcript;
            const summary = event.data.summary;

            // Update the Streamlit interface using Streamlit's capabilities
            window.parent.postMessage({ type: 'updateUI', transcript: transcript, summary: summary }, '*');
        }
    });
</script>
""", height=0)

# Handle incoming UI updates
if st.session_state.get('transcript'):
    st.subheader("Transcript")
    st.write(st.session_state.transcript)

if st.session_state.get('summary'):
    st.subheader("Summary")
    st.write(st.session_state.summary)
    
    
st.markdown("## And Upload a Recorded Audio File for Transcription and Summarization")
# File uploader
uploaded_file = st.file_uploader("", type=["wav"])

if uploaded_file:
    st.audio(uploaded_file, format='audio/wav')

    # Save audio to a BytesIO object
    audio_bytes = BytesIO(uploaded_file.read())
    file_name = f"audio/{uuid.uuid4()}.wav"
    upload_to_s3(audio_bytes, file_name)

    # Transcribe the audio when it's uploaded
    with st.spinner("Transcribing audio..."):
        transcript = transcribe_audio(file_name)

    if transcript:  # Ensure the transcript is valid
        st.subheader("Transcript")
        
        # Display transcript in a textarea for editing/viewing
        transcript_area = st.text_area("Transcript", transcript, height=300)
        
        # Button to trigger the summary process
        if st.button("Summarize Transcript"):
            with st.spinner("Summarizing text..."):
                summary = summarize_text(transcript_area)

            st.subheader("Summary")
            st.write(summary)

            # Copy text button using Streamlit's built-in functionality
            if summary:
                st.download_button(
                    label="Copy Summary",
                    data=summary,
                    file_name="summary.txt",
                    mime="text/plain"
                )
    else:
        st.error("No transcript available.")
