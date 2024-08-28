import streamlit as st
import streamlit.components.v1 as components
import uuid
from io import BytesIO
import base64
import boto3
import requests
from botocore.exceptions import ClientError

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

# Function to upload audio file to S3
def upload_to_s3(audio_data, file_name):
    s3_client.upload_fileobj(audio_data, bucket_name, file_name)

# Function to transcribe audio
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

# Function to summarize transcript text
def summarize_text(text):
    init_sum = "Understand context, key takeaways and summarize the sentences: "
    conversation = [
        {
            "role": "user",
            "content": [{"text": init_sum + text}],
        }
    ]

    try:
        response = bedrock_client.converse(
            modelId="anthropic.claude-3-sonnet-20240229-v1:0",
            messages=conversation,
            inferenceConfig={"maxTokens":4096,"temperature":0},
            additionalModelRequestFields={"top_k":250}
        )
 
        response_text = response["output"]["message"]["content"][0]["text"]
        return response_text
    except (ClientError, Exception) as e:
        return str(e)

st.markdown("""
    <style>
        h1 {
            text-align: center;
            font-family: 'Helvetica Neue', sans-serif;
            font-size: 2.5em;
            color: #333;
            letter-spacing: 2px;
            margin-bottom: 20px;
            text-transform: uppercase;
            text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.1);
        }

        h1::after {
            content: '';
            display: block;
            width: 80px;
            margin: 10px auto;
            border-bottom: 3px solid #4CAF50;
        }
    </style>
    <h1>AWS: Speech-to-Text and Summarization App</h1>
""", unsafe_allow_html=True)


# Embed JavaScript recorder
st.markdown("## Record Audio Using Browser")

html_code = """
<style>
    .container {
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 10px;
    }

    button {
        font-size: 16px; 
        padding: 10px 20px;
        margin: 5px;
        cursor: pointer;
        border: none;
        border-radius: 5px;
        background-color: #4CAF50;
        color: white;
    }

    #stopButton {
        background-color: red;
    }

    button:disabled {
        background-color: #ccc;
        cursor: not-allowed;
    }
    
    audio {
        margin-left: 10px;
    }
</style>

<div class="container">
    <button id="recordButton">Start Recording</button>
    <button id="stopButton" disabled>Stop Recording</button>
    <audio id="audioPlayback" controls></audio>
</div>

<script>
    let mediaRecorder;
    let audioChunks = [];

    document.getElementById('recordButton').addEventListener('click', () => {
        navigator.mediaDevices.getUserMedia({ audio: true })
            .then(stream => {
                mediaRecorder = new MediaRecorder(stream);
                mediaRecorder.start();
                audioChunks = [];

                mediaRecorder.addEventListener('dataavailable', event => {
                    audioChunks.push(event.data);
                });

                mediaRecorder.addEventListener('stop', () => {
                    const audioBlob = new Blob(audioChunks, { type: 'audio/wav' });
                    const audioUrl = URL.createObjectURL(audioBlob);
                    document.getElementById('audioPlayback').src = audioUrl;

                    // Convert audioBlob to Base64
                    const reader = new FileReader();
                    reader.readAsDataURL(audioBlob);
                    reader.onloadend = function() {
                        const base64data = reader.result.split(',')[1];
                        // Send the Base64 data to Streamlit
                        window.parent.postMessage({ type: 'audioRecorded', base64Audio: base64data }, '*');
                    };
                });

                document.getElementById('stopButton').disabled = false;
                document.getElementById('recordButton').disabled = true;
            })
            .catch(error => {
                console.error('Error accessing media devices:', error);
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

components.html(html_code, height=150)

# JavaScript that listens for the Base64 audio data sent to Streamlit
audio_base64 = st.experimental_get_query_params().get("audioRecorded", None)

if audio_base64:
    audio_bytes = BytesIO(base64.b64decode(audio_base64[0]))
    file_name = f"audio/{uuid.uuid4()}.wav"

    # Upload to S3
    upload_to_s3(audio_bytes, file_name)

    # Transcribe the audio
    with st.spinner("Transcribing recorded audio..."):
        transcript = transcribe_audio(file_name)

    if transcript:
        st.subheader("Transcript")
        transcript_area = st.text_area("Transcript", transcript, height=300)

        if st.button("Summarize Transcript"):
            with st.spinner("Summarizing text..."):
                summary = summarize_text(transcript_area)

            st.subheader("Summary")
            st.write(summary)

            if summary:
                st.download_button("Download Summary", summary, file_name="summary.txt", mime="text/plain")
    else:
        st.error("No transcript available.")

st.markdown("## And Upload a Recorded Audio File for Transcription and Summarization")
# Fallback file uploader for manual uploads
uploaded_file = st.file_uploader("", type=["wav"])

if uploaded_file:
    st.audio(uploaded_file, format='audio/wav')
    audio_bytes = BytesIO(uploaded_file.read())
    file_name = f"audio/{uuid.uuid4()}.wav"
    upload_to_s3(audio_bytes, file_name)

    # Transcribe the uploaded file
    with st.spinner("Transcribing uploaded audio..."):
        transcript = transcribe_audio(file_name)

    if transcript:
        st.subheader("Transcript")
        transcript_area = st.text_area("Transcript", transcript, height=300)

        if st.button("Summarize Transcript"):
            with st.spinner("Summarizing text..."):
                summary = summarize_text(transcript_area)

            st.subheader("Summary")
            st.write(summary)

            if summary:
                st.download_button("Download Summary", summary, file_name="summary.txt", mime="text/plain")
    else:
        st.error("No transcript available.")
