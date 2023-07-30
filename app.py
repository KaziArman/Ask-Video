from PIL import Image
import requests
from io import BytesIO
import openai
import streamlit as st
import utils as utl
import os
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains import ConversationalRetrievalChain
from youtube_transcript import YT_transcript
from chunk import chunking
import moviepy.editor as mp
from moviepy import video as mp2
import whisper
import os
import tempfile
from io import BytesIO
import shutil
#from pydub import AudioSegment
image_path = "meetai.png"
def resize_image(image_url, width):
    response = requests.get(image_url)
    image = Image.open(BytesIO(response.content))
    image = image.resipze((width, int(image.size[1] * (width / image.size[0]))))
    return image
ss = st.session_state
st.set_page_config(
	#page_icon="https://i.pinimg.com/originals/4b/85/c9/4b85c95c93eff810b0fe0a755be081a6.png",
	page_icon="meetai.png",
	layout="wide",
	page_title='MeetAi!',
	initial_sidebar_state="expanded"
)
image_url = "https://i.imgur.com/c6vFnyp.png"
image_width = 200 # Set the desired width of the image
# resized_image = resize_image("meetai.png", image_width)
st.markdown('<div class="right">', unsafe_allow_html=True)
st.image("meetai.png", caption="Simplify Meetings, Empower Answers: MeetAI!")




st.markdown(f'<div class="header"><figure><figcaption><h1>Welcome to MeetAI</h1></figcaption></figure><h3>MeetAi is a conversional AI based tool, where you can ask about any recorded video and it will answer.</h3></div>', unsafe_allow_html=True)
#st.markdown(f'<div class="header"><figure><img src="logo.png" width="500"><figcaption><h1>Welcome to Ask Youtube</h1></figcaption></figure><h3>Ask Youtube is a conversional AI based tool, where you can ask about any youtube video and it will answer.</h3></div>', unsafe_allow_html=True)

with st.expander("How to use MeetAI 🤖", expanded=False):

	st.markdown(
		"""
		Please refer to [our dedicated guide](https://www.impression.co.uk/resources/tools/oapy/) on how to use MeetAI.
		"""
    )

with st.expander("Credits 🏆", expanded=True):

	st.markdown(
		"""
		MeetAI was created by [Kazi Arman Ahmed](https://www.linkedin.com/in/r4h4t/) and [Md Shamim Hasan](https://www.linkedin.com/in/md-shamim-hasan/)  at [LandQuire](https://www.linkedin.com/company/landquire/) in Bangladesh.
	    """
    )

#st.markdown("---")
# Load your API key
api_key = st.text_input('Enter your API key')
# print(api_key)
openai.api_key = api_key
os.environ["OPENAI_API_KEY"] = api_key


video_file = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])
if video_file is not None:
    # Process the uploaded video file (you can add your processing code here)
    # st.video(video_file)
    temp_dir = tempfile.TemporaryDirectory()
    temp_file_path = os.path.join(temp_dir.name, "temp_video.mp4")
    with open(temp_file_path, "wb") as temp_file:
        temp_file.write(video_file.read())
    video=mp.VideoFileClip(temp_file_path)
    success_message = st.empty()
    success_message.success("Processing audio.......")
    aud=video.audio.write_audiofile("demo.mp3")
    model=whisper.load_model("base")
    result=model.transcribe("audio_out.mp3")

# a_file=st.audio("demo.mp3", format="audio/mp3")
    # Audio=mp.AudioFileClip("demo.mp3")
    # model=whisper.load_model("base")
    # result=model.transcribe(Audio)
# if a_file is not None:
# 	temp_dir1=tempfile.TemporaryDirectory()
# 	temp_file_path1 = os.path.join(temp_dir1.name, "temp_audio.mp3")
    # with open(temp_file_path1, "wb") as temp_file1:
    #     temp_file1.write(a_file.read())

	# audio_p=mp.AudioFileClip(temp_file_path1)
	# model=whisper.load_model("base")
	# result=model.transcribe(temp_file_path1)
    
    # with open("transcription.text",'w') as f:
    #     f.write(result['text'])
    # mp3_file_path=os.path.join(temp_dir.name,'output_audio.mp3')
    # audio.export(mp3_file_path,format='mp3')
    # st.success("Video successfully converted to mp3!")
    # st.audio(mp3_file_path)
    # temp_dir.cleanup()

    # video = mp.VideoFileClip(video_file)
    # audio = video.audio
    # mp3_file_path = "output_audio.mp3"
    # audio.export(mp3_file_path, format="mp3")
    # st.success("Video successfully converted to MP3!")
    # st.audio(mp3_file_path)
    # aud=video.audio
    # with st.spinner("Processing audio..."):
    #     aud.export("demo.mp3", format="mp3")
    #     st.success("Audio processing complete!")
	

# Process the audio data and write it to an audio file

# Load your Youtube Video Link
# youtube_link = st.text_input('Enter your YouTube video link')

temp_slider = st.sidebar.slider('Set the temperature of the completion. Higher values make the output more random,  lower values make it more focused.', 0.0, 1.0, 0.7)
def ui_question():
	st.write('### Ask Questions')
	disabled = False
	st.text_area('question', key='question', height=100, placeholder='Enter question here', help='', label_visibility="collapsed", disabled=disabled)
ui_question()

def output_add(q,a):
	if 'output' not in ss: ss['output'] = ''
	q = q.replace('$',r'\$')
	a = a.replace('$',r'\$')
	new = f'#### {q}\n{a}\n\n'
	ss['output'] = new + ss['output']
	st.markdown(new)
generate = st.button('Generate!')
def ui_output():
	output = ss.get('output','')
	st.markdown(output)
ui_output()
if generate:
	with st.spinner('Classifying...'):
		openai.api_key = api_key
		os.environ["OPENAI_API_KEY"] = api_key
		file_path = next(iter(video_file))
		video=moviepy.editor.VideoFileClip(file_path)
		aud=video.audio
		
  
  
		yt_script = YT_transcript(youtube_link)
		transcript = yt_script.script()
		text = "\n\n\nI have given you the caption of a Youtube video. " \
			   "I will ask you specific question about this video. " \
			   "You have to understand the theme of the video and" \
			   " answer me very precisely according to the questions"
		transcript = transcript+text
		chnk = chunking(transcript)
		chunks = chnk.yt_data()
		# print(chunks)
		# Get embedding model
		embeddings = OpenAIEmbeddings()
		# Create vector database
		chat_history=[]
		question = ss.get('question', '')
		temperature = ss.get('temperature', 0.0)
		temperature = temp_slider
		model = OpenAI(model="text-davinci-003", temperature=temp_slider)
		db = FAISS.from_documents(chunks, embeddings)
		qa = ConversationalRetrievalChain.from_llm(model, db.as_retriever())
	result = qa({"question": question, "chat_history": chat_history})
	chat_history.append((question, result['answer']))
	# print("\n\n\n\n")
	# print(chat_history)
	q = question.strip()
	a = result['answer'].strip()
	ss['answer'] = a
	output_add(q,a)

# Loading CSS
utl.local_css("frontend.css")
utl.remote_css('https://fonts.googleapis.com/icon?family=Material+Icons')
utl.remote_css('https://fonts.googleapis.com/css2?family=Red+Hat+Display:wght@300;400;500;600;700&display=swap')
