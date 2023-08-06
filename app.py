import openai
import streamlit as st
import os
import utils as utl
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.chains.question_answering import load_qa_chain
from youtube_transcript import YT_transcript
from chunk import chunking
import moviepy.editor as mp
import tempfile
import whisper
from moviepy.editor import VideoFileClip

IFRAME = '<iframe src="https://ghbtns.com/github-btn.html?user=KaziArman&repo=ask-youtube&type=star&count=true&size=large" frameborder="0" scrolling="0" width="170" height="30" title="GitHub"></iframe>'

ss = st.session_state
st.set_page_config(
	page_icon="https://i.pinimg.com/originals/4b/85/c9/4b85c95c93eff810b0fe0a755be081a6.png",
	#page_icon="web.png",
	layout="wide",
	page_title='Ask Youtube',
	initial_sidebar_state="expanded"
)

st.markdown(f'<div class="header"><figure style="background-color: #FFFFFF;"><img src="https://i.pinimg.com/originals/41/f6/4d/41f64d3b4b21cb08eb005b11016bf707.png" width="400" style="display: block; margin: 0 auto; background-color: #FFFFFF;"><figcaption></figcaption></figure><h4 style="text-align: center"> Ask Youtube is a conversional AI based tool, where you can ask about any youtube video and it will answer  {IFRAME}</h4></div>', unsafe_allow_html=True)
#st.markdown(f'<div class="header"><figure><img src="logo.png" width="500"><figcaption><h1>Welcome to Ask Youtube</h1></figcaption></figure><h3>Ask Youtube is a conversional AI based tool, where you can ask about any youtube video and it will answer.</h3></div>', unsafe_allow_html=True)

with st.expander("How to use Ask Youtube 🤖", expanded=False):
	st.markdown(
		"""
		Please refer to [our dedicated guide](https://www.impression.co.uk/resources/tools/oapy/) on how to use Ask Youtube.
		"""
    )

with st.sidebar.expander("Credits 🏆", expanded=True):
	st.markdown(
		"""Ask Youtube was created by [Kazi Arman Ahmed](https://www.linkedin.com/in/r4h4t/) and [Md Shamim Hasan](https://www.linkedin.com/in/md-shamim-hasan/)  at [LandQuire](https://www.linkedin.com/company/landquire/) in Bangladesh 
	    """
    )
def on_btn_click():
    del st.session_state["messages"] 
    st.session_state["messages"] = [{"role": "assistant", "content": "Hi, I am an AI bot created by LandQuire Data Team\nHow can I help you regarding this YouTube video?"}]


api_key = st.sidebar.text_input('Enter your API key')



def convert_video_to_audio(input_video_path, output_audio_path):
    try:
        # Load the video file
        video_clip = VideoFileClip(input_video_path)

        # Extract audio from the video clip
        audio_clip = video_clip.audio

        # Save the audio to the specified output path
        audio_clip.write_audiofile(output_audio_path)

        # Close the video and audio clips
        video_clip.close()
        audio_clip.close()

        st.success("Conversion complete. Audio file saved at: " + output_audio_path)
    except Exception as e:
        st.error("Error occurred: " + str(e))




uploaded_file = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])
if uploaded_file:
    # temp_dir = tempfile.TemporaryDirectory()
    # temp_file_path = os.path.join(temp_dir.name, "temp_video.mp4")
    try:
        with open("temp_video.mp4", "wb") as f:
            f.write(uploaded_file.read())
        output_audio_file = "output_audio.mp3"
        convert_video_to_audio("temp_video.mp4", output_audio_file)
        os.remove("temp_video.mp4")
        uploaded_file=st.empty()
    except:
        output_audio_file = "output_audio.mp3"
    
    # video=mp.VideoFileClip(video_file)
    # success_message = st.empty()
    # success_message.success("Processing audio.......")
    # aud=video_file.audio.write_audiofile("demo180.mp3")
    model=whisper.load_model("base")
    result=model.transcribe(output_audio_file)
    success_message = st.empty()
    # print(result)
    success_message.success("Audio Processing Done")
    


    text = """\n\n\nPlease briefly summarize the video transcription mentioned above. Now make it in 2-3 concise paragraphs.
            First, analyze the video line-by-line, distilling each line into a simple summary sentence.
            Then combine these sentence summaries into a long summary covering all the points. 
            Please make sure to accurately capture the essence and meaning of the video content.
            I will ask some questions about the video - please answer them with relevant details from the video transcriuption.
            Aim for answers that are around 50 lines long.
            The goal is for your summary to allow me to get all the information from the video without watching it, and have enough detail to answer specific questions"""

    transcript = result['text'] + text
    print(transcript)
    chnk = chunking(transcript)
    chunks = chnk.yt_data()
    if api_key:
        openai.api_key = api_key
        os.environ["OPENAI_API_KEY"] = api_key
        # Get embedding model
        embeddings = OpenAIEmbeddings()
        db = FAISS.from_documents(chunks, embeddings)
    
    else:
        st.info("Please add your OpenAI API key to continue.")



temp_slider = st.sidebar.slider('Set the temperature of the completion. Higher values make the output more random,  lower values make it more focused.', 0.0, 1.0, 0.7)
st.sidebar.button("Clear messages", use_container_width=True,on_click=on_btn_click)
summarize = st.sidebar.button("Summarize", use_container_width=True)
chatgpt = st.sidebar.checkbox('Use ChatGPT', False,
			help='Allow the bot to collect answers to any specific questions from outside of the video content')



st.write('### Ask Questions')
chat = st.chat_input()

if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "Hi, I am an AI bot created by LandQuire Data Team\nHow can I help you regarding this YouTube video?"}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])


if chatgpt:

    if prompt := chat:
        if not api_key:
            st.info("Please add your OpenAI API key to continue.")
            st.stop()
        openai.api_key = api_key
        prompt = f'##### {prompt}'
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)

        temp = temp_slider
        response = openai.ChatCompletion.create(model="gpt-3.5-turbo",temperature=temp, messages=st.session_state.messages)
        msg = response.choices[0].message
        st.session_state.messages.append(msg)
        st.chat_message("assistant").write(msg.content)


else:
    if summarize:
        if not api_key:
            st.info("Please add your OpenAI API key to continue.")
            st.stop()
        data = db
        prompt1 = "Please summarize this YouTube video briefly. Your summary should be brief enough that I don't need to watch the video to understand the key points. Include the most important details and topics covered in the video by accurately extracting the core essence and main ideas from the content. Make sure to cover all the key points mentioned without leaving any major details out. The objective is to provide a details at least 3 paragrapghs and comprehensive text summary so I can get all the key information from the video without having to view it."
        prompt3 ='First, divide the entire video transcription into three parts based on duration. Then, begin by describing the content of the first part according to the dividation. Your response should be accurate, and contain at least 50 words or 5 sentences.'
        prompt4 ='Now, begin by describing the content of the second part according to the previous dividation. Your response should be accurate, and contain at least 50 words or 5 sentences.'
        prompt5 = 'Now, begin by describing the content of the third part according to the previous dividation. Your response should be accurate, and contain at least 50 words or 5 sentences.'
        prompt6 = 'Now, provide a summary of the entire video  while ensuring all the topics covered are included.Your response should be accurate, and contain at least 100 words or 10 sentences.'
        prompt2 = "Summarize the video"
        openai.api_key = api_key
        prompt = f'##### {prompt3}'
        st.session_state.messages.append({"role": "user", "content": prompt2})
        st.chat_message("user").write(prompt2)
        out = ""
        ll = [prompt3,prompt4,prompt5,prompt6]
        for i in ll:
            pp = i
            print(pp)
            temp = temp_slider
            model = OpenAI(model="text-davinci-003", temperature=temp)
            chain = load_qa_chain(model, chain_type="stuff")  # chain_type="stuff"
            docs = data.similarity_search(pp)
            res = chain.run(input_documents=docs, question=pp, messages=st.session_state.messages)
            #print(res)
            out = out+"\n\n"+res
            #print(type(res))
            print(out)
        st.session_state.messages.append({"role": "assistant", "content": out})
        st.chat_message("assistant").write(out)

    if prompt := chat:
        if not api_key:
            st.info("Please add your OpenAI API key to continue. ")
            st.stop()
        data = db

        openai.api_key = api_key
        prompt = f'##### {prompt}'

        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)

        temp = temp_slider
        model = OpenAI(model="text-davinci-003", temperature=temp)
        chain = load_qa_chain(model, chain_type="stuff") #chain_type="stuff"
        docs = data.similarity_search(prompt)
        res = chain.run(input_documents=docs, question=prompt, messages=st.session_state.messages)
        st.session_state.messages.append({"role": "assistant", "content": res})
        st.chat_message("assistant").write(res)






