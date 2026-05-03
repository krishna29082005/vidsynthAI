import time
import os

from dotenv import load_dotenv
import re
import streamlit as st

from youtube_transcript_api import YouTubeTranscriptApi

from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.output_parsers import StrOutputParser

from langchain_core.prompts import PromptTemplate

from langchain_core.messages import HumanMessage, AIMessage



load_dotenv()

# This ensures the same model is used for creating and loading
EMBEDDING_MODEL = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
VECTOR_STORE_DIR = "vector_stores"

def get_vector_store_path(chat_id):
    """Returns the persistent storage path for a given chat_id."""
    return os.path.join(VECTOR_STORE_DIR, chat_id)

# Function to extract video ID from a YouTube URL (Helper Function)
def extract_video_id(url):
    """
    Extracts the YouTube video ID from any valid YouTube URL.
    """
    match = re.search(r"(?:v=|\/)([0-9A-Za-z_-]{11}).*", url)
    if match:
        return match.group(1)
    st.error("Invalid YouTube URL. Please enter a valid video link.")
    return None


# function to get transcript from the video.
def get_transcript(video_id, language):
    ytt_api= YouTubeTranscriptApi()
    """ytt_api = YouTubeTranscriptApi(
    proxy_config = WebshareProxyConfig(
    proxy_username="198.23.239.134" ,
    proxy_password="80ulzdf20b0o"))
    """
    try:
        transcript= ytt_api.fetch(video_id, languages=[language])
        full_transcript= " ".join([i.text for i in transcript])
        time.sleep(10)
        return full_transcript
    except Exception as e:
        st.error(f"Error fething video {e}")
        st.error(f"Could not fetch transcript for video {video_id} with language '{language}'. The video might not have transcripts available in that language.")
        return None # Return None on failure



# function to translate the transcript into english.
    # initialize the gemini model
llm= ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite",
    temperature=0.2,
    streaming = True
)


def translate_transcript(transcript):
    try:
        prompt=ChatPromptTemplate.from_template("""
        You are an expert translator with deep cultural and linguistic knowledge.
        I will provide you with a transcript. Your task is to translate it into English with absolute accuracy, preserving:
        - Full meaning and context (no omissions, no additions).
        - Tone and style (formal/informal, emotional/neutral as in original).
        - Nuances, idioms, and cultural expressions (adapt appropriately while keeping intent).
        - Speaker’s voice (same perspective, no rewriting into third-person).
        Do not summarize or simplify. The translation should read naturally in the target language but stay as close as possible to the original intent.

        Transcript:
        {transcript}
        """)

        #Runnable chain
        chain= prompt|llm|StrOutputParser()

        #Run chain
        response= chain.invoke({"transcript":transcript})

        return response
    
    except Exception as e:
        st.error(f"Error fething video {e}")
        return None


# function to get important topics
def get_important_topics(transcript):
    try:
        prompt = ChatPromptTemplate.from_template("""
               You are an assistant that extracts the 5 most important topics discussed in a video transcript or summary.

               Rules:
               - Summarize into exactly 5 major points.
               - Each point should represent a key topic or concept, not small details.
               - Keep wording concise and focused on the technical content.
               - Do not phrase them as questions or opinions.
               - Output should be a numbered list.
               - show only points that are discussed in the transcript.
               Here is the transcript:
               {transcript}
               """)

        # Runnable chain
        chain = prompt | llm | StrOutputParser()

        # Run chain
        yield from chain.stream({"transcript":transcript})


    except Exception as e:
        st.error(f"Error fething video {e}")



# FUNCTION TO GET NOTES FROM THE VIDEO
def generate_notes(transcript):
    try:
        prompt = ChatPromptTemplate.from_template("""
                You are an AI note-taker. Your task is to read the following YouTube video transcript 
                and produce well-structured, concise notes.

                ⚡ Requirements:
                - Present the output as **bulleted points**, grouped into clear sections.
                - Highlight key takeaways, important facts, and examples.
                - Use **short, clear sentences** (no long paragraphs).
                - If the transcript includes multiple themes, organize them under **subheadings**.
                - Do not add information that is not present in the transcript.

                Here is the transcript:
                {transcript}
                """)

        # Runnable chain
        chain = prompt | llm |StrOutputParser()

        # Run chain
        yield from chain.stream({"transcript":transcript})


    except Exception as e:
        st.error(f"Error fething video {e}")




# funtions to create chunks 
def create_chunks(transcript):
    text_splitters= RecursiveCharacterTextSplitter(chunk_size=10000,chunk_overlap=1000)
    doc= text_splitters.create_documents([transcript])
    return doc




def create_vector_store(docs, chat_id):
    """Creates and persists a Chroma vector store to disk."""
    persist_directory = get_vector_store_path(chat_id)
    
    try:
        vector_store = Chroma.from_documents(
            documents=docs, 
            embedding=EMBEDDING_MODEL, 
            persist_directory=persist_directory
        )
        return vector_store
    except Exception as e:
        st.error(f"Error creating vector store: {e}")
        return None
    
def load_vector_store(chat_id):
    """Loads a persistent Chroma vector store from disk."""
    persist_directory = get_vector_store_path(chat_id)
    
    if not os.path.exists(persist_directory):
        st.error(f"Vector store not found for chat {chat_id}. Please re-process the video.")
        return None
        
    try:
        vector_store = Chroma(
            persist_directory=persist_directory, 
            embedding_function=EMBEDDING_MODEL
        )
        return vector_store
    except Exception as e:
        st.error(f"Error loading vector store: {e}")
        return None

# Dictionary mapping common language inputs (lowercase) to language codes
LANGUAGE_MAP = {
    'english': 'en',
    'en': 'en',
    'eng': 'en',
    'hindi': 'hi',
    'hi': 'hi',
    'spanish': 'es',
    'es': 'es',
    'espanol': 'es',
    'french': 'fr',
    'fr': 'fr',
    'francais': 'fr',
    'german': 'de',
    'de': 'de',
    'deutsch': 'de',
    'chinese': 'zh',
    'zh': 'zh',
    'japanese': 'ja',
    'ja': 'ja',
    'russian': 'ru',
    'ru': 'ru',
    'portuguese': 'pt',
    'pt': 'pt',
    'italian': 'it',
    'it': 'it',
    'korean': 'ko',
    'ko': 'ko',
    'arabic': 'ar',
    'ar': 'ar',
    # Add more mappings as needed
}

def normalize_language(user_input):
    """
    Normalizes user language input to a standard language code.
    Converts input to lowercase, strips whitespace, and maps to a code.
    """
    if not user_input:
        return "en" # Default to English if input is empty
        
    # Clean the input: convert to lowercase and remove leading/trailing spaces
    cleaned_input = user_input.lower().strip()
    
    # Look up the cleaned input in the map.
    # .get() provides a default value (the cleaned_input itself) if the key is not found.
    # This way, if a user enters a valid code like 'en-GB', it passes through.
    normalized_code = LANGUAGE_MAP.get(cleaned_input, cleaned_input)
    
    return normalized_code


def generate_chat_title(transcript):
    """
    Generates a concise 5-7 word title for a chat session based on the transcript.
    """
    try:
        prompt = ChatPromptTemplate.from_template("""
        Based on the following video transcript, generate a very short and descriptive title (about 5 to 7 words) that summarizes the main topic.
        
        - Do not use quotes.
        - Do not include "Video:" or "Chat:" prefixes.
        - Just provide the title itself.

        Transcript (first 5000 characters):
        {transcript}
        
        Title:
        """)

        # Runnable chain
        # Note: Using the non-streaming .invoke() method here
        chain = prompt | llm | StrOutputParser()

        # Run chain
        response = chain.invoke({"transcript": transcript[:5000]}) # Use first 5000 chars for speed
        
        # Clean up response (e.g., remove quotes or newlines)
        return response.strip().replace('"', '')

    except Exception as e:
        print(f"Error generating chat title: {e}")
        return None # Return None on failure



def generate_summary(transcript):
    """
    Generates a concise paragraph-style summary of the transcript.
    """
    try:
        prompt = ChatPromptTemplate.from_template("""
                 You are an expert summarizer. Your task is to read the following YouTube video transcript 
                 and produce a concise, paragraph-style summary.

                  Requirements:
                 - The output should be a single block of text (a paragraph).
                 - **Do not use bullet points or numbered lists.**
                 - Capture the main ideas, key arguments, and conclusions.
                 - The summary should be easy to read and flow naturally.
                 - Do not add information that is not present in the transcript.

                 Here is the transcript:
                 {transcript}
                 """)

        # Runnable chain
        chain = prompt | llm |StrOutputParser()

        # Run chain
        yield from chain.stream({"transcript":transcript})

    except Exception as e:
        st.error(f"Error generating summary: {e}")




def rag_answer(question, vectorstore):
    #  Add check in case vector store failed to load ---
    if vectorstore is None:
        yield "I'm sorry, but I can't access the video information for this chat. Please try creating a new chat."
        return

    try:
        results= vectorstore.similarity_search(question,k=4)
        context_text = "\n".join([i.page_content for i in results])

        prompt = ChatPromptTemplate.from_template("""
                    You are a kind, polite, and precise assistant.
                    - Begin with a warm and respectful greeting (avoid repeating greetings every turn).
                    - Understand the user’s intent even with typos or grammatical mistakes.
                    - Answer ONLY using the retrieved context.
                    - If answer not in context, say:
                        "I couldn’t find that information in the video transcript. Could you please rephrase or ask something else?"
                    - Keep answers clear, concise, and friendly.

                    Context:
                    {context}

                    User Question:
                    {question}

                    Answer:
                    """)

        #chain
        chain = prompt | llm | StrOutputParser() 

        # This turns the function into a generator for st.write_stream
        yield from chain.stream({"context":context_text,"question":question})
    
    except Exception as e:
        st.error(f"Error during RAG: {e}")
        yield "I'm sorry, I encountered an error while trying to answer your question."




#  from pytube import YouTube

# def get_youtube_title(url):
#     """
#     Fetches the actual video title from a YouTube URL.
#     """
#     try:
#         yt = YouTube(url)
#         return yt.title
#     except Exception as e:
#         print(f"Error fetching YouTube title: {e}")
#         return None # Return None on failure






