import streamlit as st
from streamlit import spinner
import re 
import datetime
import data_base as db 
import os

from supporting_functions import (
    extract_video_id,
    get_transcript,
    translate_transcript,
    generate_notes,
    get_important_topics,
    create_chunks,
    create_vector_store,
    load_vector_store, 
    rag_answer,
    normalize_language,
    generate_chat_title,
    generate_summary
)

# initialize database on first run 
db.init_db()

# session state initialization
def initialize_state():
    """Initializes session state, loading chat history from DB if needed."""
    
    if "active_chat_id" not in st.session_state:
        st.session_state.active_chat_id = None

    # 'chat_sessions' holds all chat metadata (title, messages)
    # It's now loaded from the DB once and kept in sync
    if "chat_sessions" not in st.session_state:
        st.session_state.chat_sessions = db.load_chat_sessions()

    # 'vector_stores' is NO LONGER stored in session state.
    if "vector_stores" in st.session_state:
        del st.session_state.vector_stores

initialize_state()

# Callback function to switch chats 
def set_active_chat(chat_id):
    st.session_state.active_chat_id = chat_id

# Sidebar (Inputs)
with st.sidebar:
    st.title("🎬 Youtube_Rag")
    st.markdown("---")

    st.button("New Video Analysis", on_click=set_active_chat, args=(None,), use_container_width=True)

    st.markdown("### Chat History")

    # Get chat ids from the loaded session state
    chat_ids = list(st.session_state.chat_sessions.keys())
    
    # DB query already orders by most recent

    if not chat_ids:
        st.caption("No chats yet ")

    # display a button for each past chat session 
    for chat_id in chat_ids :
        session = st.session_state.chat_sessions[chat_id]
        st.button(session["title"], 
                key=chat_id, 
                on_click=set_active_chat, 
                args=(chat_id,), 
                use_container_width=True
                )
    st.markdown("---")


# --- "NEW CHAT" PAGE ---
if st.session_state.active_chat_id is None:
    
    st.title("YouTube Content Synthesizer")
    st.markdown("Transform any YouTube video into key topics, a podcast, or a chatbot.")
    st.markdown("### Input Details")

    youtube_url = st.text_input("YouTube URL", placeholder="https://www.youtube.com/watch?v=...")

    user_language_input = st.text_input(
        "Video Language", 
        placeholder="e.g., English, Hindi, en, hi", 
        value="English"
    )

    task_option = st.radio(
        "Choose what you want to generate:",
        ["Chat with Video", "Notes For You", "Get Summary"],
        key="task_option"
    )

    submit_button = st.button("✨ Start Processing")
    st.markdown("---")

    if submit_button:
        language = normalize_language(user_language_input)
        
        if youtube_url and language:
            video_id = extract_video_id(youtube_url)
            
            if video_id:
                full_transcript = None # Initialize
                with spinner(f"Step 1/3 : Fetching Transcript (Language: {language})....."):
                    full_transcript = get_transcript(video_id, language)

                # Check if transcript was successfully fetched
                if full_transcript and language != "en":
                    with spinner("Step 1.5/3 : Translating Transcript into English..."):
                        translated_transcript = translate_transcript(full_transcript)
                        if translated_transcript:
                            full_transcript = translated_transcript
                        else:
                            st.error("Failed to translate transcript. Aborting.")
                            full_transcript = None # Mark as failed

                # Proceed only if we have a valid transcript (in English)
                if full_transcript:
                    if task_option == "Notes For You":
                        with spinner("Step 2/3: Extracting important Topics..."):
                            import_topics = get_important_topics(full_transcript)
                            st.subheader("Important Topics")
                            st.write_stream(import_topics)
                            st.markdown("---")

                        with spinner("Step 3/3 : Generating Notes for you."):
                            notes = generate_notes(full_transcript)
                            st.subheader("Notes for you")
                            st.write_stream(notes)

                        st.success("Summary and Notes Generated.")

                    elif task_option == "Get Summary":
                        with spinner("Generating Summary..."):
                            summary = generate_summary(full_transcript)
                            st.subheader("Video summary")
                            st.write_stream(summary)
                        st.success("Summary Generated.")
                        

                    elif task_option == "Chat with Video":
                        new_chat_id = f"chat_{datetime.datetime.now().timestamp()}"
                        vectorstore = None
                        with st.spinner("Step 2/3: Creating chunks and vector store...."):
                            chunks = create_chunks(full_transcript)
                            vectorstore = create_vector_store(chunks, new_chat_id)
                        
                        if vectorstore: # Check if vector store creation was successful
                            chat_title = None

                            with st.spinner("step 3/3: Generating chat title..."):
                                chat_title = generate_chat_title(full_transcript)
                            
                            if not chat_title or len(chat_title) < 5:
                                chat_title = f"Chat: {video_id}"   #fallback

                            st.success('Step 3/3: Video is ready for chat!')

                             
                            db.save_chat_session(new_chat_id, chat_title, youtube_url)
                            
                            initial_message = "Video processed! You can now ask me any questions about it."
                            db.save_message(new_chat_id, "assistant", initial_message)


                            #update session state (newest first) 
                            new_session = {
                                new_chat_id:{
                                    "title":chat_title,
                                    "video_url":youtube_url,
                                    "messages":[{"role": "assistant", "content":initial_message}]
                                }
                            }
                            
                            # Prepend the new session to the existing sessions
                            st.session_state.chat_sessions = {**new_session, **st.session_state.chat_sessions}

                            
                            # st.session_state.chat_sessions[new_chat_id] = {
                            #     "title": chat_title,
                            #     "video_url": youtube_url,
                            #     "messages": [{"role": "assistant", "content": initial_message}]
                            # }
                            
                            # Set this new chat as active and rerun
                            set_active_chat(new_chat_id)
                            st.rerun()
                        else:
                            st.error("Failed to create vector store. Please try again.")
                
                elif not full_transcript and video_id:
                    # Error was already shown by get_transcript or translate_transcript
                    pass 
            # else: (error already shown by extract_video_id)
        else:
            st.warning("Please provide both a YouTube URL and a language.")


# --- ACTIVE CHAT PAGE ---
else:
    # Get the data for the *currently active* chat
    chat_id = st.session_state.active_chat_id
    
    if chat_id not in st.session_state.chat_sessions:
        st.error("Chat session not found. Starting a new analysis.")
        set_active_chat(None)
        st.rerun()
    
    session = st.session_state.chat_sessions[chat_id]

    # Load vector store on-demand from disk 
    vector_store = load_vector_store(chat_id)
    
    st.header(f"Chat about: {session['title']}")
    st.caption(f"Source: {session['video_url']}")
    st.divider()

    # Load messages from DB if not already in session 
    if "messages" not in session:
        session["messages"] = db.load_messages(chat_id)

    # Display the entire history for THIS chat
    for message in session['messages']:
        with st.chat_message(message['role']):
            st.write(message['content'])

    # user_input
    prompt = st.chat_input("Ask me anything about the video.")
    if prompt:
        # Append to session state and save to DB
        session['messages'].append({'role': 'user', 'content': prompt})
        db.save_message(chat_id, 'user', prompt)
        
        with st.chat_message('user'):
            st.write(prompt)

        with st.chat_message('assistant'):
            response_chunks = []

            def stream_and_capture():
                # --- MODIFIED: Pass the loaded vector_store ---
                for chunk in rag_answer(prompt, vector_store):
                    response_chunks.append(chunk)
                    yield chunk

            st.write_stream(stream_and_capture())

            full_response = "".join(response_chunks)
            
            # Append to session state and save to DB
            session['messages'].append({'role': 'assistant', 'content': full_response})
            db.save_message(chat_id, 'assistant', full_response)

