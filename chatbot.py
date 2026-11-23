from langchain_core.messages import AIMessage, SystemMessage, HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import streamlit as st

load_dotenv()

sys_msg = "You are helpful IA GEMINI Chatbot"
# Initialize chat history in session state if not already present
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        SystemMessage(content=sys_msg)
    ]

# Initialize the LLM model
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
)

st.header("GEMINIA ChatBot")

# Display existing chat messages
for message in st.session_state.chat_history:
    if isinstance(message, HumanMessage):
        st.write(f"You: {message.content}")
    elif isinstance(message, AIMessage):
        st.write(f"Bot: {message.content}") # Changed "Response" to "Bot" for consistency

# Input field for user
user_input = st.text_input("You: ", key="user_input")

if st.button("Submit"):
    if user_input:
        # Append human message to chat history
        st.session_state.chat_history.append(HumanMessage(content=user_input))

        # Handle "exit" command
        if user_input.lower() == "exit":
            st.session_state.chat_history = [SystemMessage(content=sys_msg)]
            st.warning("Chat has been reset.")
            st.rerun() # Rerun to clear the input and display reset state
        else:
            try:
                with st.spinner("Generating Response: "):
                    # Invoke the LLM with the current chat history
                    result = llm.invoke(st.session_state.chat_history)
                    # Append AI message to chat history
                    st.session_state.chat_history.append(AIMessage(content=result.content))
                    # Rerun to display the new message and clear input
                    st.rerun()
            except Exception as e:
                st.error(f"Error: {type(e).__name__}: {e}")
    else:
        st.warning("Please enter some input.")

# Optionally, add a button to clear the chat
if st.button("Clear Chat"):
    st.session_state.chat_history = [SystemMessage(content=sys_msg)]
    st.info("Chat history cleared.")
    st.rerun()
