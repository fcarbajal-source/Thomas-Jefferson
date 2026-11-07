import streamlit as st
from google import genai
from google.genai import types

# -----------------------------------------------------------------------------
# Configuration & Initialization
# -----------------------------------------------------------------------------
st.set_page_config(page_title="Thomas Jefferson Chatbot", page_icon="📜", layout="centered")

# Retrieve API key securely from Streamlit secrets
api_key = st.secrets.get("GEMINI_API_KEY")
if not api_key:
    st.error("Missing Gemini API Key. Please configure it in your Streamlit secrets.")
    st.stop()

# Initialize Google GenAI client
client = genai.Client(api_key=api_key)

# Specify Gemini model (Adjust to "gemini-3.5-flash-lite" or current flash tier)
MODEL_ID = "gemini-2.5-flash"

# -----------------------------------------------------------------------------
# Jefferson Persona & System Prompt
# -----------------------------------------------------------------------------
SYSTEM_PROMPT = """
You are Thomas Jefferson (1743–1826): author of the Declaration of Independence, 
third President of the United States, diplomat, architect, and founder of the University of Virginia.

Your Audience:
American high school students reading at a 10th-grade level.

Voice & Tone:
Speak in an authentic, polite 18th-century statesman's voice (e.g., "my young friend," "it is my firm conviction"), 
but keep sentence structures and vocabulary clear, accessible, and engaging for a tenth grader.

Knowledge Base & Sources:
Base all statements on authentic records:
- Letters to and from Jefferson (Founders Online / National Archives; Papers of Thomas Jefferson project at Princeton).
- Official documents and speeches (Library of Congress Thomas Jefferson Papers, Avalon Project at Yale).
- University of Virginia Digital Archive; Monticello Research Databases; Massachusetts Historical Society (Coolidge Collection); Swem Library (William & Mary).
- Congressional acts, legislation, and Supreme Court rulings from your era.

Constraints:
1. WORD COUNT: Responses MUST be strictly between 15 and 120 words total.
2. CITATION: At the bottom of each answer, include a clear reference to the specific historical source (e.g., "Source: Letter to James Madison, Dec. 20, 1787, Founders Online").
3. VISUAL / MAP LINK: Whenever relevant, provide a markdown link or direct URL to a verified historical map, portrait, or document image from sources such as the Library of Congress, Wikimedia Commons, National Archives, or Monticello.
"""

# -----------------------------------------------------------------------------
# Session State & History Management
# -----------------------------------------------------------------------------
st.title("📜 In Council with Thomas Jefferson")
st.caption("A dialogue with the Sage of Monticello | Designed for 10th Grade Historians")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display prior chat messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# User prompt input
if user_prompt := st.chat_input("Ask Mr. Jefferson a question..."):
    # Append user question to history
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    with st.chat_message("user"):
        st.markdown(user_prompt)

    # Constraint: Only remember the previous 6 questions (and their paired replies)
    # 6 interactions = max 12 items in the conversation window
    recent_history = st.session_state.messages[-12:]

    # Format history for Gemini SDK
    gemini_contents = []
    for m in recent_history:
        role = "user" if m["role"] == "user" else "model"
        gemini_contents.append(
            types.Content(
                role=role,
                parts=[types.Part.from_text(text=m["content"])]
            )
        )

    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Mr. Jefferson is consulting his library..."):
            try:
                response = client.models.generate_content(
                    model=MODEL_ID,
                    contents=gemini_contents,
                    config=types.GenerateContentConfig(
                        system_instruction=SYSTEM_PROMPT,
                        temperature=0.3,
                        max_output_tokens=300
                    )
                )
                bot_reply = response.text
                st.markdown(bot_reply)

                # Store assistant response
                st.session_state.messages.append({"role": "assistant", "content": bot_reply})

            except Exception as e:
                st.error(f"An error occurred while generating the response: {e}")
