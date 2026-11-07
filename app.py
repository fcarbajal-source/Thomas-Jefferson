import streamlit as st
from google import genai
from google.genai import types

# -----------------------------------------------------------------------------
# Page Configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Thomas Jefferson - 3rd U.S. President",
    page_icon="📜",
    layout="centered"
)

# -----------------------------------------------------------------------------
# API Key Authentication (Streamlit Secrets)
# -----------------------------------------------------------------------------
api_key = st.secrets.get("GEMINI_API_KEY")
if not api_key:
    st.error(
        "Gemini API key not found. Please add GEMINI_API_KEY to your Streamlit secrets "
        "(in .streamlit/secrets.toml locally or in Streamlit Cloud App Settings)."
    )
    st.stop()

# Initialize Google GenAI Client
client = genai.Client(api_key=api_key)

# Model configuration
MODEL_ID = "gemini-3.5-flash-lite"

# -----------------------------------------------------------------------------
# Compact, Token-Efficient Persona & System Prompt
# -----------------------------------------------------------------------------
SYSTEM_INSTRUCTION = """
You are Thomas Jefferson: third President of the United States, draftsman of the 
Declaration of Independence, and founder of the University of Virginia.

Audience:
American 10th-grade high school history students.

Tone & Style:
Speak with the polite, dignified cadence of an 18th-century statesman (e.g., "my young friend," 
"it is my earnest conviction"), but keep explanations clear, direct, and accessible for a tenth grader.

Knowledge Base:
Base answers on authentic primary and secondary records:
- Letters to and from Jefferson (Founders Online / National Archives; Princeton Papers of Thomas Jefferson)
- Speeches, congressional acts, state papers, and Supreme Court cases (Library of Congress, Avalon Project at Yale)
- Monticello Research Databases, UVA Digital Archive, MHS Coolidge Collection, and Swem Library (William & Mary).

Strict Rules:
1. WORD LIMIT: Your entire answer must be strictly between 15 and 120 words.
2. CITATION: End every answer with a brief text attribution stating the document and archive (e.g., "[Source: Letter to James Madison, Dec. 20, 1787, Founders Online]").
3. NO PICTURES OR MAPS: Do NOT include URLs, hyperlinks, images, or maps. Keep the output plain text.
"""

# -----------------------------------------------------------------------------
# Chat Interface & History
# -----------------------------------------------------------------------------
st.title("📜 In Dialogue with Thomas Jefferson")
st.markdown("*Third President of the United States • Author of the Declaration of Independence*")
st.caption("10th-Grade History Edition • Fast & Quota-Optimized")

# Initialize conversation history in session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous conversation turns
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# User input
if user_input := st.chat_input("Inquire of Mr. Jefferson..."):
    # Append user question
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Maintain memory of strictly the previous 6 questions (up to 12 total messages)
    recent_messages = st.session_state.messages[-12:]

    # Convert session messages to Gemini SDK Content format
    contents = []
    for m in recent_messages:
        role = "user" if m["role"] == "user" else "model"
        contents.append(
            types.Content(
                role=role,
                parts=[types.Part.from_text(text=m["content"])]
            )
        )

    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Mr. Jefferson is composing his thoughts..."):
            try:
                response = client.models.generate_content(
                    model=MODEL_ID,
                    contents=contents,
                    config=types.GenerateContentConfig(
                        system_instruction=SYSTEM_INSTRUCTION,
                        temperature=0.3,
                        max_output_tokens=220,  # Limits token burn and enforces brevity
                    ),
                )

                answer_text = response.text or "I beg your pardon, but I could not formulate a reply."
                st.markdown(answer_text)

                # Save assistant response to session state
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer_text
                })

            except Exception as e:
                st.error(f"An error occurred: {e}")
