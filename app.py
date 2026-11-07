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
        "(in .streamlit/secrets.toml locally or in your Streamlit Cloud App Settings)."
    )
    st.stop()

# Initialize Google GenAI Client
client = genai.Client(api_key=api_key)

# Model ID
MODEL_ID = "gemini-3.5-flash-lite"

# -----------------------------------------------------------------------------
# Jefferson Persona, Academic Level & Archival Prompt
# -----------------------------------------------------------------------------
SYSTEM_INSTRUCTION = """
You are Thomas Jefferson: third President of the United States, primary draftsman 
of the Declaration of Independence, polymath, and founder of the University of Virginia.

Target Audience:
You are addressing American high school students reading at a 10th-grade reading level.

Tone & Demeanor:
Speak in the authentic, courteous cadence of an 18th-century statesman (e.g., "my young friend," 
"it is my considered judgment," "upon reflection"), but keep vocabulary and sentence structures 
understandable, engaging, and clear for a tenth grader.

Authorized Knowledge Base & Archives:
Base your answers strictly on historical evidence, primary sources, and scholarship from:
1. Founders Online (National Archives)
2. The Papers of Thomas Jefferson project at Princeton University
3. Library of Congress Manuscript Division (Thomas Jefferson Papers, 1606–1827)
4. Thomas Jefferson Foundation / Monticello Research Databases & Online Resources
5. Jefferson Library Digital Archive & UVA Thomas Jefferson Digital Archive
6. Massachusetts Historical Society (Coolidge Collection of Jefferson Manuscripts)
7. Swem Library Thomas Jefferson Project (College of William & Mary)
8. The Avalon Project at Yale Law School (treaties, statutes, and congressional records)
9. Official government documents, congressional debates, state papers, and Supreme Court rulings.

Strict Rules:
1. WORD LIMIT: Your response must strictly contain between 15 and 120 words in total.
2. CITATION: Conclude every answer with a clear citation showing where the primary source or record is preserved (e.g., "[Source: Letter to James Madison, Dec. 20, 1787, Founders Online / LoC]").
3. VISUAL / MAP URL: Provide a functional markdown link or URL to a relevant historical image, document scan, or map (e.g., from Library of Congress, National Archives, Monticello, or Wikimedia Commons historical collections) to assist the student.
"""

# -----------------------------------------------------------------------------
# Chat Interface & History
# -----------------------------------------------------------------------------
st.title("📜 In Dialogue with Thomas Jefferson")
st.markdown(
    "*Third President of the United States • Primary Author of the Declaration of Independence*"
)
st.caption("Historical dialogue for 10th-grade American History students.")

# Initialize conversation history in session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous messages from session state
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# User question input
if user_input := st.chat_input("Inquire of Mr. Jefferson..."):
    # Append user question
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Maintain memory of only the previous six questions (up to 12 total messages)
    recent_messages = st.session_state.messages[-12:]

    # Convert chat history into Google GenAI Content types
    contents = []
    for m in recent_messages:
        role = "user" if m["role"] == "user" else "model"
        contents.append(
            types.Content(
                role=role,
                parts=[types.Part.from_text(text=m["content"])]
            )
        )

    # Generate Jefferson's response
    with st.chat_message("assistant"):
        with st.spinner("Mr. Jefferson is dipping his quill..."):
            try:
                response = client.models.generate_content(
                    model=MODEL_ID,
                    contents=contents,
                    config=types.GenerateContentConfig(
                        system_instruction=SYSTEM_INSTRUCTION,
                        temperature=0.3,
                        max_output_tokens=350,
                    ),
                )
                answer_text = response.text
                st.markdown(answer_text)

                # Store assistant response in history
                st.session_state.messages.append({"role": "assistant", "content": answer_text})

            except Exception as e:
                st.error(f"An error occurred while generating the response: {e}")
