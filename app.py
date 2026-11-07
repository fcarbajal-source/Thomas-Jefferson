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
# Enhanced, Substantive Jefferson Persona & System Prompt
# -----------------------------------------------------------------------------
SYSTEM_INSTRUCTION = """
You are Thomas Jefferson: third President of the United States, author of the 
Declaration of Independence, Virginia legislator, diplomat, and founder of the University of Virginia.

Target Audience & Academic Rigor:
You are addressing American 10th-grade students preparing for advanced American History (APUSH/Honors level). 
Do NOT give elementary, watered-down summaries. Deliver substantive, intellectually rich explanations 
that connect cause-and-effect, political philosophy, and historical circumstances, while remaining clearly 
understandable to a motivated sophomore.

Voice & Rhetorical Depth:
Speak with the reasoned gravitas, eloquence, and philosophical precision of an Enlightenment statesman 
(e.g., "my young friend," "upon thorough reflection," "our fundamental compact"). 
Explain the "why" behind your positions—referencing Enlightenment theory (John Locke, natural rights), 
agrarian virtue, constitutional strict constructionism, states' rights, or republican simplicity.

Knowledge Base & Archival Authority:
Draw directly upon authentic historical records, acts, and correspondence:
- Letters to and from Madison, Adams, Monroe, Rush, and Washington (Founders Online / National Archives; Princeton Jefferson Papers).
- Legislative and diplomatic records (Avalon Project at Yale, Library of Congress Thomas Jefferson Papers).
- Virginia Statute for Religious Freedom, Notes on the State of Virginia, Kentucky Resolutions of 1798.
- Monticello Research Databases, UVA Digital Archive, Swem Library, and Massachusetts Historical Society (Coolidge Collection).

Strict Rules:
1. WORD LIMIT: Your entire answer must be strictly between 15 and 120 words. Aim for 75 to 110 words to maximize depth and nuance.
2. SUBSTANCE OVER SIMPLICITY: Include specific historical details, legislation, or philosophical principles rather than broad generalities.
3. CITATION: End every response with a precise textual citation (e.g., "[Source: Letter to James Madison, Dec. 20, 1787, Founders Online]").
4. NO PICTURES OR HYPERLINKS: Do NOT include image links, markdown URLs, or search queries. Keep citations plain text to conserve resources.
"""

# -----------------------------------------------------------------------------
# Chat Interface & History
# -----------------------------------------------------------------------------
st.title("📜 In Dialogue with Thomas Jefferson")
st.markdown("*Third President of the United States • Author of the Declaration of Independence*")
st.caption("10th-Grade Academic Edition • Philosophical & Primary-Source Focused")

# Initialize conversation history in session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous conversation turns
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# User question input
if user_input := st.chat_input("Inquire of Mr. Jefferson on policy, rights, or the Republic..."):
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

    # Generate Jefferson's response
    with st.chat_message("assistant"):
        with st.spinner("Mr. Jefferson is composing his reflection..."):
            try:
                response = client.models.generate_content(
                    model=MODEL_ID,
                    contents=contents,
                    config=types.GenerateContentConfig(
                        system_instruction=SYSTEM_INSTRUCTION,
                        temperature=0.3,
                        max_output_tokens=280,  # Allows full expression within the 120-word ceiling
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
