import streamlit as st
from urllib.parse import quote_plus
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

# Model configuration
MODEL_ID = "gemini-3.5-flash-lite"

# -----------------------------------------------------------------------------
# Jefferson Persona, Academic Level & Prompt
# -----------------------------------------------------------------------------
SYSTEM_INSTRUCTION = """
You are Thomas Jefferson: third President of the United States, primary author 
of the Declaration of Independence, polymath, and founder of the University of Virginia.

Target Audience:
American high school students reading at a 10th-grade reading level.

Tone & Demeanor:
Speak in the authentic, courteous cadence of an 18th-century statesman (e.g., "my young friend," 
"it is my considered conviction"), but keep vocabulary and phrasing clear and engaging for a 10th grader.

Knowledge Base & Archives:
Base your answers strictly on historical evidence from:
1. Founders Online (National Archives)
2. Papers of Thomas Jefferson project at Princeton University
3. Library of Congress (Thomas Jefferson Papers, 1606–1827)
4. Thomas Jefferson Foundation / Monticello Research Databases
5. Jefferson Library Digital Archive & UVA Thomas Jefferson Digital Archive
6. Massachusetts Historical Society (Coolidge Collection)
7. Swem Library Thomas Jefferson Project (College of William & Mary)
8. The Avalon Project at Yale Law School

CRITICAL RULES FOR CITATIONS & HYPERLINKS:
1. WORD LIMIT: Your spoken answer must strictly contain between 15 and 120 words.
2. DO NOT GUESS OR FABRICATE HYPERLINKS. Hallucinated file paths or fake URLs will break.
3. For primary documents and letters, use search links formatted as:
   - Founders Online: [Founders Online Record](https://founders.archives.gov/?q=KEYWORDS)
   - Library of Congress: [Library of Congress Archive](https://www.loc.gov/collections/thomas-jefferson-papers/?q=KEYWORDS)
   - Monticello Research: [Monticello Resource](https://www.monticello.org/search/?q=KEYWORDS)
   - Avalon Project: [The Avalon Project](https://avalon.law.yale.edu/)
4. For pictures, portraits, and maps, link to verified searches or live Wikimedia Commons images formatted as:
   - [View Historical Map/Image](https://commons.wikimedia.org/w/index.php?search=KEYWORDS)
   Replace KEYWORDS with relevant URL-encoded search terms (e.g., Louisiana+Purchase+Map or Jefferson+Portrait+Peale).
5. Only provide direct full page URLs if you verified them directly via Google Search tool results.
"""

# -----------------------------------------------------------------------------
# Chat Interface & State
# -----------------------------------------------------------------------------
st.title("📜 In Dialogue with Thomas Jefferson")
st.markdown("*Third President of the United States • Primary Author of the Declaration of Independence*")
st.caption("Historical dialogue for 10th-grade American History students.")

# Initialize conversation history in session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display prior chat messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "grounding_links" in msg and msg["grounding_links"]:
            with st.expander("🔗 Verified Archival & Search Sources"):
                for title, url in msg["grounding_links"]:
                    st.markdown(f"- [{title}]({url})")

# User prompt input
if user_input := st.chat_input("Inquire of Mr. Jefferson..."):
    # Append user question
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Maintain memory of only the previous six questions (up to 12 total messages)
    recent_messages = st.session_state.messages[-12:]

    # Build contents for Gemini SDK
    contents = []
    for m in recent_messages:
        role = "user" if m["role"] == "user" else "model"
        contents.append(
            types.Content(
                role=role,
                parts=[types.Part.from_text(text=m["content"])]
            )
        )

    # Generate Jefferson's response with Google Search Grounding enabled
    with st.chat_message("assistant"):
        with st.spinner("Mr. Jefferson is consulting the archives..."):
            try:
                response = client.models.generate_content(
                    model=MODEL_ID,
                    contents=contents,
                    config=types.GenerateContentConfig(
                        system_instruction=SYSTEM_INSTRUCTION,
                        temperature=0.3,
                        max_output_tokens=400,
                        # Enables real-time web search grounding to fetch valid, live URLs
                        tools=[types.Tool(google_search=types.GoogleSearch())],
                    ),
                )

                answer_text = response.text or "I beg your pardon, but I was unable to retrieve that record."
                st.markdown(answer_text)

                # Extract verified web citations from Google Search grounding metadata
                grounding_links = []
                if (
                    response.candidates 
                    and response.candidates[0].grounding_metadata 
                    and response.candidates[0].grounding_metadata.grounding_chunks
                ):
                    for chunk in response.candidates[0].grounding_metadata.grounding_chunks:
                        if chunk.web and chunk.web.uri:
                            title = chunk.web.title or chunk.web.uri
                            grounding_links.append((title, chunk.web.uri))

                # Display grounded sources if found
                if grounding_links:
                    with st.expander("🔗 Verified Archival & Search Sources"):
                        for title, url in grounding_links:
                            st.markdown(f"- [{title}]({url})")

                # Store assistant response and its links
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer_text,
                    "grounding_links": grounding_links
                })

            except Exception as e:
                st.error(f"An error occurred while generating the response: {e}")
