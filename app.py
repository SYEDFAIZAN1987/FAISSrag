import streamlit as st
import os
from dotenv import load_dotenv
from pprint import pprint
from langchain_community.vectorstores import FAISS
from langchain.docstore import InMemoryDocstore
from langchain.embeddings import OpenAIEmbeddings

import openai
from datetime import datetime


# Page configuration
st.set_page_config(
    page_title="ALY 6080 Group 1 Report Chat Bot",
    page_icon="📘",
    layout="wide"
)

# Load environment variables from .env file
load_dotenv()

# Custom CSS for better styling
st.markdown("""
    <style>
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    .chat-message {
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        display: flex;
        flex-direction: column;
        max-width: 800px;
        margin-left: auto;
        margin-right: auto;
    }
    .user-message {
        background-color: #f0f2f6;
        border-left: 4px solid #002D62;
    }
    .assistant-message {
        background-color: #e8f4f9;
        border-left: 4px solid #c41e3a;
    }
    .source-box {
        background-color: #f8f9fa;
        border: 1px solid #002D62;
        border-radius: 0.25rem;
        padding: 0.75rem;
        margin-top: 0.5rem;
        font-size: 0.9em;
    }
    .header-container {
        padding: 1rem;
        background-color: white;
        color: #002D62;
        margin-bottom: 2rem;
        border-bottom: 3px solid #002D62;
    }
    .logo-container {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 1rem 0;
    }
    .logo-item {
        text-align: center;
        flex: 1;
    }
    .logo-item img {
        max-height: 60px;
        object-fit: contain;
    }
    .header-text {
        color: #002D62;
        margin: 0;
        text-align: center;
    }
    .source-tag {
        background-color: #002D62;
        color: white;
        padding: 0.2rem 0.5rem;
        border-radius: 0.25rem;
        font-size: 0.8em;
        margin-right: 0.5rem;
    }
    .disclaimer {
        font-size: 0.8em;
        color: #666;
        text-align: center;
        padding: 1rem;
        border-top: 1px solid #eee;
    }
    .divider {
        height: 2px;
        background-color: #002D62;
        margin: 1rem 0;
    }
    .chat-container {
        max-width: 800px;
        margin: 2rem auto;
        padding: 0 1rem;
    }
    .example-button {
        margin: 0.25rem;
    }
    .chat-input {
        max-width: 800px;
        margin: 1rem auto;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'current_question' not in st.session_state:
    st.session_state.current_question = ''

# Function to handle example question clicks
def set_example_question(question):
    st.session_state.current_question = question

# Header with logos and branding
st.markdown("""
    <div class="header-container">
        <div class="divider"></div>
        <h1 class="header-text">ALY 6080 Group 1 Capstone Project</h1>
        <p class="header-text" style="font-size: 1.5em; font-weight: bold; color: red;">
            Built by Syed Faizan, Team Lead of Group 1, ALY 6080, Northeastern University
        </p>
    </div>
""", unsafe_allow_html=True)


# Display the logo using st.image
col1, col2, col3 = st.columns([1, 2, 1])

with col1:
    st.write("")  # Empty column for spacing

with col2:
    st.image("images/univ.png", caption="University Logo", use_column_width=True)

with col3:
    st.write("")  # Empty column for spacing

col1, col2, col3, col4, col5, col6 = st.columns(6)

with col1:
    st.image("images/faizan.jpeg", use_column_width=True, caption="Team Lead: Syed Faizan")

with col2:
    st.image("images/Christiana.jpeg", use_column_width=True, caption="Christiana")

with col3:
    st.image("images/Pravalika.jpeg", use_column_width=True, caption="Pravalika")

with col4:
    st.image("images/VrajShah.jpeg", use_column_width=True, caption="Vraj Shah")

with col5:
    st.image("images/Emelia.jpeg", use_column_width=True, caption="Emelia Doku")

with col6:
    st.image("images/Schicheng.jpeg", use_column_width=True, caption="Shicheng Wan")


# Introduction text in chat container
with st.container():
    st.markdown("""
        This knowledge assistant is designed to answer questions related to the ALY 6080 Capstone Project, covering topics such as:
        - Housing Stability
        - Financial Trends
        - Demographics and Socio-Economic Data
        - Predictions from Machine Learning Models
        - Insights from Exploratory Data Analysis (EDA)
    """)

# OpenAI API key check
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    st.error("⚠️ OpenAI API key not found! Please set the OPENAI_API_KEY environment variable.")
    st.stop()

openai.api_key = openai_api_key

# Initialize FAISS vector store with dangerous deserialization
try:
    embeddings = OpenAIEmbeddings()
    vector_store = FAISS.load_local("db", embeddings, allow_dangerous_deserialization=True)
    # st.success("FAISS vector store loaded successfully!")
    # Commented out the warning to suppress it
    # st.warning(
    #     "⚠️ Dangerous deserialization is enabled. Ensure the FAISS index (`db`) is from a trusted source "
    #     "to avoid potential security risks."
    # )
except Exception as e:
    st.error(f"⚠️ Error connecting to the database: {str(e)}")
    st.stop()


def rag(query, n_results=5):
    """RAG function with multi-source context"""
    try:
        docs = vector_store.similarity_search(query, k=n_results)
        joined_information = '; '.join([doc.page_content for doc in docs])
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a knowledgeable assistant with expertise in socio-economic data, housing stability, "
                    "financial trends, and predictive analytics. Provide clear, accurate answers based on provided "
                    "information and your ALY 6080 Group 1 Project context."
                )
            },
            {"role": "user", "content": f"Question: {query}\nInformation: {joined_information}"}
        ]
        
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=messages,
            temperature=0.7
        )
        
        return response.choices[0].message['content'], docs
    except Exception as e:
        raise Exception(f"Error generating response: {str(e)}")

# Chat interface in container
chat_container = st.container()

with chat_container:
    # Display chat history
    for message in st.session_state.chat_history:
        if message["role"] == "user":
            st.markdown(f"<div class='chat-message user-message'>💭 You: {message['content']}</div>", 
                       unsafe_allow_html=True)
        else:
            st.markdown(f"""
                <div class='chat-message assistant-message'>
                    <div style='margin-bottom: 0.5rem;'>📘 Assistant: {message['response']}</div>
                </div>
            """, unsafe_allow_html=True)

# If there's a current question in the session state, use it as the default value
user_query = st.text_input(
    label="Ask your question about the ALY 6080 project",
    help="Type your question or click an example below",
    placeholder="Example: What are the key trends in housing stability?",
    value=st.session_state.current_question,
    key="unique_user_input_key"  # Assign a unique key
)

# Query input with examples
st.markdown("<div class='chat-input'>", unsafe_allow_html=True)


# Example questions as buttons
example_questions = [
    "What are the key findings about housing stability?",
    "What are the income trends for Toronto CMA?",
    "How does the living wage prediction look for 2030?",
    "What are the demographic trends observed in GTA?"
]

st.markdown("### 💡 Example Questions")
cols = st.columns(2)
for idx, question in enumerate(example_questions):
    # Assign a unique key for each button
    if cols[idx % 2].button(question, key=f"example_question_key_{idx}"):
        st.session_state.current_question = question  # Set the selected example question

# Single "Ask" Button Logic
if st.button("Ask", type="primary", use_container_width=True):  # Ensure only one button
    if not user_query:
        st.warning("⚠️ Please enter a question!")
    else:
        # Append user's query to chat history
        st.session_state.chat_history.append({"role": "user", "content": user_query})
        
        with st.spinner("🔍 Analyzing sources..."):
            try:
                # Call the RAG function and get response and sources
                response, sources = rag(user_query)
                
                # Append assistant's response to chat history
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "response": response,
                    "sources": sources
                })
                
                # Clear the current question after processing
                st.session_state.current_question = ''
                
                # Update app state to refresh UI
                st.session_state['force_rerun'] = True  # Trigger UI update
            except Exception as e:
                # Log and display a detailed error message
                st.error(f"⚠️ An error occurred while processing your query: {str(e)}")

# Close the chat input container
st.markdown("</div>", unsafe_allow_html=True)



# Sidebar with information
with st.sidebar:
    st.image("images/uwgt.png", caption="Sponsor Logo", width=150)

    if st.button("Clear Chat History"):
        st.session_state.chat_history = []
        st.rerun()
    
    st.markdown("""
    ### About This Assistant
    
    This knowledge assistant Chat Bot based on RAG is designed by Team Lead Syed Faizan to answer questions related to the ALY 6080 Capstone Project by Group 1, Northeastern University.
    
    **Project Highlights:**
    - **Housing Stability:** Trends in eviction applications, housing starts, and affordability.
    - **Financial Stability:** Income distribution, Gini coefficient trends, and living wage predictions.
    - **Demographics:** Analysis of population growth, diversity, and socio-economic factors in the Greater Toronto Area (GTA).
    - **Machine Learning Models:** Predictive analysis of low-income measures and unemployment rates.
    - **Actionable Insights:** Recommendations for stakeholders based on analytical findings.

    #### 💡 Tips for Better Results
    - Be specific in your questions.
    - Focus on project-related topics.
    - Explore housing, financial, and demographic trends in detail.
    """)

# Enhanced footer with credits
st.markdown("---")
st.markdown("""
<div style="padding: 10px; font-size: 14px; color: #444;">
    <p><strong>Team Members:</strong> Syed Faizan (Team Lead), Emelia Doku, Vraj Shah, Shicheng Wan, Pravalika Sorda, and Christiana Adjei.</p>
    <p><strong>Data Sources:</strong> Public datasets, machine learning models, and EDA outputs from the ALY 6080 Capstone Project.</p>
    <p>For additional resources or inquiries, please contact the Team Lead, Syed Faizan, via the contact section.</p>
</div>
""", unsafe_allow_html=True)

# Path to the project report file
project_report_file = "ALY_6080_Experential_learning_Group_1_Module_12_Capstone_Sponsor_Deliverable.pdf"

# Section Header
st.markdown("""
<div class="section">
    <h3 style="color: #002D62; font-weight: bold;">About Our ALY 6080 Group 1 Project Deliverables</h3>
</div>
""", unsafe_allow_html=True)

# Download Button for the Capstone Project Report
with open(project_report_file, "rb") as file:
    st.download_button(
        label="📄 Click here to download our ALY 6080 Capstone Project Report",
        data=file,
        file_name="ALY_6080_Experential_learning_Group_1_Module_12_Capstone_Sponsor_Deliverable.pdf",
        mime="application/pdf"
    )

# Project Highlights
st.write("""
- **Deliverables:** Our deliverables include:
    - Demographic Analysis
    - Power BI Dashboards
    - Jupyter Notebook and R files
    - GitHub Repository
    - Detailed Report
    - This Interactive App
- **Project Focus:** Our project emphasizes:
    - Demographics, Housing Stability, and Financial Stability in the Toronto GTA.
    - The interaction between these three critical domains.
- **Technologies Used:**
    - **Machine Learning Predictive Algorithms**
    - **Python** and **R** for Exploratory Data Analysis (EDA)
    - **LaTeX** for Report Formulation
    - **GitHub** and **OpenAI API**
    - **HTML**, **CSS**, and **Streamlit** for App Development
""")


# Contact Me section
st.markdown('<div class="section"><h3>Contact the Team Lead Syed Faizan</h3>', unsafe_allow_html=True)
col1, col2, col3 = st.columns(3)

with col2:
     st.markdown('<div class="button"><a href="https://www.linkedin.com/in/drsyedfaizanmd/">LinkedIn</a></div>', unsafe_allow_html=True)
with col1:
     st.markdown('<a href="mailto:faizan.s@northeastern.edu">Email Syed Faizan</a>', unsafe_allow_html=True)
with col2:
     st.markdown('<div class="button"><a href="https://twitter.com/faizan_data_ml">Twitter</a></div>', unsafe_allow_html=True)
with col3:
     st.markdown('<div class="button"><a href="https://github.com/SYEDFAIZAN1987">GitHub</a></div>', unsafe_allow_html=True)
     st.markdown('</div>', unsafe_allow_html=True)

  
