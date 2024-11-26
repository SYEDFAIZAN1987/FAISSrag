# %% Packages
import os
import re
from dotenv import load_dotenv
from pprint import pprint
from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter, SentenceTransformersTokenTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from openai import OpenAI

# Load environment variables from .env file
load_dotenv()

# Initialize the OpenAI client
client = OpenAI(
    api_key=os.environ['OPENAI_API_KEY']  # Optional, as this is the default behavior
)

# %% Load the PDF file
project_report_file = "ALY_6080_Experential_learning_Group_1_Module_12_Capstone_Sponsor_Deliverable.pdf"
reader = PdfReader(project_report_file)
report_texts = [page.extract_text().strip() for page in reader.pages]

# %% Filter out unnecessary sections (adjust as needed)
filtered_texts = report_texts[5:-5]  # Modify indices based on the document structure

# Remove headers/footers or other unwanted text patterns
cleaned_texts = [re.sub(r'\d+\n.*?\n', '', text) for text in filtered_texts]

# %% Split text into chunks
char_splitter = RecursiveCharacterTextSplitter(
    separators=["\n\n", "\n", ". ", " ", ""],
    chunk_size=1000,
    chunk_overlap=200  # Set overlap to 200 characters (adjust as needed)
)

texts_char_splitted = char_splitter.split_text('\n\n'.join(cleaned_texts))
print(f"Number of chunks: {len(texts_char_splitted)}")

# %% Token splitting for efficient querying
token_splitter = SentenceTransformersTokenTextSplitter(
    chunk_overlap=0.2,
    tokens_per_chunk=256
)

texts_token_splitted = []
for text in texts_char_splitted:
    try:
        texts_token_splitted.extend(token_splitter.split_text(text))
    except Exception as e:
        print(f"Error in text: {text}. Error: {e}")
        continue

print(f"Number of tokenized chunks: {len(texts_token_splitted)}")

# %% Create or Load FAISS Vector Store
embeddings = OpenAIEmbeddings()
faiss_index_path = "faiss_index"

# Always rebuild the FAISS index to avoid deserialization errors
docstore = FAISS.from_texts(texts_token_splitted, embedding=embeddings)
docstore.save_local(faiss_index_path)
print("Rebuilt and saved FAISS index to local storage.")

def rag(query, n_results=5):
    """RAG function with multi-source context"""
    try:
        # Retrieve documents from the vector store
        docs = vector_store.similarity_search(query, k=n_results)
        joined_information = "; ".join([doc.page_content for doc in docs])

        # Construct the messages for the chat completion
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

       # Making a chat completion request with the new client
response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "system", "content": "You are an assistant."},
        {"role": "user", "content": "Hello, how are you?"}
    ],
    temperature=0.7
)

# Accessing the response
answer = response['choices'][0]['message']['content']

    except Exception as e:
        raise Exception(f"Error generating response: {str(e)}")


# %% Example Query
query = "What are the key findings regarding financial stability in the Greater Toronto Area?"
response = rag(query=query, n_results=5)
pprint(response)
