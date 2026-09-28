import streamlit as st
from pypdf import PdfReader
from google import genai
import chromadb
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

st.set_page_config(page_title="Chat With Your PDF")

st.title("📚 Chat With Your PDF")


# Load model only once
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


model = load_model()


# Upload PDF
uploaded_file = st.file_uploader(
    "📄 Upload your PDF",
    type=["pdf"]
)


if uploaded_file:

    st.success(f"Uploaded: {uploaded_file.name}")

    if st.button("Process PDF"):

        with st.spinner("Processing PDF..."):

            pdf = PdfReader(uploaded_file)

            splitter = RecursiveCharacterTextSplitter(
                chunk_size=800,
                chunk_overlap=100
            )

            chunks = []
            pages = []

            for page_number, page in enumerate(pdf.pages, start=1):

                text = page.extract_text() or ""

                page_chunks = splitter.split_text(text)

                for chunk in page_chunks:
                    chunks.append(chunk)
                    pages.append(page_number)

            if not chunks:
                st.error("Could not extract text from this PDF.")
                st.stop()

            embeddings = model.encode(chunks)

            client = chromadb.Client()

            collection = client.get_or_create_collection(
                "uploaded_pdf"
            )

            # Clear previous PDF
            old_ids = collection.get()["ids"]

            if old_ids:
                collection.delete(ids=old_ids)

            collection.add(
                documents=chunks,
                embeddings=embeddings.tolist(),
                ids=[str(i) for i in range(len(chunks))],
                metadatas=[{"page": page} for page in pages]
            )

            st.session_state["collection"] = collection

            st.success("✅ PDF processed! You can now ask questions.")


# Ask questions
if "collection" in st.session_state:

    question = st.text_input(
        "💬 Ask a question about your PDF:"
    )

    if st.button("Ask"):

        if not question.strip():

            st.warning("Please enter a question.")

        else:

            collection = st.session_state["collection"]

            question_embedding = model.encode([question])[0]

            results = collection.query(
                query_embeddings=[
                    question_embedding.tolist()
                ],
                n_results=5
            )

            context = "\n\n".join(
                results["documents"][0]
            )

            prompt = f"""
Answer the question using only the information provided from the PDF.

If the answer is not found in the provided context, say:

"I could not find this information in the PDF."

Give a clear and concise answer.

PDF context:
{context}

Question:
{question}

Answer:
"""

            gemini_client = genai.Client()

            response = gemini_client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt
            )

            st.subheader("🤖 Answer")

            st.write(response.text)

            st.subheader("📖 Sources")

            for i, chunk in enumerate(
                results["documents"][0]
            ):

                page = results["metadatas"][0][i]["page"]

                st.write(f"**Page {page}**")
                st.write(chunk)