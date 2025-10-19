import PyPDF2
from transformers import pipeline
import streamlit as st
import io

def extract__text_from_pdf(pdf_file):
  """
  Extracts text from uploaded PDF file
  Args:
      pdf_file:Uploades PDF file object
  Returns:
      text: Extracted text as string
   """
  try:
    #Create PDF reader object
    pdf_reader = PyPDF2.PdfReader(pdf_file)

    #Intialize empty string
    text = ""

    # Loop through all pages
    for page_num in range(len(pdf_reader.pages)):
      #Extract text from each page
      page = pdf_reader.pages[page_num]
      text += page.extract_text()

    return text
  except Exception as e:

    return f"Error extracting text: {str(e)}"

# Step 4: Initialize Question Answering Model
@st.cache_resource #Cache the model to avoild reloading
def load_qa_model():
  """
  Loads pre-trained question answering model from Hugging Face
  Returns:
      qa_pipeline: Question Answering pipeline
  """

  # Using DistilBERT model trained on SQuAD dataset
  qa_pipeline = pipeline(
      "question-answering",
      model="distilbert-base-ucased-distilled-squad",
      tokenizer="distilbert-base-uncased-distilled-squad"


  )
  return qa_pipeline

# Step 5: Answer question  Function
def answer_question(question, context, qa_model):
  """

  Answers question based on context from PDF
  Args:
      question: User's question
      context: Text Extracted from PDF
      qa_model: pre-loaded QA model
  Returns:
      answer: Models answer with confidence score
  """

  try:
    # Handle long contexts(models have token limits)
    max_length = 512 * 3  # Approximate character limit
    if len(context) > max_length:
      # Take relevant chunks (simple approach)
      context = context[:max_length]


    #get answer model
    result = qa_model(question=question, context=context)

    return {
        'answer': result['answer'],
        'confidence': round(result['score'] * 100, 2)
    }

  except Exception as e:
    return {
        'answer': f"Error: {str(e)}",
        'confidence': 0
    }

# Step 6: Build Streamlit Interface

def  main():
  """
  Main function to run Streamlit app
  """

  # Page configuration
  st.set_page_config(
      page_title="PDF Question Answering Chatbot",
      page_icon="📄",
      layout="wide"
  )

  # Title and description
  st.title("📄 PDF Question Answering Chatbot")
  st.markdown("""
  Uplod a PDF document and ask question about its content.
  The AI will extract answers directly from your documnet!
  """)

  # Sidebar for PDF upload
  with st.sidebar:
    st.header("Upload PDF")
    uploaded_file = st.file_uploader(
        "Choose a PDF file",
        type=['pdf'],
        help="Upload a PDF documnet to ask questions about"
    )

    if uploaded_file:
      st.success("PDF uploaded successfully!")
      file_details = {
          "Filename": uploaded_file.name,
          "FileSize": f"{uploaded_file.size / 1024:.2f}"
      }
      st.json(file_details)

  # Main content area
  if uploaded_file is not None:
    # Extract text from pdf
    with st.spinner("Extracting text from PDF..."):
      pdf_text = extract_text_from_pdf(uploaded_file)

    # Show extracted text in expander
    with st.expander("View Extracted Text"):
      st.text_area("PDF Content", pdf_text, height=300)

    # Load QA model
    with st.spinner("Loading AI model..."):
      qa_model = load_qa_model()

    st.success("Ready to answer questions!")

    # Question input
    st.subheader("Ask a Question")
    user_question = st.text_input(
        "Enter your question:",
        placeholder="e.g., What is the main topic of this documnet?"


    )

    # Answer button
    if st.button("Get Answer", type="primary"):
      if user_question:
        with st.spinner("Thinking...."):
          result = answer_question(user_question, pdf_text, qa_model)

        # Display answer

        st.markdown("### Answer:")
        st.info(result['answer'])
        st.caption(f"Confidence: {result['confidence']}%")

      else:
        st.warning("Please enter a question!")

    # Chat history (optional enhancement)

    if 'history' not in st.session_state:
      st.session_state.history = []

    if user_question and st.session_state.get('last_question') != user_question:
      result = answer_question(user_question, pdf_text, qa_model) # Ensure result is available
      st.session_state.history.append({
          'question': user_question,
          'answer': result['answer'] if 'result' in locals() else None
      })
      st.session_state.last_question = user_question # Corrected typo here

    # Show chat history

    if st.session_state.history:
      st.markdown("---")
      st.subheader("Chat History")
      for i, qa in enumerate(reversed(st.session_state.history[-5:])): # Corrected typo here
        if qa['answer']:
          st.markdown(f"**Q{len(st.session_state.history)-i}:** {qa['question']}")
          st.markdown(f"**A:** {qa['answer']}")
          st.markdown("")
  else:
  # Instructions when no file uploaded

    st.info(" Please upload a PDF file from the sidebar to get started!")

    st.markdown(
        """
        ###How to use:
        1. Upload a PDF document using the sidebar
        2. Wait for text extraction
        3. Type your question about the document
        4. Click 'Get Answer' to get AI-powered responses
        """
        
if __name__ == "__main__":
    main()
       
