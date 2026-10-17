"""Streamlit interface for a reconstructed legal document summarization demo."""
import streamlit as st
from src.document_processing import extract_text
from src.summarization import lexrank_summary, transformer_summary, word_count

st.set_page_config(page_title='AI Legal Document Summarizer', page_icon='📄', layout='wide')
st.title('📄 AI Legal Document Summarizer')
st.caption('Research prototype | Extractive NLP and optional transformer inference')
st.warning('Research/demo only. Summaries can omit or misstate legal details and are not legal advice. Do not upload confidential documents to untrusted environments.')

uploaded = st.file_uploader('Upload a legal document', type=['pdf', 'docx', 'txt'])
method = st.selectbox('Summarization method', ['LexRank (fast, extractive)', 'T5-small (transformer)', 'DistilBART CNN (transformer)'])
use_ocr = st.checkbox('Enable OCR for image-only PDF pages (requires local Tesseract)')
num_sentences = st.slider('Extractive summary sentences', 2, 15, 5)

if uploaded:
    try:
        content = uploaded.getvalue()
        if len(content) > 15 * 1024 * 1024:
            st.error('Maximum document size is 15 MB.')
            st.stop()
        with st.spinner('Extracting text...'):
            text = extract_text(uploaded.name, content, enable_ocr=use_ocr)
        if not text:
            st.error('No text found. For scanned PDFs, enable OCR and install Tesseract.')
            st.stop()
        st.metric('Extracted words', word_count(text))
        with st.expander('Preview extracted text'):
            st.text_area('Document text', text[:20000], height=220)
        if st.button('Generate summary', type='primary'):
            with st.spinner('Summarizing... first transformer run may download model weights'):
                if method.startswith('LexRank'):
                    summary = lexrank_summary(text, num_sentences)
                else:
                    model = 't5-small' if method.startswith('T5') else 'sshleifer/distilbart-cnn-12-6'
                    summary = transformer_summary(text, model)
            st.subheader('Generated summary')
            st.write(summary or 'No summary generated.')
            st.download_button('Download summary', summary, file_name='summary.txt', mime='text/plain')
    except Exception as exc:
        st.error(f'Processing failed: {exc}')
