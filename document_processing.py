"""Extract text from uploaded PDF, DOCX and TXT files."""
from io import BytesIO


def extract_text(filename: str, content: bytes, enable_ocr: bool = False) -> str:
    ext = filename.rsplit('.', 1)[-1].lower()
    if ext == 'txt':
        return content.decode('utf-8-sig', errors='replace').strip()
    if ext == 'docx':
        from docx import Document
        doc = Document(BytesIO(content))
        return '\n'.join(p.text for p in doc.paragraphs if p.text.strip()).strip()
    if ext == 'pdf':
        import fitz
        with fitz.open(stream=content, filetype='pdf') as doc:
            pages = []
            for page in doc:
                text = page.get_text('text').strip()
                if not text and enable_ocr:
                    try:
                        import pytesseract
                        from PIL import Image
                        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
                        img = Image.frombytes('RGB', [pix.width, pix.height], pix.samples)
                        text = pytesseract.image_to_string(img).strip()
                    except (ImportError, pytesseract.pytesseract.TesseractNotFoundError) as exc:
                        raise RuntimeError('OCR requires pytesseract, Pillow, and the Tesseract system executable.') from exc
                pages.append(text)
            return '\n\n'.join(pages).strip()
    raise ValueError('Supported formats: PDF, DOCX, TXT')
