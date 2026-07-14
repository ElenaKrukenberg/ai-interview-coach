"""Resume file handling: text extraction and safety validation. No Streamlit."""

from typing import Tuple

from pypdf import PdfReader

from security import MAX_RESUME_LENGTH, validate_user_input


def extract_resume_text(uploaded_file) -> str:
    """Extract plain text from an uploaded resume (.pdf, .txt, or .md).

    `uploaded_file` is any file-like object with `.name` and `.getvalue()`
    (Streamlit's UploadedFile in the app, a stub in tests).
    """
    if uploaded_file.name.lower().endswith(".pdf"):
        reader = PdfReader(uploaded_file)
        return "\n".join(page.extract_text() or "" for page in reader.pages).strip()
    return uploaded_file.getvalue().decode("utf-8", errors="replace").strip()


def process_resume_file(uploaded_file) -> Tuple[str, str]:
    """Extract and validate an uploaded resume.

    Returns (text, error): on success error is "", on a blocked or unsafe
    file text is "" so stale content can never leak into prompts.
    """
    text = extract_resume_text(uploaded_file)
    is_safe, error = validate_user_input(text, max_length=MAX_RESUME_LENGTH)
    if not is_safe:
        return "", error
    return text, ""
