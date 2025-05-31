import streamlit as st
from docxtpl import DocxTemplate
from io import BytesIO
import os
import tempfile
from docx2pdf import convert

TEMPLATE_PATH = os.path.join("templates", "resume_template.docx")

questions = [
    "Full Name",
    "Email Address",
    "Phone Number",
    "LinkedIn Profile URL",
    "Portfolio or Website",
    "Professional Summary",
    "Work Experience",
    "Key Projects",
    "Key Skills (comma-separated)",
    "Education Background",
    "Certifications and Awards",
    "Desired Job Position",
    "Hobbies and Interests",
    "References (if any)"
]

def render_form():
    with st.form("resume-form"):
        col1, col2 = st.columns(2)
        responses = []
        for i, q in enumerate(questions):
            if i < 7:
                response = col1.text_area(q) if i >= 5 else col1.text_input(q)
            else:
                response = col2.text_area(q) if i >= 5 else col2.text_input(q)
            responses.append(response)
        submitted = st.form_submit_button("📄 Generate Resume")
    return submitted, responses

def generate_pdf_resume(responses):
    doc = DocxTemplate(TEMPLATE_PATH)
    context = {
        'full_name': responses[0],
        'email': responses[1],
        'phone': responses[2],
        'linkedin': responses[3],
        'portfolio': responses[4],
        'professional_summary': responses[5],
        'work_experience': responses[6],
        'key_projects': responses[7],
        'key_skills': responses[8],
        'education_background': responses[9],
        'certifications_awards': responses[10],
        'desired_job_position': responses[11],
        'hobbies_interests': responses[12],
        'references': responses[13]
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        docx_path = os.path.join(tmpdir, "resume.docx")
        pdf_path = os.path.join(tmpdir, "resume.pdf")

        doc.render(context)
        doc.save(docx_path)

        convert(docx_path, pdf_path)

        with open(pdf_path, "rb") as f:
            pdf_bytes = f.read()

    return pdf_bytes

def show_resume_builder():
    st.title("📝 AI Resume Builder")
    st.write("Fill in the following details to generate a polished, professional resume:")

    submitted, responses = render_form()
    if submitted:
        st.success("🎉 Resume generated successfully!")
        buffer = generate_pdf_resume(responses)
        st.download_button(
            label="📥 Download Resume (PDF)",
            data=buffer,
            file_name="resume.pdf",
            mime="application/pdf"
        )
