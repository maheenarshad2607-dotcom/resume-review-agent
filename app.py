import streamlit as st
import os
from io import BytesIO
from pypdf import PdfReader

st.set_page_config(
    page_title="Resume Review Agent",
    page_icon="📄",
    layout="wide",
)

st.title("📄 Resume Review Agent")
st.caption("Powered by CrewAI + Groq (llama-3.3-70b-versatile)")

def get_groq_key():
    try:
        return st.secrets["GROQ_API_KEY"]
    except (KeyError, FileNotFoundError):
        return None

GROQ_API_KEY = get_groq_key()

if not GROQ_API_KEY:
    st.error(
        "⚠️ **GROQ_API_KEY not found in Streamlit secrets.**\n\n"
        "Add it in your Streamlit Cloud dashboard under **Settings → Secrets**:\n\n"
        "```toml\nGROQ_API_KEY = \"gsk_...\"\n```"
    )
    st.stop()

os.environ["GROQ_API_KEY"] = GROQ_API_KEY

try:
    from crewai import Agent, Task, Crew, LLM
except Exception as e:
    st.error(f"Failed to import CrewAI: {e}")
    st.stop()

def extract_text_from_pdf(uploaded_file) -> str:
    try:
        reader = PdfReader(BytesIO(uploaded_file.read()))
        text_parts = []
        for page in reader.pages:
            page_text = page.extract_text() or ""
            text_parts.append(page_text)
        return "\n".join(text_parts).strip()
    except Exception as e:
        st.warning(f"Could not read PDF: {e}")
        return ""

def run_resume_review(resume_text: str, job_description: str) -> str:
    try:
        llm = LLM(
            model="groq/llama-3.3-70b-versatile",
            temperature=0.3,
            api_key=GROQ_API_KEY,
        )
    except Exception as e:
        return f"❌ Failed to initialize LLM: {e}"

    reviewer = Agent(
        role="Senior Resume Reviewer & Career Coach",
        goal=(
            "Compare a candidate's resume against a target job description "
            "and produce honest, structured, actionable feedback."
        ),
        backstory=(
            "You have 15+ years of experience as a technical recruiter and "
            "career coach. You are strict about NOT inventing qualifications "
            "the candidate does not have. You always ground your feedback "
            "in the exact text of the resume and job description."
        ),
        llm=llm,
        verbose=False,
        allow_delegation=False,
    )

    task_description = f"""
You are reviewing a candidate's resume against a target job description.

=== RESUME ===
{resume_text}

=== JOB DESCRIPTION ===
{job_description}

=== YOUR TASK ===
Produce a structured report with EXACTLY these sections:

1. **Match Score (0–100)** — One number, plus one sentence of justification.
2. **Top Strengths (3–5 bullets)** — What already matches well. Quote from the resume.
3. **Critical Gaps (3–5 bullets)** — Requirements in the JD that are NOT evidenced in the resume.
4. **Missing Keywords / Skills** — Important keywords from the JD missing in the resume.
5. **Actionable Improvement Suggestions (5–7 bullets)** — Specific rewrites and additions.
6. **Honesty Check** — Remind the candidate NOT to claim skills they don't have.

RULES:
- Do NOT fabricate experience, skills, or credentials.
- If the resume does not mention something, say "not evidenced in resume."
- Be specific and quote resume lines.
- Keep the tone professional, supportive, and direct.
"""

    review_task = Task(
        description=task_description,
        expected_output="A well-formatted markdown report with all 6 sections.",
        agent=reviewer,
    )

    crew = Crew(agents=[reviewer], tasks=[review_task], verbose=False)

    try:
        result = crew.kickoff()
        return str(result)
    except Exception as e:
        err = str(e).lower()
        if "rate" in err or "429" in err:
            return "⏳ **Groq rate limit hit.** Please wait ~30 seconds and try again."
        if "api" in err or "auth" in err or "401" in err:
            return "🔑 **Groq API error.** Please verify your GROQ_API_KEY."
        return f"❌ Unexpected error during review: {e}"

col1, col2 = st.columns(2)

with col1:
    st.subheader("1️⃣ Your Resume")
    input_method = st.radio(
        "How would you like to provide your resume?",
        ("Paste text", "Upload PDF"),
        horizontal=True,
    )

    resume_text = ""

    if input_method == "Paste text":
        resume_text = st.text_area(
            "Paste your resume here:",
            height=350,
            placeholder="Paste the full text of your resume...",
        )
    else:
        uploaded_pdf = st.file_uploader("Upload your resume (PDF):", type=["pdf"])
        if uploaded_pdf is not None:
            with st.spinner("Extracting text from PDF..."):
                resume_text = extract_text_from_pdf(uploaded_pdf)
            if resume_text:
                st.success(f"✅ Extracted {len(resume_text)} characters.")
                with st.expander("Preview extracted text"):
                    st.text(resume_text[:2000] + ("..." if len(resume_text) > 2000 else ""))
            else:
                st.error("Could not extract text. Try a different PDF.")

with col2:
    st.subheader("2️⃣ Target Job Description")
    job_description = st.text_area(
        "Paste the job description here:",
        height=350,
        placeholder="Paste the full job posting...",
    )

st.divider()

run_clicked = st.button("🚀 Review My Resume", type="primary", use_container_width=True)

if run_clicked:
    if not resume_text or len(resume_text.strip()) < 50:
        st.warning("⚠️ Please provide a resume with at least 50 characters.")
    elif not job_description or len(job_description.strip()) < 50:
        st.warning("⚠️ Please provide a job description with at least 50 characters.")
    else:
        with st.spinner("🤖 The agent is analyzing your resume..."):
            result = run_resume_review(resume_text.strip(), job_description.strip())

        st.divider()
        st.subheader("📊 Review Report")

        if result.startswith(("❌", "⏳", "🔑")):
            st.error(result)
        else:
            st.markdown(result)
            st.download_button(
                "⬇️ Download Report (.md)",
                data=result,
                file_name="resume_review.md",
                mime="text/markdown",
  )
