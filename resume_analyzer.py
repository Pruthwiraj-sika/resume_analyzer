from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.models.google import Gemini
from pypdf import PdfReader
from agno.models.ollama import Ollama
from flask import Flask,request,render_template
from dotenv import load_dotenv
load_dotenv()

app = Flask(__name__)
def build_agent():
    return Agent(
        model=Gemini(id="gemini-3.6-flash"),
        debug_mode=True,
        instructions=[
    # 1. ROLE
    "You are a strict but fair resume reviewer and ATS specialist.",
    "Analyze ONLY the text provided. NEVER invent skills, jobs, degrees, companies or numbers.",
 
    # 2. MODE
    "The input has a RESUME and sometimes a JOB DESCRIPTION (JD).",
    "If a JD is present: judge the resume against that JD. Every weak point must be linked to a specific JD requirement.",
    "If no JD is present: judge the resume for general quality and say so in one line.",
 
    # 3. SCORING RUBRIC (total 100, be consistent, do not inflate)
    "Score out of 100 using this rubric and show each part:",
    "  - Skills & keyword match (JD if given): 30",
    "  - Work experience & measurable impact: 25",
    "  - Projects & achievements: 15",
    "  - Education & certifications: 10",
    "  - ATS format & structure (clear headings, no clutter): 10",
    "  - Clarity, grammar & conciseness: 10",
    "Overall score = sum of the parts. Below 50 = weak, 50-69 = average, 70-84 = good, 85+ = excellent.",
 
    # 4. OUTPUT FORMAT
    "Reply in Markdown with exactly these sections:",
    "# 📄 Resume Report",
    "## 🎯 Overall Score: X/100 (rating label + one line reason)",
    "Then a table: Category | Score | Max.",
    "## 🧩 JD Match: X% (ONLY if a JD was given) - list Matched skills and Missing skills.",
    "## ✅ Strengths (3-5 bullets, specific)",
    "## ⚠️ Weak Points (3-6 items). For each: **Problem** -> **Why it hurts (cite the JD requirement if JD given)** -> **Fix**.",
    "## 🔑 Missing Keywords (terms from the JD or role that should be added)",
    "## ✍️ Bullet Rewrites (3 weak bullets, Before -> After, add measurable impact only if the resume supports it)",
    "## 🚀 Top 5 Action Items (prioritised, most impactful first)",
 
    # 5. TONE
    "Be honest, specific and concise. No flattery, no generic advice.",
],
markdown=True,
add_datetime_to_context=True
        )

def extract_pdf_text(file,description="")->str:
    """file=PDF ka path (string) ya Streamlit ka uploaded file"""
    reader = PdfReader(file)
    text= ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text+=page_text+"\n"
    text  = text.strip()
    if len(text)<100:
        raise ValueError("PDF se text nahi nikla. Text-based PDF use karo.")
    prompt = f"RESUME:\n{text}\n\n JOB DESCRIPTION:\n{description}"
    agent = build_agent()
    return agent.run(prompt).content


@app.route("/", methods=["GET","POST"])
def analyze_resume():
    report = None
    error = None
    if request.method == "POST":
        file = request.files["resume"]
        description = request.form.get("description", "")
        try:
            report = extract_pdf_text(file=file.stream,description=description)
        except Exception as e:
            error = str(e)

    return render_template("index.html",report=report,error=error)
if __name__== "__main__":
    app.run(debug=True)
