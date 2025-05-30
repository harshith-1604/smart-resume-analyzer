import streamlit as st
import pickle 
import re
import nltk
import PyPDF2
import docx
import spacy

nltk.download('punkt')
nltk.download('stopwords')

# Loading models
clf=pickle.load(open('clf.pkl','rb'))
tfidfd=pickle.load(open('tfidf.pkl','rb'))
le=pickle.load(open('labelencoder.pkl','rb'))
nlp=spacy.load("en_core_web_sm")

# Skill list
Skills=[
    "python","java","c++","sql","excel","machine learning","deep learning","nlp","pandas","numpy",
    "scikit-learn","tensorflow","keras","django","flask","aws","azure","git","docker","redis",
    "linux","javascript","html","css","react","node.js","mongodb","spark","hadoop","tableau",
    "accounting","financial analysis","budgeting","forecasting","auditing","taxation","payroll",
    "investment","risk management","financial modeling","sap","quickbooks","erp","cost accounting",
    "banking","capital markets","equity research","valuation","treasury","compliance",
    "recruitment","talent acquisition","onboarding","employee relations","performance management",
    "training","hr policies","payroll management","benefits administration","hr analytics",
    "labor law","conflict resolution","succession planning","organizational development",
    "autocad","solidworks","catia","ansys","mechanical design","manufacturing","maintenance",
    "thermodynamics","fluid mechanics","hvac","mechatronics","cad","cam","fea","plc",
    "structural analysis","staad pro","etabs","autocad civil","construction management",
    "surveying","quantity estimation","project management","geotechnical engineering",
    "transportation engineering","primavera","ms project","site supervision","bim",
    "communication","leadership","teamwork","problem solving","project planning","presentation",
    "negotiation","customer service","sales","marketing","data entry","supply chain","logistics"
]

# Cleaning function
def cleanResume(txt):
    cleanttext=re.sub(r'http\S+',' ',txt)
    cleanttext=re.sub(r'\bRT\b|\bcc\b',' ',cleanttext)
    cleanttext=re.sub(r'#\S+', ' ', cleanttext)
    cleanttext=re.sub(r'@\S+',' ',cleanttext)
    cleanttext=re.sub(r'[!"#$%&\'()*+,-./:;<=>?@\[\]^_`{|}~]',' ',cleanttext)
    cleanttext=re.sub(r'[^\x00-\x7f]',' ',cleanttext)
    cleanttext=re.sub(r'\s+',' ',cleanttext)
    return cleanttext.strip()

# Skill extraction
def extractskills(text):
    text=text.lower()
    skills=set()
    for skill in Skills:
        if skill in text:
            skills.add(skill)
    return skills

# Skill highlight
def highlightkeys(text,keywords):
    for kw in keywords:
        text=text.replace(kw,f"<span style='background-color:#E0F7FA;color:#00796B;font-weight:bold'>{kw}</span>")
    return text

# Extract text from file
def extract(fileupload):
    if fileupload.type=="application/pdf":
        reader=PyPDF2.PdfReader(fileupload)
        text=""
        for page in reader.pages:
            text+=page.extract_text() or ""
        return text
    elif fileupload.type in ["application/vnd.openxmlformats-officedocument.wordprocessingml.document","application/msword"]:
        doc=docx.Document(fileupload)
        return "\n".join([para.text for para in doc.paragraphs])
    else:
        try:
            return fileupload.read().decode('utf-8')
        except UnicodeDecodeError:
            fileupload.seek(0)
            return fileupload.read().decode('latin-1')
# Web App
def main():
    st.set_page_config(page_title="Resume Screening App", layout="wide")
    st.markdown(
        """
        <style>
        .main {
            background-color: #f9f9f9;
            padding: 2rem;
            font-family: 'Segoe UI', sans-serif;
        }
        h1 {
            color: #00796B;
        }
        .stTextArea, .stFileUploader {
            margin-top: 10px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    st.title("📄 Resume Screening App")
    st.markdown("Upload a resume and compare it with a job description to check skill match and predict resume category.")
    st.markdown("---")
    col1,col2=st.columns(2)
    with col1:
        st.subheader("📝 Job Description")
        jdtext=st.text_area("Paste the Job Description here...", height=250)
    with col2:
        st.subheader("📁 Upload Resume")
        fileupload=st.file_uploader("Supported formats:PDF,DOCX,TXT",type=["txt","pdf","docx"])
    st.markdown("---")
    if fileupload is not None and jdtext.strip():
        resumetxt=extract(fileupload)
        cleaned_resume=cleanResume(resumetxt)
        input_features=tfidfd.transform([cleaned_resume])
        prediction_id=clf.predict(input_features)[0]
        category_name=le.inverse_transform([prediction_id])[0]
        st.success(f"✅ **Predicted Resume Category:** {category_name}")
        # Skill Matching
        jd_skills=extractskills(jdtext)
        resume_skills=extractskills(resumetxt)
        matched_skills=jd_skills&resume_skills
        missing_skills=jd_skills-resume_skills
        match_percent=(len(matched_skills)/len(jd_skills)*100) if jd_skills else 0
        st.markdown(f"### 🔍 Skill Match: `{len(matched_skills)}/{len(jd_skills)}`({match_percent:.0f}%)")
        if matched_skills:
            st.markdown("✅ **Matched Skills:** "+", ".join([f"`{s}`" for s in matched_skills]))
        if missing_skills:
            st.markdown("⚠️ **Missing Skills:** "+", ".join([f"`{s}`" for s in missing_skills]))
        st.markdown("#### 📌 Job Description with Highlighted Skills")
        st.markdown(highlightkeys(jdtext.lower(), matched_skills),unsafe_allow_html=True)
        st.markdown("#### 📌 Resume with Highlighted Skills")
        st.markdown(highlightkeys(resumetxt.lower(), matched_skills),unsafe_allow_html=True)
    elif fileupload is None and jdtext.strip():
        st.warning("⚠️ Please upload a resume to proceed.")
    elif fileupload is not None and not jdtext.strip():
        st.warning("⚠️ Please paste a job description to proceed.")

if __name__=='__main__':
    main()