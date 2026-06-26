import io
import re
from typing import List, Optional

import PyPDF2

from app.models.schemas import ResumeInfo

COMMON_SKILLS = {
    "python", "java", "javascript", "typescript", "c++", "c#", "go", "rust", "ruby", "php",
    "sql", "postgresql", "mysql", "mongodb", "redis", "elasticsearch", "cassandra",
    "aws", "azure", "gcp", "docker", "kubernetes", "terraform", "jenkins", "github actions",
    "react", "vue", "angular", "next.js", "node.js", "django", "flask", "fastapi", "spring",
    "tensorflow", "pytorch", "keras", "scikit-learn", "pandas", "numpy", "matplotlib",
    "machine learning", "deep learning", "nlp", "computer vision", "reinforcement learning",
    "data structures", "algorithms", "system design", "microservices", "rest api", "graphql",
    "kafka", "rabbitmq", "celery", "airflow", "spark", "hadoop", "hive", "databricks",
    "linux", "git", "agile", "scrum", "ci/cd", "devops", "mlops", "data engineering"
}

COMMON_DOMAINS = {
    "backend", "frontend", "fullstack", "ai", "machine learning", "data science",
    "devops", "cloud", "mobile", "web", "distributed systems", "security", "blockchain"
}

class ResumeParser:
    def extract_text_from_pdf(self, file_bytes: bytes) -> str:
        try:
            pdf_reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() or ""
            return text
        except Exception as e:
            raise ValueError(f"Failed to parse PDF: {str(e)}")
    
    def extract_skills(self, text: str) -> List[str]:
        text_lower = text.lower()
        found_skills = []
        for skill in COMMON_SKILLS:
            pattern = r'\b' + re.escape(skill) + r'\b'
            if re.search(pattern, text_lower):
                found_skills.append(skill)
        return list(set(found_skills))
    
    def extract_technologies(self, text: str, skills: List[str]) -> List[str]:
        tech_keywords = {
            "react", "angular", "vue", "django", "flask", "fastapi", "spring boot",
            "tensorflow", "pytorch", "keras", "scikit-learn", "pandas", "numpy",
            "docker", "kubernetes", "aws", "azure", "gcp", "terraform", "ansible",
            "mongodb", "postgresql", "mysql", "redis", "elasticsearch", "kafka",
            "spark", "hadoop", "airflow", "mlflow", "wandb", "jupyter", "colab"
        }
        text_lower = text.lower()
        found_techs = [tech for tech in tech_keywords if tech in text_lower]
        return list(set(found_techs + skills))
    
    def extract_domains(self, text: str) -> List[str]:
        text_lower = text.lower()
        found_domains = [domain for domain in COMMON_DOMAINS if domain in text_lower]
        return list(set(found_domains))
    
    def extract_experience_years(self, text: str) -> Optional[float]:
        patterns = [
            r'(\d+)\+?\s*years?\s*(?:of\s*)?experience',
            r'experience\s*[:\-]?\s*(\d+)\+?\s*years?',
            r'(\d+)\+?\s*yrs?\s*(?:of\s*)?exp',
        ]
        for pattern in patterns:
            match = re.search(pattern, text.lower())
            if match:
                return float(match.group(1))
        return None
    
    def extract_education(self, text: str) -> List[str]:
        edu_keywords = ["bachelor", "master", "phd", "doctorate", "b.tech", "m.tech", "b.e.", "m.e.", "b.s.", "m.s."]
        text_lower = text.lower()
        education = []
        for keyword in edu_keywords:
            if keyword in text_lower:
                education.append(keyword)
        return education
    
    def parse(self, file_bytes: bytes, filename: str) -> tuple[str, ResumeInfo]:
        if filename.lower().endswith('.pdf'):
            text = self.extract_text_from_pdf(file_bytes)
        else:
            text = file_bytes.decode('utf-8', errors='ignore')
        return self.parse_text(text)

    def parse_text(self, text: str) -> tuple[str, ResumeInfo]:
        skills = self.extract_skills(text)
        technologies = self.extract_technologies(text, skills)
        domains = self.extract_domains(text)
        experience = self.extract_experience_years(text)
        education = self.extract_education(text)
        
        resume_info = ResumeInfo(
            skills=skills,
            technologies=technologies,
            domains=domains,
            experience_years=experience,
            education=education
        )

        return text, resume_info

resume_parser_service = ResumeParser()
