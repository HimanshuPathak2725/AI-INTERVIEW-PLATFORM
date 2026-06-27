import io
import re
from typing import List, Optional, Set, Tuple

import PyPDF2

from app.models.schemas import ResumeInfo

MAX_RESUME_SIZE_BYTES = 10 * 1024 * 1024  # 10 MB

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

TECH_KEYWORDS = {
    "react", "angular", "vue", "django", "flask", "fastapi", "spring boot",
    "tensorflow", "pytorch", "keras", "scikit-learn", "pandas", "numpy",
    "docker", "kubernetes", "aws", "azure", "gcp", "terraform", "ansible",
    "mongodb", "postgresql", "mysql", "redis", "elasticsearch", "kafka",
    "spark", "hadoop", "airflow", "mlflow", "wandb", "jupyter", "colab"
}

COMMON_DOMAINS = {
    "backend", "frontend", "fullstack", "ai", "machine learning", "data science",
    "devops", "cloud", "mobile", "web", "distributed systems", "security", "blockchain"
}

EDU_KEYWORDS = {
    "bachelor", "master", "phd", "doctorate", "b.tech", "m.tech", "b.e.", "m.e.", "b.s.", "m.s."
}

# Pre-compile regex with word boundaries to eliminate false positives
def _compile_patterns(keywords: Set[str]) -> dict[str, re.Pattern]:
    return {
        kw: re.compile(rf"\b{re.escape(kw)}\b", re.IGNORECASE)
        for kw in keywords
    }

SKILL_PATTERNS  = _compile_patterns(COMMON_SKILLS)
TECH_PATTERNS   = _compile_patterns(TECH_KEYWORDS)
DOMAIN_PATTERNS = _compile_patterns(COMMON_DOMAINS)
EDU_PATTERNS    = _compile_patterns(EDU_KEYWORDS)


class ResumeParser:
    def extract_text_from_pdf(self, file_bytes: bytes) -> str:
        if len(file_bytes) > MAX_RESUME_SIZE_BYTES:
            raise ValueError(f"Resume size exceeds maximum limit of {MAX_RESUME_SIZE_BYTES / (1024*1024)}MB")
            
        try:
            pdf_reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
            text_parts = []
            for page in pdf_reader.pages:
                extracted = page.extract_text()
                if extracted:
                    text_parts.append(extracted)
            
            # Use double newline to prevent words merging across page breaks
            return "\n\n".join(text_parts)
        except Exception as e:
            raise ValueError(f"Failed to parse PDF: {str(e)}")
    
    def extract_skills(self, text: str) -> List[str]:
        found_skills = [skill for skill, pat in SKILL_PATTERNS.items() if pat.search(text)]
        return sorted(set(found_skills))
    
    def extract_technologies(self, text: str) -> List[str]:
        # Semantic separation maintained: we no longer blindly append skills here
        found_techs = [tech for tech, pat in TECH_PATTERNS.items() if pat.search(text)]
        return sorted(set(found_techs))
    
    def extract_domains(self, text: str, skills: List[str], techs: List[str]) -> List[str]:
        found_domains = [domain for domain, pat in DOMAIN_PATTERNS.items() if pat.search(text)]
        
        # Rule-based domain inference based on extracted tech
        combined = set(skills + techs)
        if combined & {"react", "vue", "angular", "next.js"}:
            found_domains.append("frontend")
        if combined & {"django", "fastapi", "flask", "spring", "spring boot", "node.js"}:
            found_domains.append("backend")
        if combined & {"tensorflow", "pytorch", "scikit-learn", "machine learning", "deep learning"}:
            found_domains.append("ai")
        if combined & {"aws", "docker", "kubernetes", "terraform", "ci/cd"}:
            found_domains.append("devops")
            
        return sorted(set(found_domains))
    
    def extract_experience_years(self, text: str) -> Optional[float]:
        # Handles decimals (e.g., "2.5") and various common resume phrasings
        patterns = [
            r'(\d+(?:\.\d+)?)\+?\s*years?\s*(?:of\s*)?experience',
            r'experience\s*[:\-]?\s*(\d+(?:\.\d+)?)\+?\s*years?',
            r'(\d+(?:\.\d+)?)\+?\s*yrs?\s*(?:of\s*)?exp',
            r'total\s*experience\s*[:\-]?\s*(\d+(?:\.\d+)?)',
            r'professional\s*experience\s*.*?(\d+(?:\.\d+)?)\+?\s*years?'
        ]
        
        text_lower = text.lower()
        for pattern in patterns:
            match = re.search(pattern, text_lower)
            if match:
                try:
                    return float(match.group(1))
                except ValueError:
                    continue
        return None
    
    def extract_education(self, text: str) -> List[str]:
        education = [edu for edu, pat in EDU_PATTERNS.items() if pat.search(text)]
        return sorted(set(education))
    
    def extract_text_from_docx(self, file_bytes: bytes) -> str:
        try:
            import docx
            doc = docx.Document(io.BytesIO(file_bytes))
            return "\n".join([para.text for para in doc.paragraphs])
        except ImportError:
            raise ValueError("python-docx is not installed.")
        except Exception as e:
            raise ValueError(f"Failed to parse DOCX: {str(e)}")

    def extract_text_from_image(self, file_bytes: bytes) -> str:
        try:
            import pytesseract
            from PIL import Image
            image = Image.open(io.BytesIO(file_bytes))
            return pytesseract.image_to_string(image)
        except ImportError:
            raise ValueError("pytesseract or Pillow is not installed.")
        except Exception as e:
            raise ValueError(f"Failed to parse Image: {str(e)}")
            
    def parse(self, file_bytes: bytes, filename: str) -> Tuple[str, ResumeInfo]:
        filename_lower = filename.lower()
        if filename_lower.endswith('.pdf'):
            text = self.extract_text_from_pdf(file_bytes)
        elif filename_lower.endswith('.docx'):
            text = self.extract_text_from_docx(file_bytes)
        elif filename_lower.endswith(('.png', '.jpg', '.jpeg')):
            text = self.extract_text_from_image(file_bytes)
        else:
            text = file_bytes.decode('utf-8', errors='ignore')
        return self.parse_text(text)

    def parse_text(self, text: str) -> Tuple[str, ResumeInfo]:
        skills = self.extract_skills(text)
        technologies = self.extract_technologies(text)
        domains = self.extract_domains(text, skills, technologies)
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