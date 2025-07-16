"""Text processing utilities for resume classification."""

import re
from typing import List, Optional

import spacy
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from ..utils.config import config

# Download required NLTK data
try:
    import nltk

    nltk.download("punkt", quiet=True)
    nltk.download("stopwords", quiet=True)
    nltk.download("wordnet", quiet=True)
    nltk.download("averaged_perceptron_tagger", quiet=True)
except ImportError:
    pass


class TextProcessor:
    """Text processing utilities for resume and job description text."""

    def __init__(self, spacy_model: str = "en_core_web_sm"):
        """Initialize text processor.

        Args:
            spacy_model: Name of the spaCy model to use
        """
        self.spacy_model = spacy_model
        self.nlp = None
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words("english"))
        self._setup_spacy()

    def _setup_spacy(self):
        """Setup spaCy model."""
        try:
            self.nlp = spacy.load(self.spacy_model)
        except OSError:
            raise ValueError(
                f"spaCy model '{self.spacy_model}' not found. "
                f"Please install it with: python -m spacy download {self.spacy_model}"
            )

    def clean_text(self, text: str) -> str:
        """Clean and normalize text.

        Args:
            text: Input text

        Returns:
            Cleaned text
        """
        if not text:
            return ""

        # Convert to lowercase
        text = text.lower()

        # Remove extra whitespace
        text = re.sub(r"\s+", " ", text)

        # Remove special characters but keep basic punctuation
        text = re.sub(r"[^a-zA-Z0-9\s.,!?;:()]", "", text)

        # Remove multiple punctuation
        text = re.sub(r"([.,!?;:])\1+", r"\1", text)

        return text.strip()

    def remove_stopwords(self, text: str) -> str:
        """Remove stopwords from text.

        Args:
            text: Input text

        Returns:
            Text with stopwords removed
        """
        words = word_tokenize(text)
        filtered_words = [word for word in words if word.lower() not in self.stop_words]
        return " ".join(filtered_words)

    def lemmatize_text(self, text: str) -> str:
        """Lemmatize text.

        Args:
            text: Input text

        Returns:
            Lemmatized text
        """
        words = word_tokenize(text)
        lemmatized_words = [self.lemmatizer.lemmatize(word) for word in words]
        return " ".join(lemmatized_words)

    def extract_entities(self, text: str) -> dict:
        """Extract named entities from text.

        Args:
            text: Input text

        Returns:
            Dictionary of entity types and their values
        """
        if self.nlp is None:
            return {}

        doc = self.nlp(text)
        entities = {}

        for ent in doc.ents:
            if ent.label_ not in entities:
                entities[ent.label_] = []
            entities[ent.label_].append(ent.text)

        return entities

    def extract_skills(self, text: str) -> List[str]:
        """Extract skills from text.

        Args:
            text: Input text

        Returns:
            List of extracted skills
        """
        # Common skill patterns
        skill_patterns = [
            r"\b(python|java|c\+\+|javascript|html|css|sql|r|matlab|scala|go|rust|php|ruby|swift|kotlin)\b",
            r"\b(machine learning|deep learning|ai|artificial intelligence|nlp|natural language processing)\b",
            r"\b(data science|data analysis|statistics|mathematics|linear algebra|calculus)\b",
            r"\b(aws|azure|gcp|docker|kubernetes|jenkins|git|github|gitlab)\b",
            r"\b(tensorflow|pytorch|scikit-learn|pandas|numpy|matplotlib|seaborn)\b",
            r"\b(react|angular|vue|node\.js|django|flask|spring|express)\b",
            r"\b(mongodb|postgresql|mysql|redis|elasticsearch|kafka|spark)\b",
            r"\b(agile|scrum|kanban|waterfall|devops|ci/cd|tdd|bdd)\b",
        ]

        skills = set()

        for pattern in skill_patterns:
            matches = re.findall(pattern, text.lower())
            skills.update(matches)

        return list(skills)

    def extract_education(self, text: str) -> List[str]:
        """Extract education information from text.

        Args:
            text: Input text

        Returns:
            List of education degrees/certifications
        """
        education_patterns = [
            r"\b(bachelor|master|phd|doctorate|associate|diploma|certificate)\b",
            r"\b(bs|ba|ms|ma|phd|mba|mfa|bsc|msc|beng|meng)\b",
            r"\b(computer science|engineering|mathematics|physics|chemistry|biology)\b",
            r"\b(business|economics|finance|marketing|management|accounting)\b",
        ]

        education = set()

        for pattern in education_patterns:
            matches = re.findall(pattern, text.lower())
            education.update(matches)

        return list(education)

    def extract_experience_years(self, text: str) -> Optional[int]:
        """Extract years of experience from text.

        Args:
            text: Input text

        Returns:
            Years of experience or None if not found
        """
        experience_patterns = [
            r"(\d+)\+?\s*years?\s*of?\s*experience",
            r"experience\s*of?\s*(\d+)\+?\s*years?",
            r"(\d+)\+?\s*years?\s*in\s*the\s*field",
            r"(\d+)\+?\s*years?\s*working\s*experience",
        ]

        for pattern in experience_patterns:
            match = re.search(pattern, text.lower())
            if match:
                return int(match.group(1))

        return None

    def extract_company_names(self, text: str) -> List[str]:
        """Extract company names from text.

        Args:
            text: Input text

        Returns:
            List of company names
        """
        if self.nlp is None:
            return []

        doc = self.nlp(text)
        companies = []

        for ent in doc.ents:
            if ent.label_ == "ORG":
                companies.append(ent.text)

        return companies

    def extract_job_titles(self, text: str) -> List[str]:
        """Extract job titles from text.

        Args:
            text: Input text

        Returns:
            List of job titles
        """
        job_title_patterns = [
            r"\b(software engineer|data scientist|product manager|project manager)\b",
            r"\b(frontend|backend|full stack|devops|ml|ai)\s+(engineer|developer)\b",
            r"\b(senior|junior|lead|principal|staff)\s+(engineer|developer|scientist)\b",
            r"\b(analyst|consultant|specialist|coordinator|director|manager)\b",
        ]

        job_titles = set()

        for pattern in job_title_patterns:
            matches = re.findall(pattern, text.lower())
            job_titles.update(matches)

        return list(job_titles)

    def extract_locations(self, text: str) -> List[str]:
        """Extract locations from text.

        Args:
            text: Input text

        Returns:
            List of locations
        """
        if self.nlp is None:
            return []

        doc = self.nlp(text)
        locations = []

        for ent in doc.ents:
            if ent.label_ in ["GPE", "LOC"]:
                locations.append(ent.text)

        return locations

    def extract_contact_info(self, text: str) -> dict:
        """Extract contact information from text.

        Args:
            text: Input text

        Returns:
            Dictionary with contact information
        """
        contact_info = {}

        # Email pattern
        email_pattern = r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"
        email_match = re.search(email_pattern, text)
        if email_match:
            contact_info["email"] = email_match.group()

        # Phone pattern
        phone_pattern = (
            r"(\+?1?[-.\s]?)?\(?([0-9]{3})\)?[-.\s]?([0-9]{3})[-.\s]?([0-9]{4})"
        )
        phone_match = re.search(phone_pattern, text)
        if phone_match:
            contact_info["phone"] = "".join(phone_match.groups())

        # LinkedIn pattern
        linkedin_pattern = r"linkedin\.com/in/[A-Za-z0-9-]+"
        linkedin_match = re.search(linkedin_pattern, text)
        if linkedin_match:
            contact_info["linkedin"] = linkedin_match.group()

        return contact_info

    def extract_keywords(self, text: str, top_n: int = 20) -> List[tuple]:
        """Extract most common keywords from text.

        Args:
            text: Input text
            top_n: Number of top keywords to return

        Returns:
            List of (keyword, frequency) tuples
        """
        if self.nlp is None:
            return []

        doc = self.nlp(text.lower())

        # Filter out stopwords, punctuation, and short words
        keywords = [
            token.text
            for token in doc
            if not token.is_stop and not token.is_punct and len(token.text) > 2
        ]

        # Count frequencies
        from collections import Counter

        keyword_freq = Counter(keywords)

        return keyword_freq.most_common(top_n)

    def get_text_statistics(self, text: str) -> dict:
        """Get text statistics.

        Args:
            text: Input text

        Returns:
            Dictionary with text statistics
        """
        if not text:
            return {
                "length": 0,
                "word_count": 0,
                "sentence_count": 0,
                "avg_word_length": 0,
            }

        words = text.split()
        sentences = text.split(".")

        return {
            "length": len(text),
            "word_count": len(words),
            "sentence_count": len([s for s in sentences if s.strip()]),
            "avg_word_length": sum(len(word) for word in words) / len(words)
            if words
            else 0,
        }

    def preprocess_text(
        self, text: str, remove_stopwords: bool = True, lemmatize: bool = True
    ) -> str:
        """Preprocess text with multiple steps.

        Args:
            text: Input text
            remove_stopwords: Whether to remove stopwords
            lemmatize: Whether to lemmatize

        Returns:
            Preprocessed text
        """
        # Clean text
        text = self.clean_text(text)

        # Remove stopwords
        if remove_stopwords:
            text = self.remove_stopwords(text)

        # Lemmatize
        if lemmatize:
            text = self.lemmatize_text(text)

        return text

    def validate_text(self, text: str) -> bool:
        """Validate text quality.

        Args:
            text: Input text

        Returns:
            True if text is valid, False otherwise
        """
        if not text or not text.strip():
            return False

        # Check minimum length
        if len(text.strip()) < config.data.MIN_TEXT_LENGTH:
            return False

        # Check maximum length
        if len(text) > config.data.MAX_TEXT_LENGTH:
            return False

        # Check if text contains meaningful content
        words = text.split()
        if len(words) < 5:
            return False

        return True

    def process_batch(self, texts: List[str], **kwargs) -> List[str]:
        """Process a batch of texts.

        Args:
            texts: List of input texts
            **kwargs: Additional arguments for preprocessing

        Returns:
            List of processed texts
        """
        processed_texts = []

        for text in texts:
            if self.validate_text(text):
                processed_text = self.preprocess_text(text, **kwargs)
                processed_texts.append(processed_text)
            else:
                processed_texts.append("")

        return processed_texts
