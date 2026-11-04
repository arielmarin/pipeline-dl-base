"""Funciones de preprocesamiento del Módulo 2, reutilizadas en el TP3."""
import re
import html

import spacy

# Modelo de inglés de SpaCy (se descarga con: python -m spacy download en_core_web_sm)
nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])


def limpiar_texto(text):
    """Limpieza con Regex y normalización (minúsculas, remoción de ruido)."""
    text = html.unescape(str(text))                                    # entidades HTML
    text = re.sub(r"<[^>]+>", " ", text)                               # tags HTML
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)                 # URLs
    text = re.sub(r"\b[\w-]+(?:\.[\w-]+)*\.(?:com|net|org|edu|gov|biz|info)\b", " ", text, flags=re.I)  # dominios web
    text = re.sub(r"quote,?\s+profile,?\s+research|canadian\s+press|associated\s+press", " ", text, flags=re.I)  # firmas de agencia
    text = re.sub(r"\b(reuters|afp|ap)\b", " ", text, flags=re.I)      # firmas de agencia
    text = re.sub(r"\b(quot|amp|lt|gt);|#\d+;", " ", text, flags=re.I)  # restos de entidades
    text = text.lower()                                                # minúsculas
    text = re.sub(r"[^a-z\s]", " ", text)                              # números, puntuación, símbolos
    return re.sub(r"\s+", " ", text).strip()


def lematizar(texto_limpio):
    """Lematiza con SpaCy y quita stop-words y tokens de una sola letra."""
    doc = nlp(texto_limpio)
    return " ".join(t.lemma_ for t in doc if t.is_alpha and not t.is_stop and len(t) > 1)


def preprocess_text(text):
    """Pipeline completo para un texto: limpieza -> lematización."""
    return lematizar(limpiar_texto(text))