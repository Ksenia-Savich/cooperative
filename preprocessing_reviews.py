import numpy as np
import pandas as pd
import re
import string
import nltk
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import Counter
from nltk.corpus import stopwords
from nltk import word_tokenize, regexp_tokenize, FreqDist
from pymorphy3 import MorphAnalyzer

nltk.download("stopwords", quiet=True)

# загрузка датасета, разделитель табуляция
df = pd.read_csv("reviews.csv", encoding="UTF8", sep="\t")
df = df.rename(columns={"review": "text", "sentiment": "label"})
df.head()


# стоп слова из nltk для русского языка
STOPWORDS = set(stopwords.words("russian"))


# ==== БАЗОВАЯ ОТЧИСТКА

# приведение к нижнему регистру — уменьшает размер словаря
df["text_ready"] = df["text"].str.lower()

def remove_html(text):
    """удаляет HTML-теги"""
    return re.sub(r"<[^>]+>", "", text)
df["text_ready"] = df["text_ready"].apply(remove_html)

def remove_urls(text):
    """удаляет ссылки http(s)://... и www..."""
    return re.sub(r"https?://\S+|www\.\S+", "", text)
df["text_ready"] = df["text_ready"].apply(remove_urls)

def remove_emoji(text):
    """удаляет эмодзи по диапазонам unicode кодов"""
    emoji_pattern = re.compile(
        "["
        u"\U0001F600-\U0001F64F"  # эмотиконы
        u"\U0001F300-\U0001F5FF"  # символы и пиктограммы
        u"\U0001F680-\U0001F6FF"  # транспорт
        u"\U0001F1E0-\U0001F1FF"  # флаги
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r"", text)
df["text_ready"] = df["text_ready"].apply(remove_emoji)

# удаление цифр через регулярное выражение \d+
df["text_ready"] = df["text_ready"].str.replace(r"\d+", "", regex=True)

# удаление пунктуации: всё, что не буква/цифра/пробел
df["text_ready"] = df["text_ready"].str.replace(r"[^\w\s]", "", regex=True)

def remove_stopwords(text):
    """убирает служебные слова"""
    return " ".join(word for word in str(text).split() if word not in STOPWORDS)
df["text_ready"] = df["text_ready"].apply(remove_stopwords)

# ==== ТОКЕНИЗАЦИЯ И ЛЕММАТИЗАЦИЯ

morph = MorphAnalyzer()

def tokenize_lemmas(sent, pat=r"(?u)\b\w\w+\b"):
    """
    Разбивает строку на токены
    и приводит каждый к лемме.
    """
    tokens = regexp_tokenize(sent, pat)   # \w\w+ = минимум 2 символа
    return [morph.parse(tok)[0].normal_form for tok in tokens]

df["tokens"] = df["text_ready"].apply(tokenize_lemmas)
# для моделей, ожидающих строку
df["text_lemmas"] = df["tokens"].apply(" ".join)

print(df[["text", "label", "text_lemmas"]].head())

# ==== ЧАСТОТНЫЙ АНАЛИЗ ПО КЛАССАМ

# cчитаем частоты всех слов корпуса
cnt = Counter()
for text in df["text_ready"]:
    for word in text.split():
        cnt[word] += 1

print(cnt.most_common(20)) # для отчёта топ 20 самых частых

#удаляем 10 самых частых слов
freq_high = {w for w, _ in cnt.most_common(10)}
df["text_ready"] = df["text_ready"].apply(
    lambda t: " ".join(w for w in t.split() if w not in freq_high)
)
df["text_lemmas"] = df["tokens"].apply(" ".join)

# разделяем корпус по меткам
pos_tokens = [w for toks in df.loc[df["label"] == "positive", "tokens"] for w in toks]
neg_tokens = [w for toks in df.loc[df["label"] == "negative", "tokens"] for w in toks]

freq_pos = FreqDist(pos_tokens)
freq_neg = FreqDist(neg_tokens)

print("ТОП-20 в ПОЗИТИВНЫХ отзывах:")
print(freq_pos.most_common(20))
print("\nТОП-20 в НЕГАТИВНЫХ отзывах:")
print(freq_neg.most_common(20))

# визуализация — 20 самых частых слов для каждого класса
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# ==== ГРАФИК ДЛЯ ПОЗИТИВА
pos_words, pos_counts = zip(*freq_pos.most_common(20))
axes[0].bar(pos_words, pos_counts, color = 'green')
axes[0].set_title("positive")
axes[0].tick_params(axis='x', rotation=90) # чтобы подписи на наезжали
# ==== ГРАФИК ДЛЯ НЕГАТИВА
neg_words, neg_counts = zip(*freq_neg.most_common(20))
axes[1].bar(neg_words, neg_counts, color='red')
axes[1].set_title("Negative")
axes[1].tick_params(axis='x', rotation=90)

plt.tight_layout()
plt.savefig("freq_plots.png", dpi=150)
print("Графики сохранены в файл freq_plots.png")
