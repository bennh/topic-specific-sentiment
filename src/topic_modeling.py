# src/topic_modeling.py

import pandas as pd

from sklearn.feature_extraction import text
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation


CUSTOM_STOPWORDS = [
    # Reuters-specific
    "said",
    "reuters",
    "source",
    "sources",
    "exclusive",

    # weekdays
    "monday",
    "tuesday",
    "wednesday",
    "thursday",
    "friday",
    "saturday",
    "sunday",

    # generic news words
    "percent",
    "week",
    "weeks",
    "month",
    "months",
    "year",
    "years",

    # business boilerplate
    "company",
    "companies",
    "group",

    # web artifacts
    "com"
]


class LDATopicModel:
    """
    LDA-based topic modeling for Reuters financial news.
    """

    def __init__(
        self,
        n_topics: int = 10,
        min_df: int = 20,
        max_df: float = 0.95,
        random_state: int = 42
    ):

        self.n_topics = n_topics

        stop_words = (
            text.ENGLISH_STOP_WORDS
            .union(CUSTOM_STOPWORDS)
        )

        self.vectorizer = CountVectorizer(
            stop_words=list(stop_words),
            min_df=min_df,
            max_df=max_df,
            ngram_range=(1, 2)
        )

        self.lda = LatentDirichletAllocation(
            n_components=n_topics,
            random_state=random_state,
            learning_method="batch",
            max_iter=20
        )

        self.feature_names = None
        self.X = None

    def fit(
        self,
        texts
    ) -> "LDATopicModel":
        """
        Fit the LDA model.
        """

        self.X = self.vectorizer.fit_transform(texts)

        self.feature_names = (
            self.vectorizer.get_feature_names_out()
        )

        self.lda.fit(self.X)

        return self

    def transform(self):
        """
        Return document-topic distributions.
        """

        if self.X is None:
            raise ValueError(
                "Model must be fitted before transform()."
            )

        return self.lda.transform(self.X)

    def assign_topics(self):
        """
        Assign the most probable topic
        to each document.
        """

        topic_dist = self.transform()

        return topic_dist.argmax(axis=1)

    def get_topics(
        self,
        top_n_words: int = 15
    ) -> dict:
        """
        Return top keywords for each topic.
        """

        topics = {}

        for topic_idx, topic in enumerate(
            self.lda.components_
        ):

            top_indices = (
                topic.argsort()[-top_n_words:]
            )

            words = [
                self.feature_names[i]
                for i in top_indices
            ]

            topics[topic_idx] = words

        return topics

    def topics_dataframe(
        self,
        top_n_words: int = 15
    ) -> pd.DataFrame:
        """
        Return topics as a dataframe.
        """

        topics = self.get_topics(
            top_n_words=top_n_words
        )

        rows = []

        for topic_id, words in topics.items():

            rows.append(
                {
                    "topic_id": topic_id,
                    "keywords": ", ".join(words)
                }
            )

        return pd.DataFrame(rows)

    def add_topics_to_dataframe(
        self,
        df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Add topic assignments to dataframe.
        """

        df = df.copy()

        df["topic"] = self.assign_topics()

        return df

    def save_topics(
        self,
        filepath: str,
        top_n_words: int = 15
    ):
        """
        Save topic keywords to CSV.
        """

        topics_df = self.topics_dataframe(
            top_n_words=top_n_words
        )

        topics_df.to_csv(
            filepath,
            index=False
        )