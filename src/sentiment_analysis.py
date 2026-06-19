# src/sentiment_analysis.py

from typing import List

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)


class FinBERTSentiment:
    """
    FinBERT-based sentiment analysis
    for financial news.
    """

    LABELS = [
        "positive",
        "negative",
        "neutral"
    ]

    SCORE_MAPPING = {
        "positive": 1,
        "neutral": 0,
        "negative": -1
    }

    def __init__(
        self,
        model_name: str = "ProsusAI/finbert"
    ):

        self.device = (
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        print(
            f"Using device: {self.device}"
        )

        self.tokenizer = (
            AutoTokenizer.from_pretrained(
                model_name
            )
        )

        self.model = (
            AutoModelForSequenceClassification
            .from_pretrained(model_name)
        )

        self.model.to(
            self.device
        )

        self.model.eval()

    @torch.no_grad()
    def predict(
        self,
        text: str
    ) -> str:
        """
        Predict sentiment label
        for a single text.
        """

        text = str(text)

        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512
        )

        inputs = {
            k: v.to(self.device)
            for k, v in inputs.items()
        }

        outputs = self.model(
            **inputs
        )

        probs = torch.softmax(
            outputs.logits,
            dim=1
        )

        pred_idx = (
            torch.argmax(probs)
            .item()
        )

        return self.LABELS[pred_idx]

    def predict_score(
        self,
        text: str
    ) -> int:
        """
        Predict sentiment score.
        """

        label = self.predict(text)

        return self.SCORE_MAPPING[label]

    @torch.no_grad()
    def predict_proba(
        self,
        text: str
    ) -> dict:
        """
        Return probability distribution.
        """

        text = str(text)

        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512
        )

        inputs = {
            k: v.to(self.device)
            for k, v in inputs.items()
        }

        outputs = self.model(
            **inputs
        )

        probs = (
            torch.softmax(
                outputs.logits,
                dim=1
            )
            .cpu()
            .numpy()[0]
        )

        return {
            label: float(prob)
            for label, prob in zip(
                self.LABELS,
                probs
            )
        }

    @torch.no_grad()
    def predict_with_confidence(
        self,
        text: str
    ) -> dict:
        """
        Return label and confidence.
        """

        text = str(text)

        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512
        )

        inputs = {
            k: v.to(self.device)
            for k, v in inputs.items()
        }

        outputs = self.model(
            **inputs
        )

        probs = torch.softmax(
            outputs.logits,
            dim=1
        )

        pred_idx = (
            torch.argmax(probs)
            .item()
        )

        confidence = (
            probs[0, pred_idx]
            .item()
        )

        return {
            "label":
                self.LABELS[pred_idx],
            "confidence":
                confidence
        }

    @torch.no_grad()
    def predict_batch(
        self,
        texts,
        batch_size: int = 32
    ) -> List[str]:
        """
        Predict sentiment labels
        for multiple texts.
        """

        texts = [
            str(text)
            for text in texts
        ]

        predictions = []

        for i in range(
            0,
            len(texts),
            batch_size
        ):

            batch = texts[
                i:i + batch_size
            ]

            inputs = self.tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            )

            inputs = {
                k: v.to(self.device)
                for k, v in inputs.items()
            }

            outputs = self.model(
                **inputs
            )

            probs = torch.softmax(
                outputs.logits,
                dim=1
            )

            pred_indices = (
                torch.argmax(
                    probs,
                    dim=1
                )
                .cpu()
                .numpy()
            )

            predictions.extend(
                [
                    self.LABELS[idx]
                    for idx in pred_indices
                ]
            )

        return predictions

    @torch.no_grad()
    def predict_batch_proba(
        self,
        texts,
        batch_size: int = 32
    ):
        """
        Return probability matrix.

        Columns:
        positive
        negative
        neutral
        """

        texts = [
            str(text)
            for text in texts
        ]

        results = []

        for i in range(
            0,
            len(texts),
            batch_size
        ):

            batch = texts[
                i:i + batch_size
            ]

            inputs = self.tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            )

            inputs = {
                k: v.to(self.device)
                for k, v in inputs.items()
            }

            outputs = self.model(
                **inputs
            )

            probs = (
                torch.softmax(
                    outputs.logits,
                    dim=1
                )
                .cpu()
                .numpy()
            )

            results.extend(
                probs.tolist()
            )

        return results

    def predict_batch_scores(
        self,
        texts,
        batch_size: int = 32
    ) -> List[int]:
        """
        Predict sentiment scores.
        """

        labels = self.predict_batch(
            texts,
            batch_size=batch_size
        )

        return [
            self.SCORE_MAPPING[label]
            for label in labels
        ]