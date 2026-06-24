# src/visualization.py

import numpy as np
import pandas as pd

import matplotlib as mpl
import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.decomposition import PCA

plt.style.use("seaborn-v0_8-whitegrid")

# Resolution
mpl.rcParams["figure.dpi"] = 300
mpl.rcParams["savefig.dpi"] = 300

# Typography
mpl.rcParams["font.family"] = "STIXGeneral"
mpl.rcParams["mathtext.fontset"] = "stix"

mpl.rcParams["axes.labelsize"] = 11
mpl.rcParams["axes.titlesize"] = 12

mpl.rcParams["xtick.labelsize"] = 10
mpl.rcParams["ytick.labelsize"] = 10

mpl.rcParams["legend.fontsize"] = 10

# Grid
mpl.rcParams["grid.alpha"] = 0.15
mpl.rcParams["grid.linestyle"] = "--"
mpl.rcParams["grid.linewidth"] = 0.5

# Export
mpl.rcParams["savefig.bbox"] = "tight"


BLUE = "#2F5D8A"
RED = "#B23A48"

LIGHT_BLUE = "#D6E6F5"

DARK_GRAY = "#4D4D4D"
MID_GRAY = "#808080"
LIGHT_GRAY = "#D9D9D9"

POSITIVE_COLOR = BLUE
NEGATIVE_COLOR = RED
NEUTRAL_COLOR = MID_GRAY

SENTIMENT_COLORS = {
    "negative": RED,
    "neutral": MID_GRAY,
    "positive": BLUE
}

TOPIC_LABELS = {
    "Macroeconomy & Financial Markets": "T0",
    "Banking & Financial Services": "T1",
    "Aviation & Air Transportation": "T2",
    "US-China Trade War & Geopolitics": "T3",
    "Automotive Industry & Corporate Leadership": "T4",
    "International Trade & Economic Policy": "T5",
    "Big Tech & Digital Platforms": "T6",
    "COVID-19, Manufacturing & Labor Markets": "T7",
    "Corporate Earnings & Equity Markets": "T8",
    "Mergers & Acquisitions": "T9",
    "Oil & Energy Markets": "T10",
    "Financial Regulation & Corporate Scandals": "T11"
}

# --------------------------------------------------
# Figure 1
# Topic Distribution
# --------------------------------------------------

def plot_topic_distribution(
    df: pd.DataFrame,
    save_path: str = None
):

    topic_counts = (
        df["topic_name"]
        .value_counts()
        .sort_values(ascending=False)
    )

    topic_counts.index = [
        TOPIC_LABELS[t]
        for t in topic_counts.index
    ]

    fig, ax = plt.subplots(
        figsize=(10, 6)
    )

    topic_counts.plot(
        kind="bar",
        color=BLUE,
        edgecolor="none",
        width=0.6,
        ax=ax
    )

    ax.set_ylabel(
        "Number of Articles"
    )

    ax.set_xlabel("")

    ax.tick_params(
        axis="x",
        rotation=0
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.grid(
        axis="y",
        linestyle="--",
        alpha=0.3
    )

    plt.tight_layout()

    if save_path:

        plt.savefig(
            save_path,
            dpi=300,
            bbox_inches="tight"
        )

    plt.show()
    plt.close()


# --------------------------------------------------
# Figure 2
# Average Sentiment by Topic
# --------------------------------------------------

def plot_sentiment_by_topic(
    df: pd.DataFrame,
    save_path: str = None
):

    sentiment = (
        df.groupby("topic_name")
        ["sentiment_score"]
        .mean()
        .sort_values()
    )

    colors = [
        NEGATIVE_COLOR if x < 0
        else POSITIVE_COLOR
        for x in sentiment.values
    ]

    fig, ax = plt.subplots(
        figsize=(14, 7)
    )

    topic_labels = [
        TOPIC_LABELS[t]
        for t in sentiment.index
    ]

    bars = ax.barh(
        topic_labels,
        sentiment.values,
        color=colors,
        edgecolor="none"
    )

    # Value labels
    for bar, value in zip(
        bars,
        sentiment.values
    ):

        offset = 0.01

        if value < 0:

            ax.text(
                value - offset,
                bar.get_y()
                + bar.get_height() / 2,
                f"{value:.2f}",
                ha="right",
                va="center",
                fontsize=9
            )

        else:

            ax.text(
                value + offset,
                bar.get_y()
                + bar.get_height() / 2,
                f"{value:.2f}",
                ha="left",
                va="center",
                fontsize=9
            )

    ax.axvline(
        0,
        color="black",
        linewidth=1.2
    )

    ax.set_xlabel(
        "Average Sentiment Score"
    )

    ax.set_ylabel("")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.grid(
        axis="x",
        linestyle="--",
        alpha=0.3
    )

    plt.tight_layout()

    if save_path:

        plt.savefig(
            save_path,
            dpi=300,
            bbox_inches="tight"
        )

    plt.show()
    plt.close()


# --------------------------------------------------
# Figure 3
# Overall + Topic Dynamics
# --------------------------------------------------

def plot_sentiment_dynamics(
    df: pd.DataFrame,
    topics: list,
    freq: str = "W",
    save_path: str = None
):

    temp = df.copy()

    temp["Time"] = pd.to_datetime(
        temp["Time"]
    )

    fig, ax = plt.subplots(
        figsize=(12, 7)
    )

    # -------------------------
    # Overall Sentiment
    # -------------------------

    overall_ts = (
        temp
        .set_index("Time")
        ["sentiment_score"]
        .resample(freq)
        .mean()
    )

    ax.plot(
        overall_ts.index,
        overall_ts.values,
        color=DARK_GRAY,
        linestyle="--",
        linewidth=2.5,
        marker="o",
        markersize=4,
        label="Overall"
    )

    # -------------------------
    # Topic Sentiment
    # -------------------------

    topic_colors = [
      "#08306B",
      "#2171B5",
      "#6BAED6",
    ]

    topic_markers = [
        "s",
        "^",
        "D",
        "o"
    ]

    for topic, color, marker in zip(
        topics,
        topic_colors,
        topic_markers
    ):

        topic_df = (
            temp[
                temp["topic_name"] == topic
            ]
        )

        topic_ts = (
            topic_df
            .set_index("Time")
            ["sentiment_score"]
            .resample(freq)
            .mean()
        )

        ax.plot(
            topic_ts.index,
            topic_ts.values,
            color=color,
            linewidth=2,
            marker=marker,
            markersize=4,
            label=TOPIC_LABELS[topic]
        )

    ax.axhline(
        0,
        color="black",
        linewidth=1,
        alpha=0.6
    )

    ax.set_xlabel(
        "Date"
    )

    ax.set_ylabel(
        "Average Sentiment Score"
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.grid(
        linestyle="--",
        alpha=0.3
    )

    ax.legend(
        frameon=False,
        ncol=1,
        loc="upper left"
    )

    plt.tight_layout()

    if save_path:

        plt.savefig(
            save_path,
            dpi=300,
            bbox_inches="tight"
        )

    plt.show()
    plt.close()


# --------------------------------------------------
# Figure 4
# Topic Sentiment Heatmap
# --------------------------------------------------

def plot_topic_sentiment_heatmap(
    df: pd.DataFrame,
    save_path: str = None
):

    temp = df.copy()

    temp["Time"] = pd.to_datetime(
        temp["Time"]
    )

    temp["year_month"] = (
        temp["Time"]
        .dt.to_period("M")
        .astype(str)
    )

    heatmap_data = (
        temp.pivot_table(
            index="topic_name",
            columns="year_month",
            values="sentiment_score",
            aggfunc="mean"
        )
    )

    heatmap_data.index = [
        TOPIC_LABELS[t]
        for t in heatmap_data.index
    ]

    fig, ax = plt.subplots(
        figsize=(16, 7)
    )

    sns.heatmap(
        heatmap_data,
        cmap="vlag",
        center=0,
        vmin=-0.5,
        vmax=0.5,
        linewidths=0.3,
        linecolor="white",
        cbar_kws={
            "label":
            "Average Sentiment Score"
        },
        ax=ax
    )

    ax.set_xlabel(
        "Month"
    )

    ax.set_ylabel("")

    ax.tick_params(
        axis="y",
        labelsize=10
    )

    ax.tick_params(
        axis="x",
        rotation=45
    )

    plt.tight_layout()

    if save_path:

        plt.savefig(
            save_path,
            dpi=300,
            bbox_inches="tight"
        )

    plt.show()
    plt.close()


# --------------------------------------------------
# Figure 5
# Sentiment Confidence by Topic
# --------------------------------------------------

def plot_sentiment_confidence(
    df: pd.DataFrame,
    save_path: str = None
):

    temp = df.copy()

    temp["topic_id"] = (
        temp["topic_name"]
        .map(TOPIC_LABELS)
    )

    temp["confidence"] = (
        temp[
            [
                "positive_prob",
                "negative_prob",
                "neutral_prob"
            ]
        ]
        .max(axis=1)
    )

    topic_order = (
        temp.groupby("topic_id")
        ["confidence"]
        .mean()
        .sort_values()
        .index
    )

    fig, ax = plt.subplots(
        figsize=(12, 6)
    )

    sns.boxplot(
        data=temp,
        y="confidence",
        x="topic_id",
        order=topic_order,
        color=LIGHT_BLUE,
        width=0.6,
        fliersize=2,
        linewidth=1,
        ax=ax
    )

    sns.stripplot(
        data=temp.sample(
            min(2000, len(temp)),
            random_state=42
        ),
        y="confidence",
        x="topic_id",
        order=topic_order,
        color="black",
        alpha=0.08,
        size=2,
        ax=ax
    )

    ax.set_ylabel(
        "Prediction Confidence"
    )

    ax.set_xlabel("")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.grid(
        axis="y",
        linestyle="--",
        alpha=0.3
    )

    plt.tight_layout()

    if save_path:

        plt.savefig(
            save_path,
            dpi=300,
            bbox_inches="tight"
        )

    plt.show()
    plt.close()


# --------------------------------------------------
# Figure 6
# Topic Similarity Map
# --------------------------------------------------

def plot_topic_similarity(
    lda_model,
    topic_names,
    save_path: str = None
):

    topic_matrix = (
        lda_model
        .lda
        .components_
    )

    coords = PCA(
        n_components=2,
        random_state=42
    ).fit_transform(
        topic_matrix
    )

    explained_var = (
        PCA(
            n_components=2,
            random_state=42
        )
        .fit(topic_matrix)
        .explained_variance_ratio_
    )

    fig, ax = plt.subplots(
        figsize=(11, 8)
    )

    ax.scatter(
        coords[:, 0],
        coords[:, 1],
        s=140,
        color=BLUE,
        edgecolor="none",
        alpha=1
    )

    for i in range(len(coords)):

        ax.annotate(
            f"T{i}",
            (
                coords[i,0],
                coords[i,1]
            ),
            xytext=(3,3),
            textcoords="offset points",
            fontsize=8,
            fontweight="bold"
        )

    ax.axhline(
        0,
        color="gray",
        linewidth=0.8,
        alpha=0.5
    )

    ax.axvline(
        0,
        color="gray",
        linewidth=0.8,
        alpha=0.5
    )

    ax.grid(
        linestyle="--",
        alpha=0.3
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.set_xlabel(
        f"PCA 1 ({explained_var[0]:.1%} variance)"
    )

    ax.set_ylabel(
        f"PCA 2 ({explained_var[1]:.1%} variance)"
    )

    plt.tight_layout()

    if save_path:

        plt.savefig(
            save_path,
            dpi=300,
            bbox_inches="tight"
        )

    plt.show()
    plt.close()