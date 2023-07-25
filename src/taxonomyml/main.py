"""Main module for taxonomy creation."""

from typing import List, Union

import pandas as pd
from loguru import logger

from taxonomyml import settings
from taxonomyml.lib import gsc
from taxonomyml.lib.api import get_openai_response_chat
from taxonomyml.lib.clustering import ClusterTopics
from taxonomyml.lib.gscutils import (
    load_gsc_account_data,
)
from taxonomyml.lib.nlp import (
    clean_gsc_dataframe,
    clean_provided_dataframe,
    filter_knee,
    get_ngram_frequency,
    get_structure,
    merge_ngrams,
)
from taxonomyml.lib.prompts import (
    PROMPT_TEMPLATE_TAXONOMY,
    PROMPT_TEMPLATE_TAXONOMY_REVIEW,
)


def get_gsc_data(
    gsc_client: gsc.GoogleSearchConsole,
    prop_url: str,
    days: int = 30,
    brand_terms: Union[List[str], None] = None,
    limit_queries: Union[int, None] = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = load_gsc_account_data(gsc_client=gsc_client, prop_url=prop_url, days=days)

    # Save original dataframe
    df_original = df.copy()

    df = clean_gsc_dataframe(df, brand_terms, limit_queries)

    return df, df_original


def get_df_data(
    data: Union[str, pd.DataFrame],
    text_column: str = None,
    search_volume_column: str = None,
    brand_terms: Union[List[str], None] = None,
    limit_queries: Union[int, None] = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Get data from Google Search Console or a pandas dataframe."""

    if isinstance(data, str) and (".csv" in data):
        df = pd.read_csv(data)
    else:
        df = data

    if text_column is None:
        text_column = input("What is the name of the column with the queries? ")

    if search_volume_column is None:
        search_volume_column = input(
            "What is the name of the column with the search volume? "
        )

    # Rename columns
    df = df.rename(
        columns={text_column: "query", search_volume_column: "search_volume"}
    )

    # Check if there is a column that contains URLs in the rows
    url_columns = [
        c
        for c in df.columns
        if df[c].dtype == "object" and df.head(10)[c].str.match(r"https?://").all()
    ]

    if len(url_columns) == 1:
        logger.info(f"Found URL column: {url_columns[0]}.")
        df = df.rename(columns={url_columns[0]: "page"})
    else:
        limit_queries = None

    # Save original dataframe
    df_original = df.copy()

    # Clean
    df = clean_provided_dataframe(df, brand_terms, limit_queries)

    return df, df_original


def score_and_filter_df(
    df: pd.DataFrame,
    ngram_range: tuple = (1, 6),
    min_df: int = 2,
) -> pd.DataFrame:
    """Score and filter dataframe."""

    df_ngram = get_ngram_frequency(
        df["query"].tolist(), ngram_range=ngram_range, min_df=min_df
    )
    logger.info(f"Got ngram frequency. Dataframe shape: {df_ngram.shape}")

    df_ngram = merge_ngrams(df_ngram)
    logger.info(f"Merged Ngrams. Dataframe shape: {df_ngram.shape}")

    df_ngram = df_ngram.rename(columns={"feature": "query"})

    # Sum search_volume column for df rows where query matches all terms in df_ngram["query"]. Do not use .str.contains() because may match parts of words.
    def match_all_terms(text, longer_text):
        t = text.lower().split(" ")
        lt = longer_text.lower().split(" ")
        return all([x in lt for x in t])

    df_ngram["search_volume"] = df_ngram["query"].apply(
        lambda x: df[df["query"].apply(lambda y: match_all_terms(x, y))][
            "search_volume"
        ].sum()
    )

    # Normalize the columns: search_volume
    df_ngram["search_volume"] = (
        df_ngram["search_volume"] / df_ngram["search_volume"].max()
    )

    # Update score column to be the average of the normalized column
    df_ngram["score"] = df_ngram[["search_volume", "merged_frequency"]].mean(axis=1)

    # Sort by score
    df_ngram = df_ngram.sort_values(by=["score"], ascending=False)

    df_ngram = df_ngram.reset_index(drop=True)

    if len(df_ngram) <= settings.MAX_SAMPLES:
        logger.info(f"Final score and filter length: {len(df_ngram)}")
        print(df_ngram.head())
        return df_ngram

    df_knee = None
    max_samples = settings.MAX_SAMPLES

    # Filter by knee
    while df_knee is None or len(df_knee) > settings.MAX_SAMPLES:
        df_knee = filter_knee(df_ngram.copy(), col_name="score", knee_s=max_samples)
        max_samples -= 25

    logger.info(
        f"Filtered Knee (sensitivity={int(max_samples + 25)}). Dataframe shape: {df_knee.shape}"
    )

    return df_knee


class PromptLengthError(Exception):
    """Raised when prompt is too long."""

    def __init__(self, message="Prompt is too long."):
        self.message = message
        super().__init__(self.message)


def create_taxonomy(
    data: Union[str, pd.DataFrame],
    gsc_client: gsc.GoogleSearchConsole | None = None,
    website_subject: str = "",
    text_column: str = None,
    search_volume_column: str = None,
    cluster_embeddings_model: Union[str, None] = "local",  # "openai" or "local"
    cross_encoded: bool = False,
    days: int = 30,
    ngram_range: tuple = (1, 5),
    min_df: int = 5,
    brand_terms: List[str] = None,
    limit_queries_per_page: int = 5,
    debug_responses: bool = False,
    **kwargs,
):
    """Kickoff function to create taxonomy from GSC data.

    Args:
        data (Union[str, pd.DataFrame]): GSC Property, CSV Filename, or pandas dataframe.
        website_subject (str, optional): Subject of the website. Defaults to "".
        text_column (str, optional): Name of the column with the queries. Defaults to None.
        search_volume_column (str, optional): Name of the column with the search volume. Defaults to None.
        cluster_embeddings_model (Union[str, None], optional): Name of the cluster embeddings model. Defaults to "local".
        cross_encoded (bool, optional): Whether to use cross encoded clustering. Defaults to False.
        days (int, optional): Number of days to get data from. Defaults to 30.
        ngram_range (tuple, optional): Ngram range to use for scoring. Defaults to (1, 6).
        min_df (int, optional): Minimum document frequency to use for scoring. Defaults to 2.
        brand_terms (List[str], optional): List of brand terms to remove from queries. Defaults to None.
        limit_queries_per_page (int, optional): Number of queries to use for clustering. Defaults to 5.
        debug_responses (bool, optional): Whether to print debug responses. Defaults to False.

    Returns:
        structure, df, samples
        Tuple[List[str], pd.DataFrame, str: Taxonomy list, original dataframe, and query_data.
    """

    # Get data
    if isinstance(data, str) and ("sc-domain:" in data or "https://" in data):
        df, df_original = get_gsc_data(
            gsc_client=gsc_client,
            prop_url=data,
            days=days,
            brand_terms=brand_terms,
            limit_queries=limit_queries_per_page,
        )
    elif (
        isinstance(data, pd.DataFrame)
        or isinstance(data, str)
        and data.endswith(".csv")
    ):
        df, df_original = get_df_data(
            data=data,
            text_column=text_column,
            search_volume_column=search_volume_column,
            brand_terms=brand_terms,
            limit_queries=limit_queries_per_page,
        )
    else:
        raise ValueError(
            "Data must be a GSC Property, CSV Filename, or pandas dataframe."
        )

    logger.info(f"Got Data. Dataframe shape: {df.shape}")

    logger.info("Filtering Query Data.")
    df_ngram = score_and_filter_df(df, ngram_range=ngram_range, min_df=min_df)
    logger.info(f"Got ngram frequency. Dataframe shape: {df_ngram.shape}")
    query_data = df_ngram.head(settings.MAX_SAMPLES)[["query", "score"]].to_markdown(
        index=None
    )

    logger.info(f"Got query data as markdown. Length: {len(query_data)}")
    brand_terms = ", ".join(brand_terms) if brand_terms else ""

    prompt = PROMPT_TEMPLATE_TAXONOMY.format(
        subject=website_subject, query_data=query_data, brands=brand_terms
    )

    logger.info("Using OpenAI API.")
    response = get_openai_response_chat(prompt, model=settings.OPENAI_LARGE_MODEL)

    logger.info("Reviewing OpenAI's work.")
    prompt = PROMPT_TEMPLATE_TAXONOMY_REVIEW.format(
        taxonomy=response, brands=brand_terms
    )
    reviewed_response = get_openai_response_chat(
        prompt, model=settings.OPENAI_LARGE_MODEL
    )

    if not response or not reviewed_response:
        logger.error("No response from API.")
        return None

    if debug_responses:
        logger.info("Debugging responses.")
        logger.info("Initial response:")
        logger.info(response)
        logger.info("Reviewed response:")
        logger.info(reviewed_response)

    # Get structure
    logger.info("Getting structure.")
    structure = get_structure(reviewed_response)

    # Add categories
    logger.info("Adding categories.")

    df = df_original if len(df_original) > 0 else df

    df = add_categories(
        structure,
        df,
        cluster_embeddings_model=cluster_embeddings_model,
        cross_encoded=cross_encoded,
        **kwargs,
    )

    logger.info("Done.")

    return structure, df, query_data


def add_categories(
    structure: List[str],
    df: pd.DataFrame,
    cluster_embeddings_model: Union[str, None] = None,
    cross_encoded: bool = False,
    match_col: str = "query",
    **kwargs,
) -> pd.DataFrame:
    """Add categories to dataframe."""
    texts = df[match_col].tolist()
    structure_parts = [" ".join(s.split(" > ")) for s in structure]
    structure_map = {p: s for p, s in zip(structure_parts, structure)}
    if "<outlier>" not in structure_map:
        structure_map["<outlier>"] = "Miscellaneous"

    model = ClusterTopics(
        embedding_model=cluster_embeddings_model,
        cluster_categories=structure_parts,
    )

    if cross_encoded:
        _, text_labels = model.fit_pairwise_crossencoded(texts, **kwargs)
    else:
        _, text_labels = model.fit_pairwise(texts)

    label_lookup = {
        text: structure_map[label] for text, label in zip(texts, text_labels)
    }
    df["taxonomy"] = df[match_col].map(label_lookup)

    return df


def add_categories_clustered(
    structure: List[str],
    df: pd.DataFrame,
    cluster_embeddings_model: Union[str, None] = None,
    min_cluster_size: int = 5,
    min_samples: int = 2,
    match_col: str = "query",
) -> pd.DataFrame:
    """Add categories to dataframe."""
    texts = df[match_col].tolist()
    structure_parts = [" ".join(s.split(" > ")) for s in structure]
    structure_map = {p: s for p, s in zip(structure_parts, structure)}
    if "<outliers>" not in structure_map:
        structure_map["<outliers>"] = "Miscellaneous"

    model = ClusterTopics(
        embedding_model=cluster_embeddings_model,
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        reduction_dims=2,
        cluster_model="hdbscan",
        cluster_categories=structure_parts,
        keep_outliers=True,
    )

    labels, text_labels = model.fit(texts)
    label_lookup = {
        text: structure_map[label] for text, label in zip(texts, text_labels)
    }
    df["taxonomy"] = df[match_col].map(label_lookup)

    return df
