#! /usr/bin/env python
# coding: utf-8

"""Create clustering model using HDBScan and AgglomerativeClustering models with embeddings"""

import concurrent.futures
import math
import os
import warnings
from typing import List, Union

import numpy as np
import pandas as pd
import torch
import umap.umap_ as umap
from hdbscan import HDBSCAN
from kneed import KneeLocator
from loguru import logger
from numba.core.errors import NumbaDeprecationWarning
from sentence_transformers import CrossEncoder, SentenceTransformer
from sklearn.cluster import OPTICS, AgglomerativeClustering
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import silhouette_samples
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm

from taxonomyml import settings
from taxonomyml.lib.api import get_openai_embeddings, get_openai_response_chat
from taxonomyml.lib.ctfidf import ClassTfidfTransformer
from taxonomyml.lib.prompts import PROMPT_TEMPLATE_CLUSTER

warnings.filterwarnings("ignore", category=NumbaDeprecationWarning)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


# Clustering Class
class ClusterTopics:
    def __init__(
        self,
        embedding_model: Union[str, None] = None,
        min_cluster_size: int = 10,
        min_samples: Union[int, bool] = None,
        reduction_dims: Union[int, float] = 0,
        cluster_model: str = "hdbscan",
        use_llm_descriptions: bool = False,
        cluster_categories: List[str] = None,
        use_elbow: bool = True,
        keep_outliers: bool = False,
        n_jobs: int = 6,
        openai_api_key: Union[str, float] = None,
    ):
        """This class takes a list of sentences and clusters them using embeddings."""

        self.embedding_model = embedding_model or settings.LOCAL_EMBEDDING_MODEL
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples or round(math.sqrt(self.min_cluster_size))
        self.reduction_dims = reduction_dims
        self.cluster_model = cluster_model
        self.use_llm_descriptions = use_llm_descriptions

        self.cluster_categories = cluster_categories
        if cluster_categories:
            self.cluster_categories = [
                c for c in filter(len, list(set(cluster_categories)))
            ]

        self.use_elbow = use_elbow
        self.keep_outliers = keep_outliers
        self.n_jobs = n_jobs
        self.embeddings = None
        self.embedding_size = self.get_embeddings(["test"])[0].shape[0]
        self.corpus = None
        self.labels = None
        self.text_labels = None
        self.model = None
        self.model_data = None
        self.post_process = None
        self.openai_api_key = (
            openai_api_key if openai_api_key else settings.OPENAI_API_KEY
        )

    def get_embeddings(self, sentences: List[str]) -> np.ndarray:
        """Converts text to embeddings"""

        # Check if embeddings are already in memory
        if self.embeddings is not None and all([s in self.corpus for s in sentences]):
            idx = [np.where(self.corpus == s)[0][0] for s in sentences]
            return self.embeddings[idx]

        # Need to build embeddings
        if self.embedding_model == "openai":
            return get_openai_embeddings(
                sentences, n_jobs=self.n_jobs, openai_api_key=self.openai_api_key
            )

        else:
            logger.info("Using local embeddings")

            if self.embedding_model == "e5":
                self.embedding_model = "intfloat/e5-base-v2"
            elif self.embedding_model == "local":
                self.embedding_model = settings.LOCAL_EMBEDDING_MODEL
            else:
                logger.info(f"Using custom embedding model: {self.embedding_model}")

            # Only do batching and progress if many embeddings

            if len(sentences) > 64:
                embeddings = SentenceTransformer(self.embedding_model).encode(
                    sentences, show_progress_bar=True, batch_size=64
                )
            else:
                embeddings = SentenceTransformer(self.embedding_model).encode(sentences)

            return np.asarray(embeddings)

    def get_reduced(self, embeddings: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
        """Reduce dimensions using UMAP. This can reduce clustering time and memory,
        but at the expence of reduced infomration.
        Reducing to 2 dimensions is needed for plotting.
        TODO: Finding the right cluster terms (ngrams) breaks with dim reduction."""

        # No reduction
        if self.reduction_dims <= 0:
            return np.asarray(embeddings)

        if isinstance(self.reduction_dims, float):
            n_dims = math.ceil(embeddings.shape[1] * self.reduction_dims)
        else:
            n_dims = self.reduction_dims

        logger.info("Reducing embeddings to {} dims".format(n_dims))

        # returns np.ndarray
        return umap.UMAP(
            n_neighbors=self.min_samples,
            n_components=n_dims,
            random_state=settings.RANDOM_SEED,
        ).fit_transform(embeddings)

    def get_elbow(self, embeddings: Union[torch.Tensor, np.ndarray]) -> float:
        """Gets the elbow or sorted inflection point of input data as float."""

        if self.use_elbow:
            k = self.min_samples
            nbrs = NearestNeighbors(
                n_neighbors=k, n_jobs=self.n_jobs, algorithm="auto"
            ).fit(embeddings)
            distances, _ = nbrs.kneighbors(embeddings)
            distances = [np.mean(d) for d in np.sort(distances, axis=0)]
            kneedle = KneeLocator(
                distances,
                list(range(len(distances))),
                online=True,
                interp_method="polynomial",
                curve="convex",
                direction="increasing",
            )
            epsilon = np.min(list(kneedle.all_elbows))
            if epsilon == 0.0:
                epsilon = np.mean(distances)
        else:
            epsilon = 0.5

        logger.info("Using epsilon value: {}".format(epsilon))

        return float(epsilon)

    def set_cluster_model(self, cluster_model: str) -> None:
        """Sets the cluster type for the class"""
        self.cluster_model = cluster_model

    def set_embeddings(self, embeddings: np.ndarray) -> None:
        """Sets the embeddings for the class"""
        self.embeddings = embeddings

    def get_cluster_model(self, model_name: Union[str, None] = None) -> None:
        """Gets the correct clustering model and sets them up."""

        model_name = model_name or self.cluster_model

        logger.info("Cluster Model: {}".format(model_name))

        if model_name == "hdbscan":
            # Normalize embeddings
            self.model_data = self.embeddings / np.linalg.norm(
                self.embeddings, axis=1, keepdims=True
            )

            return HDBSCAN(
                min_samples=self.min_samples,
                min_cluster_size=self.min_cluster_size,
                cluster_selection_epsilon=self.get_elbow(self.model_data),
                core_dist_n_jobs=self.n_jobs,
            )

        elif model_name == "agglomerative":
            # Normalize embeddings
            self.model_data = self.embeddings / np.linalg.norm(
                self.embeddings, axis=1, keepdims=True
            )

            # If we want to find the optimal number of clusters.
            # n_clusters = self.find_optimal_clusters_agglomerative(self.embeddings)
            n_clusters = (
                len(self.cluster_categories)
                if isinstance(self.cluster_categories, list)
                else None
            )
            distance_threshold = (
                float(self.min_samples)
                if not isinstance(self.cluster_categories, list)
                else None
            )

            return AgglomerativeClustering(
                n_clusters=n_clusters,
                compute_full_tree=True,
                distance_threshold=distance_threshold,
            )

        elif model_name == "optics":
            # Normalize embeddings
            self.model_data = self.embeddings / np.linalg.norm(
                self.embeddings, axis=1, keepdims=True
            )

            return OPTICS(
                min_samples=self.min_samples,
                eps=self.get_elbow(self.model_data),
                min_cluster_size=self.min_cluster_size,
                n_jobs=self.n_jobs,
            )

        else:
            logger.error("Only `hdbscan` and `agglomerative` are implemented.")
            raise NotImplementedError(
                "Only `hdbscan`, `optics`, and `agglomerative` are implemented."
            )

    def top_ngram_embeddings(
        self, top_n: int = 5, min_df: Union[int, float] = 2
    ) -> Union[tuple, None]:
        """Returns ngrams by cluster using tfidf and cosine similarity."""

        vocabulary = list(set(self.corpus))
        labels = np.sort(np.unique([label for label in self.labels if label > -1]))
        results = []

        try:
            vectorizer = CountVectorizer(
                stop_words="english",
                ngram_range=(1, 4),
                min_df=min_df,
                vocabulary=vocabulary,
            )
            c_vectorizer = ClassTfidfTransformer(
                bm25_weighting=True, reduce_frequent_words=True
            )
            feature_names = vectorizer.get_feature_names_out()

            docs = []
            for label in labels:
                idx = np.where(self.labels == label)[0]
                docs.append(" ".join(self.corpus[idx]))

            x1 = vectorizer.fit_transform(docs)
            x2 = c_vectorizer.fit(x1).transform(x1)

            for ldx, label in enumerate(labels):
                tfidf_scores = x2[ldx].toarray().flatten()
                df = pd.DataFrame(
                    list(zip(feature_names, tfidf_scores)),
                    columns=["feature", "tfidf_score"],
                )

                tfidf_features = df[df["tfidf_score"] > 0]["feature"].tolist()

                if len(tfidf_features) == 0:
                    results.append(
                        {
                            "features": ["<no label found>"],
                            "embedding": np.zeros(self.embedding_size),
                        }
                    )
                    continue

                # remove features with zero frequency
                df = df[df["tfidf_score"] > 0].copy()

                cluster_features = df["feature"].tolist()
                embeddings = self.get_embeddings(cluster_features)

                centroid = np.mean(embeddings, axis=0)  # centroid of cluster

                # get cosine similarity of each feature to centroid using iloc
                df.loc[:, "sim_score"] = cosine_similarity(
                    embeddings, centroid.reshape(1, -1)
                ).flatten()

                df.loc[:, "score"] = df["tfidf_score"] * df["sim_score"]

                # sort by score
                df = df.sort_values(by=["score"], ascending=False).reset_index(
                    drop=True
                )

                # get top n features
                top_features = df["feature"].tolist()[:top_n]

                # get embeddings for top n features
                top_features_embedding = np.mean(
                    self.get_embeddings(top_features), axis=0
                )

                results.append(
                    {"features": top_features, "embedding": top_features_embedding}
                )

        except ValueError as e:
            logger.error(
                "There was an error in finding top word embeddings: {}".format(str(e))
            )

        if len(results) > 0:
            return (labels, results)

        return None

    def get_text_label_mapping(self, top_n: int = 5) -> dict:
        """Finds the closest n-gram to a clusters centroid.
        Returns a dict to be used to map these to labels."""

        labels, results = self.top_ngram_embeddings(top_n=top_n)

        text_labels = [", ".join(r["features"]) for r in results]
        text_label_embeddings = np.asarray([r["embedding"] for r in results])

        # If a list of categories is given, use those
        if isinstance(self.cluster_categories, list):
            # Make sure we have a unique list of categories
            categories = list(set(self.cluster_categories))

            category_embeddings = self.get_embeddings(categories)
            neigh = NearestNeighbors(n_neighbors=1)
            neigh.fit(category_embeddings)

            # Update text_labels with closest category
            for i, label in enumerate(labels):
                top_text_idx = neigh.kneighbors(np.array([text_label_embeddings[i]]))[
                    1
                ].flatten()
                text_labels[i] = categories[top_text_idx[0]]

        mapping = {-1: "<outliers>"}

        for i, label in enumerate(labels):
            mapping[label] = (
                text_labels[i] if i < len(text_labels) else "<no label found>"
            )

        return mapping

    def get_llm_description(self, text_label: str) -> str:
        """Gets the description of a cluster using LLM"""

        # Get prompt
        prompt = PROMPT_TEMPLATE_CLUSTER.format(samples=text_label)

        explanation = get_openai_response_chat(
            prompt,
            model=settings.CLUSTER_DESCRIPTION_MODEL,
            system_message="You are an expert at understanding the intent of Google searches.",
        )

        return explanation

    def get_text_label_mapping_llm(self, top_n: int = 5) -> dict:
        """Gets explanations for each cluster using OpenAI LLM"""

        mapping = self.get_text_label_mapping(top_n=top_n)

        # Use multi-threading to get explanations and add to mapping at correct label key
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=settings.MAX_WORKERS
        ) as executor:
            futures = {
                executor.submit(self.get_llm_description, text_label): label
                for label, text_label in mapping.items()
            }

            for future in tqdm(
                concurrent.futures.as_completed(futures),
                desc="Getting LLM explanations",
                total=len(futures),
            ):
                label = futures[future]
                explanation = future.result()
                mapping[label] = explanation

        return mapping

    def cluster_centroid_deduction(self) -> np.ndarray:
        """Finds the centroids, or central point in space,
        for each cluster of texts."""

        centroids = []

        labels = np.sort(np.unique(self.labels))

        for label in labels:
            idx = np.where(self.labels == label)[0]
            centroid = np.mean(self.embeddings[idx], axis=0)
            centroids.append(centroid)

        return labels, np.array(centroids)

    def fish_additional_outliers(self) -> None:
        """Finds additional labels in the outliers getting the closest centroid
        and only returning ones with a good sillouhette score"""

        outliers_idx = np.where(self.labels == -1)[0]
        outlier_embeddings = self.embeddings[outliers_idx]
        labels, centroids = self.cluster_centroid_deduction()

        neigh = NearestNeighbors(n_neighbors=1)
        neigh.fit(centroids)
        assigned_idx = neigh.kneighbors(outlier_embeddings)[1].flatten()

        def map_to_label(x):
            return labels[x]

        assigned = map_to_label(assigned_idx)

        if len(np.unique(assigned)) > 1:
            scores = silhouette_samples(outlier_embeddings, assigned)
            matched = np.where(scores > 0)[0]
            not_matched = np.where(scores <= 0)[0]

            self.labels[outliers_idx[matched]] = assigned[matched]
            self.labels[outliers_idx[not_matched]] = assigned[not_matched]

        return None

    def recluster_outliers(self) -> None:
        """This uses agglomerative clustering to recluster any remaining
        outliers.  One negative is that agglomerative doesn't produce
        outliers and some tokens (e.g. mispellings) SHOULD be outliers."""

        outliers_idx = np.where(self.labels == -1)[0]
        model = self.get_cluster_model(model_name="agglomerative")
        labels_idx = model.fit(self.model_data[outliers_idx]).labels_
        offset = self.labels.max() + 1
        labels_idx = np.array([idx + offset for idx in labels_idx])
        self.labels[outliers_idx] = labels_idx

    def fit_pairwise_crossencoded(
        self,
        corpus: List[str],
        categories: Union[List[str], None] = None,
        top_n: int = 5,
        percentile_threshold: int = 50,
        std_dev_threshold: float = 0.1,
    ) -> tuple:
        """Fits the model first pairwise using cosine_similarity and then using cross-encoder to top n categories

        Args:
            corpus (List[str]): A list of sentences to cluster.
            categories (Union[List[str], None], optional): A list of categories to use for clustering. Defaults to None.
            top_n (int, optional): The number of categories to cross-encode. Defaults to 5.
            percentile_threshold (int, optional): The percentile threshold to use for good matches. Defaults to 50.
            std_dev_threshold (float, optional): The standard deviation threshold to use for good matches. Defaults to 0.1.

        Returns:
            tuple: A tuple of the cluster labels and text labels.
        """

        corpus_array = np.array(list(set(corpus)))

        logger.info("Getting embeddings.")
        if self.embeddings is None:
            embeddings = self.get_embeddings(corpus_array)

        if categories:
            self.cluster_categories = categories

        if not self.cluster_categories:
            raise ValueError(
                "You must provide a list of cluster categories upon class initiation to use this method."
            )

        category_embeddings = self.get_embeddings(self.cluster_categories)

        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        category_embeddings = category_embeddings / np.linalg.norm(
            category_embeddings, axis=1, keepdims=True
        )

        logger.info("Getting pairwise cosine similarity.")
        cosine_similarity_matrix = cosine_similarity(embeddings, category_embeddings)

        logger.info("Getting cross-encoder similarity.")
        cross_encoder = CrossEncoder(settings.CROSSENCODER_MODEL_NAME)

        top_n_categories = [
            [
                self.cluster_categories[x]
                for x in np.argsort(cosine_similarity_matrix[i])[-top_n:][::-1]
            ]
            for i in range(len(corpus_array))
        ]

        cross_encoder_pairs = [
            list(zip([corpus_array[i]] * len(top_n_categories[i]), top_n_categories[i]))
            for i in range(len(corpus_array))
        ]

        cross_encoder_similarity = [
            cross_encoder.predict(pairs).flatten()
            for pairs in tqdm(
                cross_encoder_pairs, desc="Getting cross-encoder similarity"
            )
        ]

        similarities_flattened = np.array(cross_encoder_similarity).flatten()
        similarity_threshold = np.percentile(
            similarities_flattened, percentile_threshold
        )

        logger.info(f"Similarity threshold: {similarity_threshold}")

        labels, text_labels = [], []
        for i, similarities in enumerate(cross_encoder_similarity):
            most_similar = np.argmax(similarities)
            most_similar_score = similarities[most_similar]
            std_dev = np.std(similarities)

            if (
                most_similar_score >= similarity_threshold
                and std_dev >= std_dev_threshold
            ):
                # Good result since there is a score that is in the top percentile and the std dev is higher than the threshold
                best_category = top_n_categories[i][most_similar]
                labels.append(self.cluster_categories.index(best_category))
                text_labels.append(best_category)
            else:
                labels.append(-1)
                text_labels.append("<outlier>")

        labels_match = []
        text_labels_match = []

        # Match back to full length corpus
        for i, item in enumerate(corpus):
            idx = np.where(corpus_array == item)[0][0]
            labels_match.append(labels[idx])
            text_labels_match.append(text_labels[idx])

        self.labels = labels_match
        self.text_labels = text_labels_match

        return (self.labels, self.text_labels)


    def fit_pairwise(
            self, corpus: List[str], categories: Union[List[str], None] = None,
            percentile_threshold: int = 50,
            std_dev_threshold: float = 0.1,
        ) -> tuple:
            """This is the main fitting function that does all the work.
    
            Args:
                corpus (List[str]): A list of sentences to cluster.
                categories (Union[List[str], None], optional): A list of categories to use for clustering.
                percentile_threshold (int, optional): The percentile threshold to use for good matches. Defaults to 50.
                std_dev_threshold (float, optional): The standard deviation threshold to use for good matches. Defaults to 0.1.
    
            Returns:
                tuple: A tuple of the cluster labels and text labels."""
    
            self.corpus = np.array(corpus)
    
            logger.info("Getting embeddings.")
            if self.embeddings is None:
                self.embeddings = self.get_embeddings(self.corpus)
    
            if categories:
                self.cluster_categories = categories
    
            if not self.cluster_categories:
                raise ValueError(
                    "You must provide a list of cluster categories upon class intitiation to use this method."
                )
    
            category_embeddings = self.get_embeddings(self.cluster_categories)
    
            # Normalize embeddings for cosine similarity
            self.embeddings = self.embeddings / np.linalg.norm(
                self.embeddings, axis=1, keepdims=True
            )
            category_embeddings = category_embeddings / np.linalg.norm(
                category_embeddings, axis=1, keepdims=True
            )
    
            logger.info("Getting pairwise cosine similarity.")
            cosine_similarity_matrix = cosine_similarity(
                self.embeddings, category_embeddings
            )
    
            similarity_threshold = np.percentile(
                cosine_similarity_matrix, percentile_threshold
            )
            logger.info(f"Similarity threshold: {similarity_threshold}")
    
            self.labels = []
            self.text_labels = []
            for i, similarities in enumerate(cosine_similarity_matrix):
                most_similar = np.argmax(similarities)
                most_similar_score = similarities[most_similar]

                # Get the standard deviation of the top 10 scores
                std_dev = np.std(np.sort(similarities)[-10:])
    
                if (
                    most_similar_score >= similarity_threshold
                    and std_dev >= std_dev_threshold
                ):
                    # Good result, as the score is in the top percentile, and the std dev is higher than the threshold
                    self.labels.append(most_similar)
                    self.text_labels.append(self.cluster_categories[most_similar])
                else:
                    self.labels.append(-1)
                    self.text_labels.append("<outlier>")
    
            return (self.labels, self.text_labels)
        

    def fit(self, corpus: List[str], top_n: int = 5) -> tuple:
        """This is the main fitting function that does all the work.

        Args:
            corpus (List[str]): A list of sentences to cluster.
            top_n (int, optional): The number of ngrams to use for cluster labels. Defaults to 5.

        Returns:
            tuple: A tuple of the cluster labels and text labels."""

        self.corpus = np.array(corpus)

        logger.info("Getting embeddings.")
        if self.embeddings is None:
            self.embeddings = self.get_embeddings(self.corpus)

        self.model = self.get_cluster_model()

        logger.info("Fitting model.")
        self.model_data = self.get_reduced(self.model_data)
        self.labels = self.model.fit(self.model_data).labels_
        logger.info(
            "Initial Model. Unique Labels: {}".format(len(np.unique(self.labels)))
        )

        if -1 in list(self.labels):
            logger.info("Running post processes for outliers.")
            self.fish_additional_outliers()
            if not self.keep_outliers:
                self.recluster_outliers()
            logger.info(
                "Post Processing. Unique Labels: {}".format(len(np.unique(self.labels)))
            )

        logger.info("Finding names for cluster labels.")

        if self.use_llm_descriptions:
            label_mapping = self.get_text_label_mapping_llm(top_n=top_n)
        else:
            label_mapping = self.get_text_label_mapping(top_n=top_n)

        self.text_labels = [label_mapping[label] for label in self.labels]

        return (self.labels, self.text_labels)
