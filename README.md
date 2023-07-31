# Auto Taxonomy Creation


## Example with CSV
```
from main import create_taxonomy

filename = "domain_data.csv"
brand_terms = ['brand', 'brand', 'brand']
website_subject = "This website is about X"

taxonomy, df, samples = create_taxonomy(filename,
                                        text_column = "keyword",
                                        search_volume_column = "search_volume",
                                        website_subject = website_subject,
                                        cross_encoded = True,
                                        min_df = 5,
                                        brand_terms = brand_terms)

df.to_csv("taxonomy.csv", index=False)

print("\n".join(taxonomy))
```

## Example with GSC
```
from main import create_taxonomy

brand_terms = ["brand"]
property = "sc-domain:domain.com"
website_subject = "This website is about X"

taxonomy, df, samples = create_taxonomy(property,
                                        days = 30,
                                        website_subject = website_subject,
                                        min_df = 2,
                                        brand_terms = brand_terms,
                                        limit_queries_per_page = 5)


df.to_csv("domain_taxonomy.csv", index=False)

df.head()
```



### Parameters
* `data` (Union[str, pd.DataFrame]): GSC Property, CSV Filename, or pandas dataframe.
* `text_column` (str, optional): Name of the column with the queries. Defaults to None.
* `search_volume_column` (str, optional): Name of the column with the search volume. Defaults to None.
* `website_subject` (str, required): Provides GPT with context about the website.
* `cluster_embeddings_model` (Union[str, None], optional): Name of the cluster embeddings model. Defaults to 'local'.
* `cross_encoded` (bool, optional): Whether to use cross encoded matching. Defaults to False. Improves matching, but takes longer.
* `days` (int, optional): Number of days back to pull data from Search Console. Defaults to 30.
* `ngram_range` (tuple, optional): Ngram range to use for scoring. Defaults to (1, 5).
* `min_df` (int, optional): Minimum document frequency to use for scoring. Defaults to 5.
* `brand_terms` (List[str], optional): List of brand terms to remove from queries. Defaults to None.
* `limit_queries_per_page` (int, optional): Number of queries per page to use for clustering. Defaults to 5.
* `debug_responses` (bool, optional): Whether to print debug responses. Defaults to False.
* `openai_api_key` (str | None, optional): OpenAI API key. Defaults to environment variable OPENAI_API_KEY if not provided.

