PROMPT_TEMPLATE_TAXONOMY = """
As an expert in taxonomy creation, we need your assistance in developing a clear, high-level website taxonomy based on a provided list of topics. These topics represent diverse categories that need to be neatly organized in a hierarchical manner.

Subject of website: {subject}

Important Topics:
{query_data}

The topics are a list of topic ngrams and their scores. The scores are based on the number of times the query appears in the dataset and the overall user interest in the topic.  Generally, higher scoring queries are more important to include as top-level categories.

Please adhere to the following dash-prefix format for your output:

- Category
  - Subcategory
    - Sub-subcategory
  - Subcategory
- Category
  - Subcategory
  ...

In order to effectively accomplish this task, you MUST follow the following guidelines:

* Brands: The Important Topics may mention these specific brands `{brands}`. When creating your taxonomy, please omit these brand terms. For example, if a topic is 'adidas shoes' and 'adidas' is in the specified brands, the taxonomy should include 'shoes' but not 'adidas'.
* Guessing: AVOID inventing or speculating any subcategory subjects that are not directly reflected in the provided Important Topics.
* Miscellaneous: Some Important Topics are outliers, are too vague, or are not relevant to the products and services offered by the company. Assign these topics to a top-level category called 'Miscellaneous'
* Depth of Taxonomy: The taxonomy should be no more than four levels deep (i.e., Category > Subcategory > Sub-subcategory > Sub-sub-subcategory) and should have only a few top-level categories. The taxonomy should be broad rather than deeply detailed. Important Topics should be categorized into subjects and subjects into cogent categories.
* Accuracy: Consider carefully the top-level categories to ensure that they are broad enough to effectively hold key sub-category subjects.
* Readability: Ensure that category names are concise yet descriptive.
* Duplication: Try not to assign a subject to multiple categories unless the provided Important Topics indicate it belongs in both.

Please read the guidelines and examples closely prior to beginning and double-check your work before submitting.

Begin!
"""

PROMPT_TEMPLATE_TAXONOMY_REVIEW = """As a master of taxonomy creation, we need your assistance in developing a clear, high-level website taxonomy. Another member of the team has already created a taxonomy, but we need your help to review it and adjust it as needed.

Here is the taxonomy that was created:
{taxonomy}

Please review the taxonomy and make any necessary changes. If you believe that the taxonomy is correct, please submit it as is.

Here are some guidelines for reviewing the taxonomy:
* Remove any Miscellaneous sub-categories that are already assigned to other categories. For example, if there is a category called 'Nike > Shoes > NEO Vulc' and another category called 'Miscellaneous > Neo Vulc', please remove the 'Miscellaneous > Neo Vulc' category.
* Make sure all category designations are accurate and appropriate.
* Ensure that categories are not duplicated. For example, if there is a category called 'Nike > Shoes > NEO Vulc' and another category called 'Nike > Shoes > NEO Vulc > NEO Vulc', please remove the 'Nike > Shoes > NEO Vulc > NEO Vulc' category.
* Review the category names for readability. Ensure that category names are concise yet descriptive.
* Only respond with the updated taxonomy.

Keep the formatting of the original taxonomy. The taxonomy should be structured as follows:
- Category
  - Subcategory
    - Sub-subcategory
  - Subcategory
  . . .
- Category

Please read the guidelines closely prior to beginning and double-check your work before submitting.

Begin!
"""


PROMPT_TEMPLATE_CLUSTER = """As an expert at understanding search intent, We need your help to provide the main subject being sought after in the following list of search queries. Please ONLY provide the subject and no other information. For example, if the search queries are 'adidas shoes, nike shoes, converse shoes', the subject is 'Name-brand shoes'.

Search Topics:
{samples}

Subject: """
