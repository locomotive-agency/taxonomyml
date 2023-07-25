class TaxonomyError(Exception):
    pass


# Google Search Console errors
class GoogleSearchConsoleError(TaxonomyError):
    pass


class PropertyNotFoundError(GoogleSearchConsoleError):
    def __init__(self, prop_url: str) -> None:
        super().__init__(f"No property was found with URL: {prop_url}")
