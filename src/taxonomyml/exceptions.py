class TaxonomyError(Exception):
    pass


# Google Search Console errors
class GoogleSearchConsoleError(TaxonomyError):
    pass


class PropertyNotFoundError(GoogleSearchConsoleError):
    def __init__(self, prop_url: str) -> None:
        super().__init__(f"No property was found with URL: {prop_url}")


class APIError(Exception):
    """Base class for API errors."""

    pass


class OpenAIError(APIError):
    """Error for OpenAI API."""

    pass
