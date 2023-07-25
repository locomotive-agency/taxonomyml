"""Search Console API utilities."""

import pandas as pd
from loguru import logger

from taxonomyml import exceptions
from taxonomyml.lib import gsc
from taxonomyml.lib.utils import DateRange


def load_gsc_account_data(
    gsc_client: gsc.GoogleSearchConsole, prop_url: str, days: int
) -> pd.DataFrame:
    """Load GSC data into a pandas dataframe."""
    try:
        prop = gsc_client.find_property(prop_url)
    except gsc.PropertyNotFoundError as e:
        raise exceptions.PropertyNotFoundError(prop_url) from e

    logger.info("Fetching data from GSC for: {}.", prop_url)
    dr = DateRange.from_past_days(days, offset=-2)
    try:
        report = prop.query.date_range(dr).dimensions(["query", "page"]).get()
    except Exception:
        logger.exception(f"Error loading data for {prop_url}.")
        raise

    logger.info("Successfully got data from GSC.")
    return report.to_dataframe()
