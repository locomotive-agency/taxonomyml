"""Search Console API wrapper."""

import re
from typing import Union

from google.oauth2 import service_account
from apiclient import discovery

import pandas as pd
from searchconsole.account import Account
from loguru import logger


import settings


credentials = service_account.Credentials.from_service_account_file(
    settings.SERVICE_ACCOUNT_CREDENTIALS,
    scopes=settings.SERVICE_ACCOUNT_SCOPES,
    subject=settings.SERVICE_ACCOUNT_SUBJECT,
)

service = discovery.build(
    serviceName="searchconsole",
    version="v1",
    credentials=credentials,
    cache_discovery=False,
)

gsc_client = Account(service, credentials)


def load_available_gsc_accounts() -> pd.DataFrame:
    """Load GSC accounts into a pandas dataframe."""
    accounts = [wp.url for wp in gsc_client.webproperties]
    df = pd.DataFrame(accounts, columns=["property"])
    df["property_domain"] = df["property"].apply(
        lambda x: re.sub(r"(https?://(www\.)?|sc-domain:)", "", x)
    )
    df["type"] = "gsc"
    df["table"] = None

    return df


def load_gsc_account_data(property: str, days: int) -> Union[None, pd.DataFrame]:
    """Load GSC data into a pandas dataframe."""

    try:
        webproperty = gsc_client[property]
        logger.info("Creating dataframe...")
        df = (
            webproperty.query.range("today", days=-days)
            .dimension("query", "page")
            .get()
            .to_dataframe()
        )
        logger.info("Dataframe created.")
        return df

    except Exception as e:
        logger.error(f"There was an error loading the data for {property}.")
        return None


def load_gsc_from_agent(query: str):
    query = re.sub(r"[^a-zA-Z0-9\:\/\,\.\-]", "", query)

    query_parts = [p.strip() for p in query.split(",")]

    if len(query_parts) == 2:
        property, days = query_parts
        days = int(days)

    else:
        return "Error: Invalid query. Please use the following format: property, days"

    df = load_gsc_account_data(property, days)

    return df
