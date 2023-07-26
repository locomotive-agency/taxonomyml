"""Interfaces for interacting with the Google Search Console API."""
from __future__ import annotations

import pathlib
from collections.abc import Sequence
from typing import Optional

import googleapiclient.discovery
import pandas as pd

import taxonomyml.lib.gauth as auth
from taxonomyml.lib import utils


class GoogleSearchConsole:
    def __init__(self, auth_manager: auth.GoogleAuthManagerBase):
        self.auth_manager = auth_manager

    @property
    def service(self) -> googleapiclient.discovery.Resource:
        """Builds the discovery document for Google Search Console.

        This is a simple facade for the Google API client discovery builder. For
        full details, refer to the following documentation:
        https://googleapis.github.io/google-api-python-client/docs/epy/googleapiclient.discovery-module.html#build
        """
        return googleapiclient.discovery.build(
            "searchconsole",
            "v1",
            credentials=self.auth_manager.authorize(),
            cache_discovery=False,
        )

    @property
    def properties(self) -> list["Property"]:
        raw_properties = self.service.sites().list().execute().get("siteEntry", [])
        return [Property(self, raw) for raw in raw_properties]

    def find_property(self, prop_url: str) -> "Property":
        """Retrieves a property from the URL string.

        Args:
            prop_url:
                The URL string of the property. Domain properties must start
                with "sc-domain:".

        Returns:
            An initialized Property object.
        """
        try:
            return [prop for prop in self.properties if prop.url == prop_url][0]
        except IndexError as e:
            raise PropertyNotFoundError(prop_url) from e


class Property:
    permission_levels = {
        "siteFullUser": 1,
        "siteOwner": 2,
        "siteRestrictedUser": 3,
        "siteUnverifiedUser": 4,
    }

    def __init__(self, gsc: GoogleSearchConsole, raw: dict):
        self.gsc = gsc
        self.raw = raw
        self.url = raw["siteUrl"]
        self.permission = raw["permissionLevel"]
        self.query = AnalyticsQuery(self.gsc, self)

    def __repr__(self):
        return f"<{type(self).__name__}(url='{self.url}')>"


class AnalyticsQuery:
    def __init__(self, gsc: GoogleSearchConsole, prop: Property):
        self.gsc = gsc
        self.prop = prop

        # Set up some sensible defaults for the query body
        default_dates = utils.DateRange.from_past_days(90, -3)
        self.raw = {
            "startDate": default_dates.start.isoformat(),
            "endDate": default_dates.end.isoformat(),
            "dimensions": ["query", "date", "page", "device", "country"],
            "startRow": 0,
            "rowLimit": 25000,
        }

    def date_range(self, date_range: utils.DateRange) -> "AnalyticsQuery":
        """Return a new query for metrics within a given date range.

        Args:
            date_range: The date range for the query.

        Returns:
            An updated AnalyticsQuery object.
        """
        self.raw.update(
            {
                "startDate": date_range.start.isoformat(),
                "endDate": date_range.end.isoformat(),
            }
        )

        return self

    def search_type(self, search_type: str) -> "AnalyticsQuery":
        """Return a new query for the specified search type.

        Args:
            search_type:
                The search type you would like to report on. Possible values:
                web (default), image, video, discover, googleNews.

        Returns:
            An updated AnalyticsQuery object.
        """
        allowed = {
            "web",
            "image",
            "video",
            "discover",
            "googleNews",
        }
        if search_type in allowed:
            self.raw["type"] = search_type

        return self

    def dimensions(self, dimensions: Sequence) -> "AnalyticsQuery":
        """Return a new query that fetches the specified dimensions.

        Args:
            dimensions:
                Dimensions you would like to report on. Possible values:
                country, device, page, query, searchAppearance.

        Returns:
            An updated AnalyticsQuery object.
        """
        allowed = {
            "query",
            "date",
            "page",
            "device",
            "country",
            "searchAppearance",
        }
        dimensions = list(dimensions)
        dimensions = [dim for dim in dimensions if dim in allowed]
        if dimensions:
            self.raw["dimensions"] = dimensions

        return self

    def filter(
        self,
        dimension: str,
        expression: str,
        operator: str = "equals",
        group_type: str = "and",
    ) -> "AnalyticsQuery":
        """Return a new query that filters rows by the specified filter.

        Args:
            dimension: Dimension you would like to filter on.
            expression: The value you would like to filter.
            operator:
                The operator you would like to use to filter. Possible values:
                equals, contains, notContains, includingRegex, excludingRegex.
            group_type:
                The way in which you would like multiple filters to combine.
                Note: currently only 'and' is supported by the API.

        Returns:
            An updated AnalyticsQuery object.
        """

        dimension_filter = {
            "dimension": dimension,
            "expression": expression,
            "operator": operator,
        }

        filter_group = {"groupType": group_type, "filters": [dimension_filter]}

        self.raw.setdefault("dimensionFilterGroups", []).append(filter_group)

        return self

    def execute(self, body: dict = None) -> list:
        """Executes a request against Google Search Console's search analytics.

        Returns:
            A list of response rows.
        """
        response = (
            self.gsc.service.searchanalytics()
            .query(siteUrl=self.prop.url, body=(body or self.raw))
            .execute()
        )
        return response.get("rows", [])

    def get(self, max_rows: Optional[int] = None) -> "AnalyticsReport":
        all_rows = []
        query = self.raw.copy()
        query["startRow"] = 0
        while True:
            if max_rows:
                if len(all_rows) >= max_rows:
                    all_rows = all_rows[:max_rows]
                    break
            rows = self.execute(body=query)
            if not rows:
                break
            all_rows.extend(rows)
            if len(rows) < query["rowLimit"]:
                break
            query["startRow"] += query["rowLimit"]
        return AnalyticsReport(all_rows, self)

    def get_by_day(self):
        report = None
        temp_query = self.raw.copy()
        date_range = utils.DateRange(self.raw["startDate"], self.raw["endDate"])

        for single_date in date_range:
            temp_query.update(
                {
                    "startDate": single_date.isoformat(),
                    "endDate": single_date.isoformat(),
                }
            )
            if report:
                report.extend(self.execute(body=temp_query))
            else:
                report = self.execute(body=temp_query)

        return AnalyticsReport(report, self)


class AnalyticsReport:
    def __init__(self, raw: list[dict], query: AnalyticsQuery):
        self.raw = raw
        self.query = query
        self.date_range = utils.DateRange(query.raw["startDate"], query.raw["endDate"])
        self.data = self._build_report()

    def _build_report(self) -> pd.DataFrame:
        built_rows = []
        for row in self.raw:
            row_copy = row.copy()
            dimensions = dict(
                zip(self.query.raw["dimensions"], row_copy.pop("keys", []))
            )
            built_rows.append({**dimensions, **row_copy})

        return pd.DataFrame(built_rows)

    def limit(
        self,
        metric: str,
        *,
        maximum: int | float | None = None,
        minimum: int | float | None = None,
    ) -> AnalyticsReport:
        """Applies a maximum and/or minimum limit to a metric column.

        Args:
            metric: The name of the metric column to affect.
            maximum:
                The maximum value to include in the column. This will be
                compared using the <= operator.
            minimum:
                The minimum value to include in the column. This will be
                compared using the >= operator.

        Returns:
            The updated AnomalyReport.
        """
        if maximum is None and minimum is None:
            raise ValueError(
                "At least one of `maximum` or `minimum` " "must be provided."
            )
        if maximum is not None:
            self.data = self.data[self.data[metric] <= maximum]
        if minimum is not None:
            self.data = self.data[self.data[metric] >= minimum]
        return self

    def to_csv(self, file_path: str | pathlib.Path) -> None:
        self.data.to_csv(file_path, index=False)

    def to_dataframe(self) -> pd.DataFrame:
        return self.data.copy()

    def _repr_html_(self):
        return self.data._repr_html_()

    def __repr__(self):
        return self.data.__repr__()


################################################################################
# EXCEPTIONS
################################################################################


class GoogleSearchConsoleError(Exception):
    pass


class PropertyNotFoundError(GoogleSearchConsoleError):
    def __init__(self, prop_url: str) -> None:
        self.prop = prop_url
        super().__init__(f"No property was found with URL: {self.prop}")
