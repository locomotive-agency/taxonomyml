"""Common flows for obtaining OAuth credentials for Google APIs."""
from __future__ import annotations

import abc
import pathlib
import warnings

import google.auth.exceptions
import google.auth.transport.requests
from google.oauth2.credentials import Credentials
from google.oauth2.service_account import Credentials as ServiceCredentials
from google_auth_oauthlib.flow import Flow, InstalledAppFlow


class GoogleAuthManagerBase(abc.ABC):
    """Base class for all types of auth manager."""

    @abc.abstractmethod
    def authorize(self):
        pass


class GoogleOAuthManager(GoogleAuthManagerBase):
    """Manages Google OAuth credentials and builds the service for installed apps.

    Attributes:
        scopes:
            A list of scopes required for the authorization. A full list of
            available scopes can be found at:
            https://developers.google.com/identity/protocols/oauth2/scopes
        client_id: The OAuth client ID from Google.
        client_secret: The OAuth client secret from Google.
        redirect_uri: The URI to redirect to after auth is complete.
    """

    def __init__(
        self,
        scopes: list[str],
        client_id: str,
        client_secret: str,
        redirect_uri: str,
    ):
        self.scopes = scopes
        self.client_id = client_id
        self.client_secret = client_secret
        self.redirect_uri = redirect_uri
        self.app_config = {
            "installed": {
                "client_id": self.client_id,
                "client_secret": self.client_secret,
                "redirect_uris": [],
                "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                "token_uri": "https://accounts.google.com/o/oauth2/token",
            }
        }
        self.flow = Flow.from_client_config(
            self.app_config,
            scopes=self.scopes,
            redirect_uri=self.redirect_uri,
        )
        self._credentials = None

    @property
    def credentials(self) -> Credentials:
        return self._credentials

    @credentials.setter
    def credentials(self, creds: Credentials) -> None:
        self._credentials = creds

    def get_auth_url(self) -> tuple[str, str]:
        """Returns the authorization URL for the flow.

        Tuple includes the auth URL and the state.
        """
        return self.flow.authorization_url(
            prompt="consent", access_type="offline", include_granted_scopes=True
        )

    def authorize(self) -> Credentials:
        """Authorizes access to Google and returns credentials.

        Credentials can be serialized to a JSON file and reloaded as needed. A
        refresh will be attempted for expired credentials. If the refresh
        doesn't succeed, the authorization flow will be attempted again.

        Returns:
            A Credentials object that can be used with Google API client
            libraries.
        """
        try:
            creds = None
            if self.credentials is not None:
                creds = self.credentials

            if creds is None:
                auth_url, state = self.get_auth_url()

            if not creds.valid:
                refresh_credentials(creds)
        except google.auth.exceptions.RefreshError:
            creds = self._run_flow()

        self.credentials = creds
        return creds


class GoogleInstalledAuthManager(GoogleAuthManagerBase):
    """Manages Google OAuth credentials and builds the service for installed apps.

    Attributes:
        scopes:
            A list of scopes required for the authorization. A full list of
            available scopes can be found at:
            https://developers.google.com/identity/protocols/oauth2/scopes
        serialize:
            If True, will serialize the credentials to a JSON file. The save
            location is specified by the `token_path` attribute. Defaults
            to False.
        secrets_path: The path to the client secrets JSON file from Google.
        token_path:
            The path to the existing token details. This doubles as the save
            location for newly serialized credentials.
    """

    def __init__(
        self,
        scopes: list[str],
        serialize: bool = False,
        secrets_path: str | pathlib.Path = "credentials/secrets.json",
        token_path: str | pathlib.Path = "credentials/token.json",
    ):
        warnings.warn(
            f"{self.__class__.__name__} should no longer be used due "
            f"to the deprecation of OAuth code copying by Google. "
            f"You should switch to the `GoogleServiceAccManager` or "
            f"another auth method.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.scopes = scopes
        self.serialize = serialize
        self.secrets_path = secrets_path
        self.token_path = token_path
        self._credentials = None

    def __init_subclass__(cls, **kwargs):
        # This throws a deprecation warning on subclassing.
        warnings.warn(
            f"{cls.__name__} should no longer be used due to the "
            f"deprecation of OAuth code copying by Google. You "
            f"should switch to the `GoogleServiceAccManager` or "
            f"another auth method.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init_subclass__(**kwargs)

    @property
    def credentials(self) -> Credentials:
        return self._credentials

    @credentials.setter
    def credentials(self, creds: Credentials) -> None:
        self._credentials = creds

    def _run_flow(self) -> Credentials:
        """Runs the InstalledAppFlow to fetch credentials."""
        flow = InstalledAppFlow.from_client_secrets_file(
            str(self.secrets_path), self.scopes
        )
        creds = flow.run_console()
        return creds

    def authorize(self) -> Credentials:
        """Authorizes access to Google and returns credentials.

        Credentials can be serialized to a JSON file and reloaded as needed. A
        refresh will be attempted for expired credentials. If the refresh
        doesn't succeed, the authorization flow will be attempted again.

        Returns:
            A Credentials object that can be used with Google API client
            libraries.
        """
        try:
            if self.credentials is not None:
                creds = self.credentials
            else:
                creds = Credentials.from_authorized_user_file(str(self.token_path))

            if creds is None:
                creds = self._run_flow()

            if not creds.valid:
                refresh_credentials(creds)
        except (FileNotFoundError, google.auth.exceptions.RefreshError):
            creds = self._run_flow()

        if self.serialize:
            with open(self.token_path, "w") as f:
                f.write(creds.to_json())

        self.credentials = creds
        return creds


class GoogleServiceAccManager(GoogleAuthManagerBase):
    """Manages Google Service Account scopes and credentials.

    Attributes:
        scopes:
            A list of scopes required for the authorization. A full list of
            available scopes can be found at:
            https://developers.google.com/identity/protocols/oauth2/scopes
        credentials_path:
            Path to the JSON file used to authenticate the service account.
        subject:
            The user account to mock when accessing data.
    """

    def __init__(
        self,
        scopes: list[str],
        *,
        credentials_path: str | pathlib.Path | None = None,
        subject: str | None = None,
    ):
        self.scopes = scopes
        self.credentials_path = credentials_path
        self.subject = subject
        self._credentials: ServiceCredentials | None = None

    @property
    def credentials(self) -> ServiceCredentials:
        return self._credentials

    @credentials.setter
    def credentials(self, creds: ServiceCredentials) -> None:
        self._credentials = creds

    def authorize(self) -> ServiceCredentials:
        """Authorizes access to Google and returns credentials.

        Returns:
            A Credentials object that can be used with Google API client
            libraries.
        """
        if self.credentials is None:
            self.credentials = ServiceCredentials.from_service_account_file(
                self.credentials_path, scopes=self.scopes, subject=self.subject
            )
        return self.credentials


def refresh_credentials(credentials: Credentials) -> None:
    """Refreshes a Google Credentials object."""
    request = google.auth.transport.requests.Request()
    credentials.refresh(request)
