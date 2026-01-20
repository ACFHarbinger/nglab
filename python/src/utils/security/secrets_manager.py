"""
Secrets Management for NGLab.

Provides a unified interface for accessing secrets from multiple backends
(HashiCorp Vault, Environment Variables, Docker Secrets) with precedence rules.
"""

import logging
import os
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class VaultConfig:
    """Configuration for HashiCorp Vault."""

    url: str
    token: str | None = None
    mount_point: str = "secret"
    path: str = "nglab"


class SecretsManager:
    """
    Unified secrets manager.

    Priority:
    1. HashiCorp Vault (if configured and available)
    2. Environment Variables
    3. Docker Secrets (optional, typical path /run/secrets/)
    """

    def __init__(self, vault_config: VaultConfig | None = None):
        """Initialize secrets manager with optional Vault config."""
        self.vault_config = vault_config
        self._vault_client = None
        self._vault_cache: dict[str, str] = {}

        if self.vault_config:
            self._init_vault()

    def _init_vault(self) -> None:
        """Initialize Vault client."""
        # Ensure vault_config is present before access
        if not self.vault_config:
            return

        try:
            import hvac  # type: ignore

            self._vault_client = hvac.Client(
                url=self.vault_config.url, token=self.vault_config.token
            )
            if self._vault_client.is_authenticated():
                logger.info(f"Connected to Vault at {self.vault_config.url}")
            else:
                logger.warning("Vault client initialized but not authenticated")
        except ImportError:
            logger.warning("hvac not installed. Vault support disabled.")
        except Exception as e:
            logger.error(f"Failed to connect to Vault: {e}")

    def get_secret(self, key: str, default: str | None = None) -> str | None:
        """
        Get a secret by key.

        Args:
            key: Secret key name (e.g. "DATABASE_URL").
            default: Default value if not found.

        Returns:
            Secret value or default.
        """
        # 1. Try Vault
        if self._vault_client:
            # Check cache first
            if key in self._vault_cache:
                return self._vault_cache[key]

            try:
                # Vault kv v2 read
                if self.vault_config:
                    response = self._vault_client.secrets.kv.v2.read_secret_version(
                        path=self.vault_config.path,
                        mount_point=self.vault_config.mount_point,
                    )
                    data = response["data"]["data"]
                    if key in data:
                        self._vault_cache[key] = data[key]
                        return data[key]
            except Exception as e:
                logger.debug(f"Secret {key} not found in Vault or error: {e}")

        # 2. Try Environment Variable (most common)
        if key in os.environ:
            return os.environ[key]

        # 3. Try Docker Secret (pseudo-implementation)
        # docker_secret_path = f"/run/secrets/{key.lower()}"
        # if os.path.exists(docker_secret_path):
        #     with open(docker_secret_path, 'r') as f:
        #         return f.read().strip()

        return default
