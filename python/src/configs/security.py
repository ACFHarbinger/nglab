"""
Security Configurations.

Contains configurations for secrets management and authentication.
"""

from dataclasses import dataclass


@dataclass
class VaultConfig:
    """Configuration for HashiCorp Vault."""

    url: str
    token: str | None = None
    mount_point: str = "secret"
    path: str = "nglab"
