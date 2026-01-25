"""
Security Configurations.

Contains configurations for secrets management and authentication.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["VaultConfig"]


@dataclass
class VaultConfig:
    """Configuration for HashiCorp Vault."""

    url: str
    token: str | None = None
    mount_point: str = "secret"
    path: str = "nglab"
