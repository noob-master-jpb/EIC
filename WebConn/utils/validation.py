from __future__ import annotations

import ipaddress
import socket
from typing import Iterable
from urllib.parse import urlparse

from pydantic import BaseModel, Field, field_validator


class URLInput(BaseModel):
    url: str = Field(min_length=1, max_length=2048)

    @field_validator("url")
    @classmethod
    def _validate_url(cls, v: str) -> str:
        parsed = urlparse(v.strip())
        if parsed.scheme not in {"http", "https"}:
            raise ValueError("Only http and https URLs are allowed")
        if not parsed.netloc:
            raise ValueError("URL must include a host")
        return v.strip()


class SearchInput(BaseModel):
    query: str = Field(min_length=1, max_length=512)


def _is_private_ip(ip: str) -> bool:
    try:
        addr = ipaddress.ip_address(ip)
    except ValueError:
        return False
    return any(
        (
            addr.is_private,
            addr.is_loopback,
            addr.is_link_local,
            addr.is_reserved,
            addr.is_multicast,
            addr.is_unspecified,
        )
    )


def _resolve_host_ips(hostname: str) -> Iterable[str]:
    infos = socket.getaddrinfo(hostname, None)
    for info in infos:
        yield info[4][0]


def is_host_allowed(hostname: str, allow_internal: bool = False) -> bool:
    lowered = hostname.lower().strip("[]")
    blocked_hosts = {"localhost", "localhost.localdomain"}
    if lowered in blocked_hosts:
        return allow_internal

    try:
        if _is_private_ip(lowered):
            return allow_internal
        return True
    except Exception:
        pass

    try:
        for ip in _resolve_host_ips(lowered):
            if _is_private_ip(ip):
                return allow_internal
    except socket.gaierror:
        return False

    return True


def validate_public_url(url: str, allow_internal: bool = False) -> URLInput:
    parsed = URLInput(url=url)
    host = urlparse(parsed.url).hostname
    if not host:
        raise ValueError("Invalid host")
    if not is_host_allowed(host, allow_internal=allow_internal):
        raise ValueError("Target host is blocked by security policy")
    return parsed
