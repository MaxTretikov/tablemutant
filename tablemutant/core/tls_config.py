#!/usr/bin/env python3
"""
TLS Configuration Module - Centralized TLS/SSL setup and HTTP client wrappers
Provides unified TLS configuration and HTTP client wrappers for all network requests
"""

import os
import sys
import ssl
import logging
import asyncio
from typing import Optional, Dict, Any, Union
import urllib.request
import urllib.parse

logger = logging.getLogger('tablemutant.core.tls_config')


class TLSConfig:
    """Centralized TLS/SSL configuration manager."""
    
    _instance = None
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not self._initialized:
            self._ssl_context, self._tls_source = self._init_tls()
            self._log_tls_info()
            TLSConfig._initialized = True
    
    def _init_tls(self):
        """Set up TLS verification; prefer system trust; otherwise certifi."""
        no_verify = str(os.environ.get("TABLEMUTANT_SSL_NO_VERIFY", "")).lower() in {"1", "true", "yes"}

        # Try system trust store injection first
        try:
            import truststore  # type: ignore
            truststore.inject_into_ssl()
            if no_verify:
                ctx = ssl._create_unverified_context()
                return ctx, "truststore + unverified"
            # Return None to signal callers to use default contexts that now point to system trust
            return None, "truststore"
        except Exception as e:
            logger.debug("truststore injection not used: %r", e)

        # Fallback to certifi bundle
        try:
            import certifi  # type: ignore
            cafile = certifi.where()
            
            # Help any other libs that read these env vars
            os.environ.setdefault("SSL_CERT_FILE", cafile)
            os.environ.setdefault("REQUESTS_CA_BUNDLE", cafile)
            
            # Patch ssl.create_default_context so libs like aiohttp pick up certifi by default
            self._patch_ssl_default_context(cafile)
            
            if no_verify:
                ctx = ssl._create_unverified_context()
                return ctx, f"certifi({cafile}) + unverified"
            ctx = ssl.create_default_context(cafile=cafile)
            return ctx, f"certifi({cafile})"
        except Exception as e:
            logger.debug("certifi not available; falling back to stdlib defaults: %r", e)

        # Last resort; may fail on some bundled interpreters if no CA path is compiled in
        if no_verify:
            ctx = ssl._create_unverified_context()
            return ctx, "stdlib + unverified"
        ctx = ssl.create_default_context()
        return ctx, "stdlib"
    
    def _patch_ssl_default_context(self, cafile: str):
        """Patch ssl.create_default_context to use certifi by default."""
        _orig_create_default_context = ssl.create_default_context

        def _patched_create_default_context(purpose=ssl.Purpose.SERVER_AUTH, cafile=None, capath=None, cadata=None):
            # If no specific parameters are provided, use our certifi cafile
            if cafile is None and capath is None and cadata is None:
                return _orig_create_default_context(purpose, cafile=cafile)
            # Otherwise, use the provided parameters as-is
            return _orig_create_default_context(purpose, cafile=cafile, capath=capath, cadata=cadata)

        # Only patch once
        if getattr(ssl.create_default_context, "__name__", "") != "_patched_create_default_context":
            ssl.create_default_context = _patched_create_default_context  # type: ignore
    
    def _log_tls_info(self):
        """Log TLS configuration information."""
        try:
            dvp = ssl.get_default_verify_paths()
            logger.debug("SSL default verify paths: cafile=%s capath=%s openssl_cafile_env=%s openssl_capath_env=%s",
                         getattr(dvp, "cafile", None), getattr(dvp, "capath", None),
                         getattr(dvp, "openssl_cafile_env", None), getattr(dvp, "openssl_capath_env", None))
        except Exception:
            pass
        logger.info("TLS source in use: %s", self._tls_source)
    
    @property
    def ssl_context(self) -> Optional[ssl.SSLContext]:
        """Get the configured SSL context."""
        return self._ssl_context
    
    @property
    def tls_source(self) -> str:
        """Get the TLS source description."""
        return self._tls_source
    
    @staticmethod
    def is_localhost_url(url: str) -> bool:
        """Check if a URL points to localhost/local addresses."""
        try:
            parsed = urllib.parse.urlparse(url)
            host = (parsed.hostname or '').lower()
            return host in ('localhost', '127.0.0.1', '::1', '')
        except Exception:
            return False
    
    def get_ssl_context_for_url(self, url: str) -> Optional[ssl.SSLContext]:
        """Get appropriate SSL context for a URL, skipping verification for localhost."""
        if self.is_localhost_url(url):
            # For localhost, don't use SSL verification
            return None
        return self._ssl_context


class HTTPClient:
    """Unified HTTP client wrapper for urllib requests."""
    
    def __init__(self):
        self.tls_config = TLSConfig()
    
    def urlopen(self, url: str, timeout: Optional[float] = None, headers: Optional[Dict[str, str]] = None, data: Optional[bytes] = None):
        """Wrapper around urllib.request.urlopen that always applies our TLS context and UA."""
        if headers is None:
            headers = {}
        headers.setdefault("User-Agent", f"TableMutant/1 Python/{sys.version_info[0]}.{sys.version_info[1]}")
        
        req = urllib.request.Request(url, headers=headers, data=data)
        # Use appropriate SSL context based on URL (None for localhost)
        ssl_context = self.tls_config.get_ssl_context_for_url(url)
        return urllib.request.urlopen(req, timeout=timeout, context=ssl_context)
    
    def request(self, url: str, method: str = "GET", timeout: Optional[float] = None,
                headers: Optional[Dict[str, str]] = None, data: Optional[bytes] = None):
        """Make an HTTP request with the configured TLS context."""
        if headers is None:
            headers = {}
        headers.setdefault("User-Agent", f"TableMutant/1 Python/{sys.version_info[0]}.{sys.version_info[1]}")
        
        req = urllib.request.Request(url, headers=headers, data=data)
        req.get_method = lambda: method
        # Use appropriate SSL context based on URL (None for localhost)
        ssl_context = self.tls_config.get_ssl_context_for_url(url)
        return urllib.request.urlopen(req, timeout=timeout, context=ssl_context)


class AsyncHTTPClient:
    """Unified async HTTP client wrapper for aiohttp requests."""
    
    def __init__(self):
        self.tls_config = TLSConfig()
    
    def get_ssl_context(self, url: str = None) -> Optional[ssl.SSLContext]:
        """Get the SSL context for aiohttp, with optional URL-based localhost detection."""
        # If URL is provided and it's localhost, return False to disable SSL verification
        if url and self.tls_config.is_localhost_url(url):
            return False  # aiohttp uses False to disable SSL verification
        
        # For aiohttp, we need to explicitly provide the SSL context
        # because aiohttp doesn't always pick up the patched ssl.create_default_context
        if self.tls_config.ssl_context is not None:
            return self.tls_config.ssl_context
        else:
            # If our TLS config returns None (truststore case), create a default context
            # that will use the system trust store
            return ssl.create_default_context()
    
    def get_connector(self, url: str = None, **kwargs) -> 'aiohttp.TCPConnector':
        """Get an aiohttp connector with proper SSL configuration."""
        try:
            import aiohttp
            ssl_context = self.get_ssl_context(url)
            # For localhost URLs, ssl_context will be False (disable SSL verification)
            # For other URLs, provide an SSL context
            if ssl_context is None:
                ssl_context = ssl.create_default_context()
            return aiohttp.TCPConnector(ssl=ssl_context, **kwargs)
        except ImportError:
            raise ImportError("aiohttp is required for async HTTP operations")
    
    def get_timeout(self, total: Optional[float] = None, connect: Optional[float] = None, 
                   sock_read: Optional[float] = None, sock_connect: Optional[float] = None) -> 'aiohttp.ClientTimeout':
        """Get an aiohttp timeout configuration."""
        try:
            import aiohttp
            return aiohttp.ClientTimeout(
                total=total,
                connect=connect,
                sock_read=sock_read,
                sock_connect=sock_connect
            )
        except ImportError:
            raise ImportError("aiohttp is required for async HTTP operations")
    
    def session(self, base_url: str = None, **kwargs) -> 'aiohttp.ClientSession':
        """Create an aiohttp session with proper SSL configuration."""
        try:
            import aiohttp
            # Use our connector if not provided
            if 'connector' not in kwargs:
                kwargs['connector'] = self.get_connector(base_url)
            
            # Add default headers if not provided
            if 'headers' not in kwargs:
                kwargs['headers'] = {}
            kwargs['headers'].setdefault("User-Agent", f"TableMutant/1 Python/{sys.version_info[0]}.{sys.version_info[1]}")
            
            return aiohttp.ClientSession(**kwargs)
        except ImportError:
            raise ImportError("aiohttp is required for async HTTP operations")


# Global instances for easy access
_http_client = None
_async_http_client = None


def get_http_client() -> HTTPClient:
    """Get the global HTTP client instance."""
    global _http_client
    if _http_client is None:
        _http_client = HTTPClient()
    return _http_client


def get_async_http_client() -> AsyncHTTPClient:
    """Get the global async HTTP client instance."""
    global _async_http_client
    if _async_http_client is None:
        _async_http_client = AsyncHTTPClient()
    return _async_http_client


# Convenience functions for direct use
def urlopen(url: str, timeout: Optional[float] = None, headers: Optional[Dict[str, str]] = None, data: Optional[bytes] = None):
    """Make a synchronous HTTP request with unified TLS configuration."""
    return get_http_client().urlopen(url, timeout, headers, data)


def request(url: str, method: str = "GET", timeout: Optional[float] = None, 
           headers: Optional[Dict[str, str]] = None, data: Optional[bytes] = None):
    """Make a synchronous HTTP request with unified TLS configuration."""
    return get_http_client().request(url, method, timeout, headers, data)


async def async_session(base_url: str = None, **kwargs) -> 'aiohttp.ClientSession':
    """Create an async HTTP session with unified TLS configuration."""
    return get_async_http_client().session(base_url=base_url, **kwargs)


def get_ssl_context() -> Optional[ssl.SSLContext]:
    """Get the configured SSL context."""
    return TLSConfig().ssl_context


def get_tls_source() -> str:
    """Get the TLS source description."""
    return TLSConfig().tls_source