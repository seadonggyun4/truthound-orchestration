"""Exercise Airflow's native credential precedence (CVE-2026-82355).

Run only in the secure Airflow host environment, independently of the adapter
suite's mocked Airflow modules. All identities and tokens are synthetic.
"""

from __future__ import annotations

import unittest
from unittest.mock import AsyncMock, patch

from airflow.api_fastapi.core_api import security
from fastapi import HTTPException
from fastapi.security import HTTPAuthorizationCredentials
from starlette.requests import Request


class CredentialPrecedenceTests(unittest.IsolatedAsyncioTestCase):
    def request(self) -> Request:
        request = Request({"type": "http", "headers": []})
        request.state.user = "cookie-principal"
        # Simulate the user cached by Airflow's JWT middleware. Older hosts
        # lack the sentinel and incorrectly trust state.user unconditionally.
        request.state.user_authenticated_via = getattr(
            security, "USER_INJECTED_BY_TRUSTED_MIDDLEWARE", None
        )
        return request

    async def test_explicit_bearer_wins_over_cached_cookie(self) -> None:
        request = self.request()
        resolver = AsyncMock(return_value="bearer-principal")
        with patch.object(security, "resolve_user_from_token", resolver):
            user = await security.get_user(
                request,
                None,
                HTTPAuthorizationCredentials(scheme="Bearer", credentials="synthetic"),
            )
        self.assertEqual(user, "bearer-principal")
        resolver.assert_awaited_once_with("synthetic")

    async def test_invalid_bearer_cannot_fall_back_to_cookie(self) -> None:
        resolver = AsyncMock(side_effect=HTTPException(status_code=403))
        with (
            patch.object(security, "resolve_user_from_token", resolver),
            self.assertRaises(HTTPException) as caught,
        ):
            await security.get_user(
                self.request(),
                None,
                HTTPAuthorizationCredentials(scheme="Bearer", credentials="invalid"),
            )
        self.assertEqual(caught.exception.status_code, 403)

    async def test_trusted_cookie_cache_without_bearer(self) -> None:
        resolver = AsyncMock()
        with patch.object(security, "resolve_user_from_token", resolver):
            user = await security.get_user(self.request(), None, None)
        self.assertEqual(user, "cookie-principal")
        resolver.assert_not_awaited()

    async def test_untrusted_state_does_not_bypass_validation(self) -> None:
        request = self.request()
        request.state.user_authenticated_via = object()
        resolver = AsyncMock(side_effect=HTTPException(status_code=401))
        with (
            patch.object(security, "resolve_user_from_token", resolver),
            self.assertRaises(HTTPException) as caught,
        ):
            await security.get_user(request, None, None)
        self.assertEqual(caught.exception.status_code, 401)
        resolver.assert_awaited_once_with(None)


if __name__ == "__main__":
    unittest.main()
