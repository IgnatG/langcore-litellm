# CHANGELOG

<!-- version list -->

## v1.0.5 (2026-02-26)

### Features

- **Rate-limit retry with exponential back-off**: Both `infer()` and `async_infer()`
  now automatically retry on `RateLimitError` (HTTP 429) with exponential back-off
  (default: 4 retries, 2s base delay doubling each attempt: 2→4→8→16s).
  Previously, rate-limit errors were immediately returned as `score=0.0` failures,
  causing cascading validation failures. The inner retry loop waits before
  re-attempting, giving the API rate-limit window time to reset.

- **Configurable retry parameters**: New constructor kwargs `rate_limit_retries`
  (default: 4) and `rate_limit_base_delay` (default: 2.0s) allow per-provider
  tuning of back-off behaviour. These keys are consumed internally and never
  forwarded to `litellm.completion()` / `litellm.acompletion()`.

## v1.0.4 (2026-02-25)

### Bug Fixes

- Update litellm dependency to version 1.81.15 in project files
  ([`38bf594`](https://github.com/IgnatG/langcore-litellm/commit/38bf59448d6781e7eabe00aa5be1c6d970c82d54))

### Chores

- Update version to 1.0.3 and enhance changelog with new error handling
  ([`adef5f4`](https://github.com/IgnatG/langcore-litellm/commit/adef5f470a261f4f4163f598c7176a100c06229d))

## v1.0.3 (2026-02-25)

### Bug Fixes

- Catch `AuthenticationError`, `RateLimitError`, `BadRequestError`, `NotFoundError`,
  and `PermissionDeniedError` from LiteLLM as clean warnings instead of logging
  full stack traces via the generic `Exception` handler

## v1.0.2 (2026-02-23)

### Bug Fixes

- Update Python version in CI and refine langcore dependency in project files
  ([`a361342`](https://github.com/IgnatG/langcore-litellm/commit/a361342a8002ef11590f101943f74cb49d5ecad9))

## v1.0.1 (2026-02-23)

### Bug Fixes

- Update package names and versions in uv.lock
  ([`17b924e`](https://github.com/IgnatG/langcore-litellm/commit/17b924ee85be65b9b7a320b10260881442518a63))

## v1.0.0 (2026-02-23)

- Initial Release
