# CHANGELOG

<!-- version list -->

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
