"""Keep model credentials outside shared configuration and startup logs."""

import os
from collections.abc import Mapping


MODEL_ENV = {
    "UNIGOAL_API_KEY": "api_key",
    "UNIGOAL_BASE_URL": "base_url",
    "UNIGOAL_LLM_MODEL": "llm_model",
    "UNIGOAL_VLM_MODEL": "vlm_model",
}


def apply_model_environment(config, environ=None):
    """Return a copy with nonempty environment values overriding model settings."""
    environ = os.environ if environ is None else environ
    result = dict(config)
    for env_name, config_name in MODEL_ENV.items():
        value = environ.get(env_name)
        if value:
            result[config_name] = value
    return result


def redact_config(config):
    """Return a log-safe copy; endpoints may also contain credentials or account IDs."""
    sensitive = ("api_key", "apikey", "password", "secret", "token", "base_url")
    if isinstance(config, Mapping):
        return {
            key: "[REDACTED]" if any(term in str(key).lower() for term in sensitive)
            else redact_config(value)
            for key, value in config.items()
        }
    if isinstance(config, (list, tuple)):
        return [redact_config(item) for item in config]
    return config
