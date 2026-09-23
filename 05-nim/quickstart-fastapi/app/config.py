"""Settings. The NGC/NVIDIA key is read from the environment — never hard-coded."""

import json
import pprint
from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

QUICKSTART_DIR = Path(__file__).resolve().parents[1]
REPO_ENV = QUICKSTART_DIR.parents[1] / "00-environment" / ".env"
LOCAL_ENV = QUICKSTART_DIR / ".env"


class Settings(BaseSettings):
    # Later files win, so a local .env overrides the shared repo one.
    model_config = SettingsConfigDict(
        env_file=(REPO_ENV, LOCAL_ENV),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    nvidia_api_key: str | None = None
    ngc_api_key: str | None = None

    nim_base_url: str = "https://integrate.api.nvidia.com/v1"
    # Hosted catalogue churns, and /models lists more than your key can invoke:
    # retired ids answer 410, un-entitled ones 404. Verified working for this key
    # on 2026-09-23. See README for how to re-probe.
    nim_model: str = "meta/llama-3.2-11b-vision-instruct"
    request_timeout_s: float = 60.0

    cors_origins: list[str] = Field(default_factory=list)

    @field_validator("cors_origins", mode="before")
    @classmethod
    def _split_origins(cls, value: object) -> object:
        if isinstance(value, str):
            return [origin.strip() for origin in value.split(",") if origin.strip()]
        return value

    @property
    def api_key(self) -> str:
        key = self.nvidia_api_key or self.ngc_api_key
        if not key:
            raise RuntimeError(
                "No NVIDIA_API_KEY (or NGC_API_KEY) found. Set it in the environment, "
                f"in {LOCAL_ENV}, or in {REPO_ENV}."
            )
        return key


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    print(json.dumps(settings.__dict__, indent=4))
    return settings
