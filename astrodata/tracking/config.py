from pydantic import BaseModel
from typing import List, Optional


class GitConfig(BaseModel):
    init: bool
    remote_url: Optional[str]
    ssh_key_path: Optional[str]


class DVCConfig(BaseModel):
    init: bool
    remote_name: Optional[str]
    remote_url: Optional[str]


class TrackConfig(BaseModel):
    data_dirs: List[str]
    code_dirs: List[str]


class TrackingConfig(BaseModel):
    git: GitConfig
    dvc: DVCConfig
    track: TrackConfig
