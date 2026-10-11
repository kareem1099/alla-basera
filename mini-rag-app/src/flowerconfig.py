import os
from helpers.config import get_settings

port = int(os.getenv("CELERY_FLOWER_PORT", "5555"))
max_tasks = 10000
auto_refresh = True

flower_password = os.getenv("CELERY_FLOWER_PASSWORD") or get_settings().CELERY_FLOWER_PASSWORD
if not flower_password:
    raise RuntimeError("CELERY_FLOWER_PASSWORD is not set")

basic_auth = [f"admin:{flower_password}"]
