import re


def sanitize_file_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]", "_", value.replace(" ", "_"))
