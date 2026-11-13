import json
from importlib.resources import files


def load_fixture(filename: str) -> dict:
    resource = files("billing_assistant").joinpath("fixtures", filename)
    return json.loads(resource.read_text(encoding="utf-8"))
