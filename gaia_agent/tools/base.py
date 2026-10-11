def make_tool(name: str, description: str, properties: dict, required: list, func) -> dict:
    """Bundle an OpenAI-style function schema with the Python callable that implements it."""
    return {
        "schema": {
            "type": "function",
            "function": {
                "name": name,
                "description": description,
                "parameters": {"type": "object", "properties": properties, "required": required},
            },
        },
        "func": func,
    }
