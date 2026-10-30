"""Magneye: one photo and a prompt become an interactive 3D exploded view, built by Meta models."""


def main() -> None:
    from .app import main as serve

    serve()
