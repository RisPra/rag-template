def to_markdown(text):
    from IPython.display import Markdown
    import textwrap
    text = text.replace("•", "*")
    return Markdown(textwrap.indent(text, '> ', predicate=lambda _: True))
