import os


def load_text_file(filepath: str) -> str:
    """
    Load a prompt text file or return a default prompt if not found.

    :param filepath: Path to the prompt file.
    :return: File contents or default prompt string.
    """
    if not os.path.exists(filepath):
        if 'system_prompt1.txt' in filepath:
            return (
                """
                You are a document structure analyzer. Your task is to classify the
                structure of a document based on a JSON representation of its initial rows.
                The possible classifications are: 'perfect_structure', 'one_splitter',
                or 'multiple_splitters'. Respond with only a JSON object like: {"class": "your_classification"}
                """
            )
        return (
            """
            You are a document structure analyzer. Based on the provided main splitter
            values and the JSON structure, identify the main and sub-splitter indices
            and the meaningful data parts within the document. Respond with only the JSON structure.
            """
        )
    with open(filepath, 'r', encoding='utf-8') as file:
        return file.read()