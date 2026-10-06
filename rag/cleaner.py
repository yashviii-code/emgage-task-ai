import re


def clean_text(text: str) -> str:
    """
    Clean and normalize text extracted from Emgage documentation.

    Args:
        text: Raw text extracted from a webpage.

    Returns:
        Cleaned text.
    """

    # Handle empty or invalid input
    if not text:
        return ""

    # Make sure the input is a string
    text = str(text)

    # Normalize different line endings
    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")

    # Replace tabs with spaces
    text = text.replace("\t", " ")

    # Remove excessive spaces
    text = re.sub(
        r"[ ]{2,}",
        " ",
        text
    )

    # Remove spaces at the beginning/end of lines
    text = "\n".join(
        line.strip()
        for line in text.split("\n")
    )

    # Remove excessive blank lines
    text = re.sub(
        r"\n{3,}",
        "\n\n",
        text
    )

    # Remove leading/trailing whitespace
    text = text.strip()

    return text


if __name__ == "__main__":

    # Simple test
    sample_text = """
        Employee Management


        Employees can be searched using employee ID.
        
        
        Employee details are available in the system.
    """

    cleaned = clean_text(sample_text)

    print("=" * 60)
    print("CLEANED TEXT")
    print("=" * 60)
    print(cleaned)
    print("=" * 60)