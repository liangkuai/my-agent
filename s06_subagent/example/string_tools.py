import re


def slugify(text: str) -> str:
    """
    Convert text to a URL-friendly slug.

    Steps performed:
    1. Convert text to lowercase
    2. Replace spaces and non-alphanumeric characters (except hyphens) with hyphens
    3. Collapse multiple consecutive hyphens into one
    4. Strip leading/trailing hyphens
    """
    # 1. Convert to lowercase
    slug = text.lower()

    # 2. Replace spaces and non-alphanumeric characters (except hyphens) with hyphens
    slug = re.sub(r'[^a-z0-9-]', '-', slug)

    # 3. Collapse multiple consecutive hyphens into one
    slug = re.sub(r'-{2,}', '-', slug)

    # 4. Strip leading/trailing hyphens
    slug = slug.strip('-')

    return slug
