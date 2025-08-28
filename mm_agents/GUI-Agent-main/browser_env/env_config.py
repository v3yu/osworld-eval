# websites domain
import os

REDDIT = os.environ.get("REDDIT", "")
SHOPPING = os.environ.get("SHOPPING", "")
SHOPPING_ADMIN = os.environ.get("SHOPPING_ADMIN", "")
GITLAB = os.environ.get("GITLAB", "")
WIKIPEDIA = os.environ.get("WIKIPEDIA", "")
MAP = os.environ.get("MAP", "")
HOMEPAGE = os.environ.get("HOMEPAGE", "")
CLASSIFIEDS = os.environ.get("CLASSIFIEDS", "")
TWITTER = os.environ.get("TWITTER", "")

assert (
    # REDDIT
    # and SHOPPING
    # and SHOPPING_ADMIN
    # and GITLAB
    WIKIPEDIA
    # and MAP
    # and CLASSIFIEDS
    # and HOMEPAGE
), (
    f"Please setup the URLs to each site. Current: "
    + f"Reddit: {REDDIT}"
    + f"Shopping: {SHOPPING}"
    + f"Shopping Admin: {SHOPPING_ADMIN}"
    + f"Gitlab: {GITLAB}"
    + f"Wikipedia: {WIKIPEDIA}"
    + f"Map: {MAP}"
    + f"Classifieds: {CLASSIFIEDS}"
    + f"Twitter: {TWITTER}"
    + f"Homepage: {HOMEPAGE}"
)


ACCOUNTS = {
    "reddit": {"username": "wenn0111", "password": "3129028Aa."},
    "gitlab": {"username": "byteblaze", "password": "hello1234"},
    "shopping": {
        "username": "emma.lopez@gmail.com",
        "password": "Password.123",
    },
    "shopping_admin": {"username": "admin", "password": "admin1234"},
    "shopping_site_admin": {"username": "admin", "password": "admin1234"},
    "classifieds": {
        "username": "wenn0111",
        "password": "123456",
    },
    "twitter": {
        "username": "wenn0111",
        "password": "123456",
    },
}

URL_MAPPINGS = {
    REDDIT: "https://www.reddit.com/",
    SHOPPING: "https://www.amazon.com/",
    SHOPPING_ADMIN: "https://www.amazon.com/admin",
    GITLAB: "https://gitlab.com/",
    WIKIPEDIA: "https://www.wikipedia.org/",
    MAP: "https://www.openstreetmap.org/",
    CLASSIFIEDS: "https://www.amazon.com/",
    HOMEPAGE: "https://www.wikipedia.org/",
    TWITTER: "https://twitter.com/",
}
