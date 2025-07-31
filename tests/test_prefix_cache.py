# pylint: disable=too-many-locals
"""
Tests for the radix tree.
"""

from parallax.server.radix_cache import RadixCache

if __name__ == "__main__":
    tree = RadixCache(page_size=64, disable=False)

    tree.insert("Hello")
    tree.insert("Hello")
    tree.insert("Hello_L.A.!")
    tree.insert("Hello_world! Happy")
    tree.insert("I love you!")
    tree.pretty_print()

    print(tree.match_prefix("I love you! aha"))

    tree.evict(5)
    tree.evict(10)
    tree.pretty_print()