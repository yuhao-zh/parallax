# pylint: disable=too-many-locals
"""
Tests for the radix tree.
"""

from parallax.server.radix_cache import RadixCache

if __name__ == "__main__":
    tree = RadixCache(None, None, page_size=1, disable=False)

    tree.insert("Hello")
    tree.insert("Hello")
    tree.insert("Hello_L.A.!")
    tree.insert("Hello_world! Happy")
    tree.insert("I love you!")
    tree.pretty_print()

    print(tree.match_prefix("I love you! aha"))

    def evict_callback(x):
       print("evict", x)
       return len(x)

    tree.evict(5, evict_callback)
    tree.evict(10, evict_callback)
    tree.pretty_print()