import secrets
import time

from uuid import UUID


def uuid7() -> UUID:
    """
    Generate a UUID version 7 using Python's standard library.

    This implementation avoids the blocked native uuid_utils extension.
    """

    # 48-bit Unix timestamp in milliseconds
    timestamp_ms = time.time_ns() // 1_000_000

    # Generate 74 random bits
    random_bits = secrets.randbits(74)

    # Split random bits into UUIDv7 fields
    rand_a = random_bits >> 62
    rand_b = random_bits & ((1 << 62) - 1)

    # Construct UUIDv7
    uuid_integer = (
        (timestamp_ms << 80)
        | (0x7 << 76)
        | (rand_a << 64)
        | (0b10 << 62)
        | rand_b
    )

    return UUID(int=uuid_integer)