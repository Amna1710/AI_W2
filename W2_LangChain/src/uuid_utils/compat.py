"""Pure-Python implementation of uuid_utils.compat.uuid7.

This module avoids loading the native _uuid_utils extension, which may be
blocked by Windows Smart App Control.
"""

import secrets
import threading
import time
import uuid


_LOCK = threading.Lock()
_LAST_TIMESTAMP_MS = -1
_COUNTER = 0


def uuid7(timestamp: int | None = None, nanos: int | None = None) -> uuid.UUID:
    """Generate a time-ordered UUID version 7.

    Args:
        timestamp: Optional Unix timestamp in seconds.
        nanos: Optional nanoseconds within the specified second.

    Returns:
        A standard-library uuid.UUID object with version 7.
    """
    global _LAST_TIMESTAMP_MS, _COUNTER

    if timestamp is None:
        timestamp_ms = time.time_ns() // 1_000_000
    else:
        timestamp_ms = timestamp * 1000
        if nanos is not None:
            timestamp_ms += nanos // 1_000_000

    # UUIDv7 stores the Unix timestamp in the first 48 bits.
    timestamp_ms &= (1 << 48) - 1

    with _LOCK:
        if timestamp_ms > _LAST_TIMESTAMP_MS:
            _LAST_TIMESTAMP_MS = timestamp_ms
            # Use a 41-bit starting counter, leaving its highest bit clear.
            _COUNTER = secrets.randbits(41)
        else:
            # Preserve monotonic ordering if multiple UUIDs are generated
            # during the same millisecond or the system clock moves backward.
            timestamp_ms = _LAST_TIMESTAMP_MS
            _COUNTER += 1

            if _COUNTER >= (1 << 41):
                timestamp_ms += 1
                _LAST_TIMESTAMP_MS = timestamp_ms
                _COUNTER = secrets.randbits(41)

        counter = _COUNTER

    counter_high = (counter >> 30) & 0xFFF
    counter_low = counter & ((1 << 30) - 1)
    random_bits = secrets.randbits(32)

    value = (
        (timestamp_ms << 80)
        | (0x7 << 76)
        | (counter_high << 64)
        | (0b10 << 62)
        | (counter_low << 32)
        | random_bits
    )

    return uuid.UUID(int=value)