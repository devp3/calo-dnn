def truncate(number, digits) -> float:
    multiplier = 10.0 ** digits
    return int(number * multiplier) / multiplier