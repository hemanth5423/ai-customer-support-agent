"""AppleSupport intent taxonomy used by the MVP.

The taxonomy is intentionally small. It is based on recurring issue types in the
AppleSupport slice of TWCS rather than trying to reproduce every product area.
"""

INTENTS = {
    "account_access": "Apple ID, login, password, verification, account lockout or security access.",
    "billing": "Charges, subscriptions, payment methods, billing information or unexpected payments.",
    "connectivity": "Wi-Fi, Bluetooth, cellular data, network, signal or connection problems.",
    "delivery": "Orders, shipping, delivery status or physical replacement/order logistics.",
    "product_issue": "A product/app feature is missing, broken, or behaving unexpectedly.",
    "technical_support": "Device/software troubleshooting such as battery, crashes, updates, display, camera, audio or performance.",
    "information_request": "A question asking how/where/when something works or where to find a feature.",
    "complaint": "Primarily dissatisfaction/frustration where the complaint itself is the main issue.",
    "other": "Does not fit the above categories."
}

def taxonomy_prompt() -> str:
    return "\n".join(f"- {k}: {v}" for k, v in INTENTS.items())
