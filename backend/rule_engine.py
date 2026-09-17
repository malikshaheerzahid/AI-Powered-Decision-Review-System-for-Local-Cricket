"""
Real LBW law, hand-coded (not learned) -- this is what actually decides
OUT vs NOT OUT, using the three sub-component predictions from the model.
Keeping this as an explicit rule (instead of a black-box "decision" head)
means the verdict is always consistent with real cricket law and is easy
to explain/defend in a viva.
"""


def lbw_rule_engine(pitching: str, impact: str, wickets: str) -> dict:
    """
    pitching: "OUTSIDE_LEG" | "OUTSIDE_OFF" | "IN_LINE"
    impact:   "OUTSIDE" | "IN_LINE"
    wickets:  "MISSING" | "HITTING"
    Returns {"decision": "OUT"|"NOT_OUT", "reason": str}
    """
    if pitching == "OUTSIDE_LEG":
        return {"decision": "NOT_OUT", "reason": "Ball pitched outside leg stump"}

    if impact == "OUTSIDE":
        return {"decision": "NOT_OUT", "reason": "Impact was outside the line of off stump"}

    if wickets == "MISSING":
        return {"decision": "NOT_OUT", "reason": "Ball projected to miss the stumps"}

    return {"decision": "OUT", "reason": "Pitched in line/outside off, impact in-line, hitting the stumps"}
