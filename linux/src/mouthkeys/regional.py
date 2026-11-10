"""Passive, local, one-time regional filler suggestions, adapted from upstream."""

import locale
import os
import re
from pathlib import Path

UPSTREAM_FILLERS = "um,uh,er,ah,eh,umm,uhh,err,ahh,ehh,hmm,hm,mm,mmm,erm,urm,ugh"
CANADIAN_ZONES = {
    "America/St_Johns",
    "America/Halifax",
    "America/Glace_Bay",
    "America/Moncton",
    "America/Goose_Bay",
    "America/Blanc-Sablon",
    "America/Toronto",
    "America/Montreal",
    "America/Nipigon",
    "America/Thunder_Bay",
    "America/Iqaluit",
    "America/Pangnirtung",
    "America/Atikokan",
    "America/Winnipeg",
    "America/Rainy_River",
    "America/Resolute",
    "America/Rankin_Inlet",
    "America/Regina",
    "America/Swift_Current",
    "America/Edmonton",
    "America/Cambridge_Bay",
    "America/Yellowknife",
    "America/Inuvik",
    "America/Creston",
    "America/Dawson_Creek",
    "America/Fort_Nelson",
    "America/Vancouver",
    "America/Whitehorse",
    "America/Dawson",
    "Canada/Newfoundland",
    "Canada/Atlantic",
    "Canada/Eastern",
    "Canada/Central",
    "Canada/Saskatchewan",
    "Canada/Mountain",
    "Canada/Pacific",
    "Canada/Yukon",
}
CENTRAL_ZONES = {
    "America/Chicago",
    "America/North_Dakota/Center",
    "America/North_Dakota/New_Salem",
    "America/North_Dakota/Beulah",
    "US/Central",
}
# First match wins, including when system region and time zone disagree.
REGIONS = (
    (
        "canada",
        "eh",
        {"CA"},
        CANADIAN_ZONES,
        'Looks like you\'re in Canada. Keep "eh" in your transcripts, eh?',
    ),
    (
        "michigan",
        "eh",
        set(),
        {"America/Detroit", "America/Menominee", "US/Michigan"},
        'Michigan, eh? Keep "eh" in your transcripts?',
    ),
    (
        "minnesota",
        "eh",
        set(),
        CENTRAL_ZONES,
        'Minnesota, by any chance? Keep "eh" in your transcripts, ya?',
    ),
    (
        "newZealand",
        "eh",
        {"NZ"},
        {"Pacific/Auckland", "Pacific/Chatham", "NZ", "NZ-CHAT"},
        'Kia ora! Keep "eh" in your transcripts? Sweet as, eh.',
    ),
    (
        "singapore",
        "ah",
        {"SG"},
        {"Asia/Singapore", "Singapore"},
        'Singapore ah? Keep "ah" in your transcripts or not?',
    ),
    (
        "malaysia",
        "ah",
        {"MY"},
        {"Asia/Kuala_Lumpur", "Asia/Kuching"},
        'Malaysia ah? Keep "ah" in your transcripts or not?',
    ),
    (
        "queensland",
        "eh",
        set(),
        {"Australia/Brisbane", "Australia/Lindeman", "Australia/Queensland"},
        'G\'day, Queensland. Keep "eh" in your transcripts, eh?',
    ),
)
_MINNESOTAN = re.compile(
    r"\b(ope|oop|uff ?da|you betcha|ya betcha|dontcha know|oh f[eo]r cute)\b", re.I
)


def region_offer(config, *, region=None, zone=None, dictation=None):
    if config.regional_filler_answered == "on" or config.remove_filler_words != "on":
        return None
    if region is None:
        language = locale.getlocale()[0] or ""
        region = language.rsplit("_", 1)[-1].upper()
    if zone is None:
        zone = os.environ.get("TZ", "").removeprefix(":")
        if not zone:
            zone = str(Path("/etc/localtime").resolve())
        if "/zoneinfo/" in zone:
            zone = zone.split("/zoneinfo/", 1)[-1]
    fillers = {part.strip().casefold() for part in config.filler_words.split(",")}
    for name, word, region_codes, zones, message in REGIONS:
        if region in region_codes or zone in zones:
            # Match region first; do not fall through if its word is already kept.
            if word not in fillers:
                return None
            if name == "minnesota" and _MINNESOTAN.search(dictation or ""):
                message = 'Ope, sounds like Minnesota. Keep "eh" in your transcripts?'
            return {"region": name, "word": word, "message": message}
    return None
