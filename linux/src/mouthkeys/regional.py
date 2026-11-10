"""Local, optional regional filler suggestions; no location/network service."""

import locale
import os
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
}


def region_offer(config, *, region=None, zone=None):
    if config.regional_filler_answered == "on" or config.remove_filler_words != "on":
        return None
    if region is None:
        language = locale.getlocale()[0] or ""
        region = language.rsplit("_", 1)[-1].upper()
    if zone is None:
        zone = os.environ.get("TZ", "").removeprefix(":")
        if not zone:
            path = str(Path("/etc/localtime").resolve())
            zone = path.split("/zoneinfo/", 1)[-1] if "/zoneinfo/" in path else ""
    word = None
    if (
        region in ("CA", "NZ")
        or zone in CANADIAN_ZONES
        or zone.startswith("Canada/")
        or zone
        in {
            "Pacific/Auckland",
            "Pacific/Chatham",
            "NZ",
            "NZ-CHAT",
            "America/Detroit",
            "US/Michigan",
            "Australia/Brisbane",
            "Australia/Lindeman",
            "Australia/Queensland",
            "America/Chicago",
            "US/Central",
            "America/North_Dakota/Center",
            "America/North_Dakota/New_Salem",
            "America/North_Dakota/Beulah",
        }
    ):
        word = "eh"
    elif region in ("SG", "MY") or zone in {
        "Asia/Singapore",
        "Singapore",
        "Asia/Kuala_Lumpur",
        "Asia/Kuching",
    }:
        word = "ah"
    if word and word in {part.strip().casefold() for part in config.filler_words.split(",")}:
        return {
            "word": word,
            "message": (
                f"If you use “{word}” as a regional expression, keep it instead of removing it?"
            ),
        }
    return None
