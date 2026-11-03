# Databricks notebook source
# MAGIC %md
# MAGIC # Individual servicing provider deep dive
# MAGIC
# MAGIC **What this notebook does.** It looks at two named servicing providers one at a time and
# MAGIC describes everything they billed. A short list of procedure codes and a short list of primary
# MAGIC diagnoses are used as a lens: they are highlighted everywhere, counted separately, and compared
# MAGIC against the rest of the provider's work. It writes one Excel workbook covering both providers
# MAGIC and emails it to the review team.
# MAGIC
# MAGIC **The most important thing to understand about the numbers.** The focus codes and the focus
# MAGIC diagnoses are a lens, not a filter. Every claim for these two providers is loaded, whatever the
# MAGIC code and whatever the diagnosis. That is deliberate, and there are four reasons for it:
# MAGIC
# MAGIC 1. A person's working day contains every timed service they billed, not four of them. Filter
# MAGIC    the data to four codes and the hours in a day come out too low.
# MAGIC 2. Every rate needs a denominator. Without the whole book you cannot say what share of this
# MAGIC    provider's work the focus codes are, which is usually the question being asked.
# MAGIC 3. The same day conflict rules compare one code against another. H0004 must not be billed on
# MAGIC    the same day as a residential or intensive outpatient per diem, and you cannot see that
# MAGIC    unless the per diem lines are loaded too.
# MAGIC 4. You cannot tell whether the focus diagnoses are this provider's whole caseload or a small
# MAGIC    corner of it unless the other diagnoses are there to compare against.
# MAGIC
# MAGIC A `focus_only` setting is available and is switched off by default. The Criteria sheet says in
# MAGIC plain language what switching it on costs.
# MAGIC
# MAGIC **What the table holds, and what it therefore cannot tell you.** `all_paid_services` holds
# MAGIC paid claim lines only. There are no denied lines in it, so nothing in this workbook can say
# MAGIC anything about what was refused, nor can a rate here be read as an approval rate. Every dollar
# MAGIC figure is money paid.
# MAGIC
# MAGIC **Columns are detected, never assumed.** Only a handful of columns are treated as certain:
# MAGIC the servicing and billing provider, the member, the procedure code, the units, the dollars and
# MAGIC the service dates. Everything else is looked for by name and, when it is not there, the parts
# MAGIC of the analysis that needed it are reported as not run rather than quietly left out. The
# MAGIC `Data_Availability` sheet lists every optional column, whether it was found, and exactly which
# MAGIC analysis was skipped or weakened without it. Read it before concluding that something is
# MAGIC absent from this provider's billing rather than absent from the extract.
# MAGIC
# MAGIC **What this is not.** It is paid claims data. Nothing here establishes fraud, intent, or
# MAGIC whether a service was medically necessary. Every row is something to look at, not a conclusion.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Settings
# MAGIC
# MAGIC Everything a reviewer might want to change lives here: the two providers, the focus codes, the
# MAGIC focus diagnoses, the thresholds and the email recipients. Each setting is printed onto the
# MAGIC Criteria sheet of the workbook so a result can be checked without reading any code.
# MAGIC
# MAGIC The settings read notebook widgets when widgets exist and fall back to the defaults when they
# MAGIC do not, so the same notebook can be scheduled several times for different providers with no
# MAGIC code edits.

# COMMAND ----------

from pyspark.sql import SparkSession
import pyspark.sql.functions as F
from pyspark.sql.window import Window
import pandas as pd
import numpy as np
import re
from datetime import date

spark = SparkSession.builder.getOrCreate()


def _param(name, default, label):
    """Read a notebook widget when one exists, otherwise fall back to the default.
    This lets the SAME notebook be scheduled once per provider without anyone editing code."""
    try:
        dbutils.widgets.text(name, default, label)          # noqa: F821
        v = dbutils.widgets.get(name)                       # noqa: F821
        return v.strip() if v and v.strip() else default
    except Exception:
        return default


def _flag(name, default, label):
    return _param(name, default, label).strip().lower() in ("1", "true", "yes", "y")


def _list(name, default, label):
    return [x.strip() for x in _param(name, default, label).split(",") if x.strip()]


ROOT = "sedo.senou@azahcccs.gov"
SERVICE_TABLE = "main.prod_input.all_paid_services"

# The two providers under review. IDs and names are kept as a pair so the workbook can show the
# name the investigator used and the name the claims carry, and flag a mismatch.
TARGETS = _list("targets", "123531,120020", "Servicing provider IDs (comma separated)")
TARGET_NAMES_GIVEN = {
    "123531": "Knudson, Rachel Elizabeth",
    "120020": "Etukudoh Akaninyene J",
}

# The lens. Codes and primary diagnoses that are highlighted everywhere but never used to
# restrict the data.
FOCUS_CODES = [c.strip().upper() for c in
               _list("focus_codes", "H2038,T1016,H0004,H0031", "Focus procedure codes")]
FOCUS_DIAGS = [d.strip().upper().replace(".", "") for d in
               _list("focus_diags", "F1020,F1520,F4310,F1010", "Focus primary diagnoses")]

# Off by default, and the Criteria sheet explains what it costs.
FOCUS_ONLY = _flag("focus_only", "false",
                   "Restrict the claims to the focus codes and diagnoses (true/false)")

EMAIL_TO = _list("email_to", "sedo.senou@azahcccs.gov", "Email recipients (comma separated)")

# Column names on main.prod_input.all_paid_services. Every one of these is checked against the
# real schema in section 5 before it is used.
SPROV_COL, BPROV_COL, RPROV_COL = "SProv_ID", "BProv_ID", "RProv_ID"
MEMBER_COL, PROC_COL = "MEMBER_KEY", "PROC_CD"
QTY_COL, PMT_COL = "QUANTITY_PAID", "PMT_AMT"
SRV_BEG_COL, SRV_END_COL = "SRV_BEG_CALENDAR_KEY", "SRV_END_CALENDAR_KEY"
MODIFIER_COLS = ["PROC_MOD_1", "PROC_MOD_2", "PROC_MOD_3", "PROC_MOD_4"]
DIAG_COLS = ["DIAG_CD_1", "DIAG_CD_2", "DIAG_CD_3", "DIAG_CD_4"]
DIAG_DESC_COLS = ["DIAG_CD_1_DESC", "DIAG_CD_2_DESC", "DIAG_CD_3_DESC", "DIAG_CD_4_DESC"]

# Values that mean "no modifier on this line". This extract carries -1 and 0 as placeholders,
# and treating either as a real modifier would invent a modifier mix that does not exist.
MOD_NULL_TOKENS = {"", "-1", "0", "NA", "N/A", "NONE", "NULL"}
# Values that mean "no referring provider on this line".
ID_NULL_TOKENS = {"", "NA", "N/A", "NONE", "NULL", "0", "-1", "UNKNOWN"}

TODAY = date.today()
RUN_TAG = TODAY.strftime("%Y-%m-%d")
TARGET_SLUG = "_".join(TARGETS)[:40]

OUTPUT_DIR = f"/Workspace/Users/{ROOT}/Projects/provider_statistics/peer_analysis/results"
OUTPUT_PATH = f"{OUTPUT_DIR}/provider_deep_dive_{TARGET_SLUG}_{RUN_TAG}.xlsx"
HISTORY_PATH = f"{OUTPUT_DIR}/_deep_dive_history_{TARGET_SLUG}.csv"

SMTP_HOST, SMTP_PORT = "smtp.office365.com", 587
EMAIL_FROM = ROOT
SMTP_SCOPE = "siu-email"
SMTP_USER_KEY = "smtp_user"
SMTP_PASS_KEY = "smtp_password"
MAX_ATTACH_MB = 15


# The guard that catches the failure mode to fear. A previous build in this project shortened
# column names to 14 characters, three labels collided, two tests were silently overwritten and
# found nothing, and only a calibration figure of 0.0% revealed it. Nothing here is ever
# shortened, and every set of names is asserted unique before it is used.
def _assert_unique(names, where):
    names = [str(n) for n in names]
    dupes = sorted({n for n in names if names.count(n) > 1})
    if dupes:
        raise ValueError(f"duplicate names in {where}: {dupes}. Rename them; do not truncate.")
    return names


_assert_unique(TARGETS, "target provider ids")
_assert_unique(FOCUS_CODES, "focus procedure codes")
_assert_unique(FOCUS_DIAGS, "focus primary diagnoses")

print(f"Run {RUN_TAG}")
print(f"Servicing providers under review: {', '.join(TARGETS)}")
for _t in TARGETS:
    print(f"   {_t}: {TARGET_NAMES_GIVEN.get(_t, '(no name supplied with the request)')}")
print(f"Focus procedure codes: {', '.join(FOCUS_CODES)}")
print(f"Focus primary diagnoses: {', '.join(FOCUS_DIAGS)}")
print(f"focus_only is {'ON: the claims ARE restricted to the lens' if FOCUS_ONLY else 'OFF: every claim for these providers is loaded'}")
print(f"Workbook will be written to: {OUTPUT_PATH}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. What one paid unit means, code by code
# MAGIC
# MAGIC This is the cell to check first if a number looks wrong. A unit on a behavioural health claim
# MAGIC means a different amount of time depending on the code. Some codes are 15 minutes a unit. A few
# MAGIC are a whole hour. Some are a whole day, whatever happened in it. And some are one session,
# MAGIC whether that session ran 30 minutes or 60.
# MAGIC
# MAGIC Three rules this notebook follows:
# MAGIC
# MAGIC - **A session is one unit no matter how long it lasted.** H0031, one of the four focus codes,
# MAGIC   is one of these. AHCCCS does not state a duration for it. It is counted as sessions and is
# MAGIC   never multiplied by a nominal length, so it contributes zero minutes to any day total.
# MAGIC - **The unit changes between tiers of the same service**, so the basis is held per code and
# MAGIC   never per service family.
# MAGIC - **Where AHCCCS has not published a unit, we do not guess.** Those codes keep their dollars
# MAGIC   and stay out of every minutes and days calculation.
# MAGIC
# MAGIC **The four focus codes, and what each one is.**
# MAGIC
# MAGIC | Code | Service | One unit means | Note |
# MAGIC |---|---|---|---|
# MAGIC | `H0004` | counselling and therapy | 15 minutes | on the published above 8 units a day documentation list |
# MAGIC | `T1016` | case management | 15 minutes | on the same documentation list; must not be billed the same day as H0018 |
# MAGIC | `H0031` | mental health assessment by a non physician | one session | AHCCCS does not state a unit. Never multiplied by a duration |
# MAGIC | `H2038` | rehabilitation programme, per diem | one day, with a five hour minimum | basis NOT verified. See the warning below |
# MAGIC
# MAGIC **The H2038 warning, and it matters.** The fee for service billing manual, chapter 19,
# MAGIC describes H2038 as a per diem with a five hour minimum duration. The code is absent from the
# MAGIC FY21 fee schedule and absent from the 2023 rate study's behavioural health outpatient code
# MAGIC list, which is why the shared specification marks its unit basis unverified. A rate notice
# MAGIC effective 2025-10-01 prices it at $488.88. This notebook therefore treats it as a per diem,
# MAGIC which means **more than one unit in a day is a finding on the face of the claim**, and keeps it
# MAGIC out of the minutes arithmetic because the basis is not confirmed. Before using any H2038 number
# MAGIC in this workbook, confirm the basis against the current fee schedule. At roughly $489 a unit a
# MAGIC unit error is expensive in both directions.

# COMMAND ----------

TIMED_15_CODES = {                       # one paid unit = 15 minutes
    "H0004",   # counselling and therapy                                   FOCUS CODE
    "H0034",   # medication training and support (4 unit doc threshold, not 8)
    "H0036",   # community psychiatric supportive treatment, face to face
    "H0038",   # self help / peer services
    "H2010",   # comprehensive medication services
    "H2011",   # crisis intervention
    "H2014",   # skills training and development
    "H2015",   # community support, above 5 hours to 10 hours tier
    "H2017",   # psychosocial rehabilitation
    "H2019",   # therapeutic behavioural service, up to 5 hours
    "H2021",   # community based wraparound
    "H2025",   # ongoing support to maintain employment
    "H2027",   # psychoeducational service
    "H2033",   # multisystemic therapy
    "S5110",   # home care training, family
    "S5150",   # unskilled respite care
    "T1002",   # RN services
    "T1003",   # LPN / LVN services
    "T1016",   # case management                                           FOCUS CODE
    "T1019",   # personal care services
    "97151", "97152", "97153", "97154", "97155", "97156", "97157", "97158",   # ABA
}
TIMED_60_CODES = {                       # one paid unit = 60 minutes
    "H2012",   # behavioural health day treatment, up to 5 hours
    "S9484",   # crisis intervention per hour, up to 5 hours
}
PER_DIEM_CODES = {                       # one paid unit = one day
    "H0015",   # intensive outpatient programme
    "S9480",   # intensive outpatient psychiatric
    "H0018",   # short term residential, BHRF, services only
    "H0019",   # long term residential                       (unit UNVERIFIED)
    "H0035",   # partial hospitalisation
    "H0037",   # community psychiatric supportive treatment programme
    "H2016",   # comprehensive community support, per diem
    "H2018",   # psychosocial rehabilitation per diem        (unit UNVERIFIED)
    "H2020",   # therapeutic behavioural service, per diem
    "H2026",   # ongoing support to maintain employment, per diem
    "H2038",   # rehabilitation programme, per diem          (unit UNVERIFIED) FOCUS CODE
    "S9485",   # crisis intervention, per diem
    "S5131", "S5140", "S5145", "S5151", "T1020", "T2020", "T2026",
}
PER_SESSION_CODES = {                    # one paid unit = one session, ANY length
    "90832", "90834", "90837",                         # individual psychotherapy
    "90833", "90836", "90838",                         # psychotherapy add on codes
    "90845", "90846", "90847", "90849", "90853",       # analysis, family, multi family, group
    "99408", "99409",                                  # SBIRT
    "H2041", "H0001", "H0002",
    "H0031",   # mental health assessment by a non physician               FOCUS CODE
    "S5109",
}
# Nominal minutes are carried for REFERENCE ONLY and are never multiplied by units. H0031 is
# deliberately absent from this map: AHCCCS states no duration for it, so there is nothing to
# carry, not even for reference.
SESSION_NOMINAL_MINUTES = {
    "90832": 30, "90834": 45, "90837": 60, "90846": 50, "90847": 50,
    "90845": 50, "90849": 50, "90853": 50,
}
PER_DOSE_CODES = {"H0020"}                                       # methadone, one dose per day
PER_MONTH_CODES = {"99484", "99492", "99493", "99494", "G2214"}  # per calendar month, never per day

# Confirmed ABSENT from the AHCCCS behavioural health code set against four independent sources.
# If one of these turns up on a target's claims that is itself worth a look, so it is counted and
# reported rather than quietly classified.
ABSENT_CODES = {"H2034", "H2036", "H0010", "H0012", "H0013", "H2013", "H2032", "H0005", "T1017"}

# Codes whose unit basis could NOT be verified against an AHCCCS statement. They keep their
# dollars and stay out of every minutes calculation, so a guess cannot quietly drive a finding.
UNVERIFIED_BASIS_CODES = {
    "H0019",            # per diem from the national HCPCS descriptor only, no AHCCCS statement
    "H2018", "H2038",   # per diem in FFS chapter 19, absent from the fee schedule and the
                        # behavioural health outpatient code list
    "H0025",            # 15 minutes inferred only from its neighbours on the documentation list
    "H0006", "H0014", "H0030", "T1503",   # no AHCCCS unit statement found
}

# Two per diem sets, and the difference between them matters.
#
# PER_DIEM_ALL is every code documented as a per diem anywhere, including H2038. It answers one
# question only: is this line a per diem, or is it a service sitting on top of one? That question
# needs no unit basis, so the unverified codes belong in it.
#
# PER_DIEM_TESTABLE drops the unverified codes. It is the only set allowed to drive the published
# one per diem per member per day rule, because a day count built on an unpublished unit is a
# guess dressed up as arithmetic. H2038 collisions are reported in their own column instead.
PER_DIEM_ALL = set(PER_DIEM_CODES)
PER_DIEM_TESTABLE = PER_DIEM_CODES - UNVERIFIED_BASIS_CODES

RESIDENTIAL_PER_DIEM = {"H0018", "H0019"}
IOP_PER_DIEM = {"H0015", "S9480", "H0035"}
# Used by the published H0004 same day test. H0019 is dropped because its unit is unverified.
RES_OR_IOP_PER_DIEM = (RESIDENTIAL_PER_DIEM | IOP_PER_DIEM) - UNVERIFIED_BASIS_CODES

BH_UNIVERSE = sorted(TIMED_15_CODES | TIMED_60_CODES | PER_DIEM_CODES | PER_SESSION_CODES |
                     PER_DOSE_CODES | PER_MONTH_CODES | ABSENT_CODES | UNVERIFIED_BASIS_CODES)


def _basis_of(code):
    if code in UNVERIFIED_BASIS_CODES:
        return "UNVERIFIED"
    if code in ABSENT_CODES:
        return "NOT_IN_AHCCCS_BH_SET"
    if code in TIMED_15_CODES:
        return "TIMED_15"
    if code in TIMED_60_CODES:
        return "TIMED_60"
    if code in PER_DIEM_CODES:
        return "PER_DIEM"
    if code in PER_SESSION_CODES:
        return "PER_SESSION"
    if code in PER_DOSE_CODES:
        return "PER_DOSE"
    if code in PER_MONTH_CODES:
        return "PER_MONTH"
    return "UNCLASSIFIED"


CODE_BASIS = {c: _basis_of(c) for c in BH_UNIVERSE}
MINUTES_PER_UNIT = {c: (15 if CODE_BASIS[c] == "TIMED_15" else
                        60 if CODE_BASIS[c] == "TIMED_60" else 0) for c in BH_UNIVERSE}

# The focus codes, with their basis spelled out for the Criteria sheet.
FOCUS_CODE_BASIS = {c: CODE_BASIS.get(c, "UNCLASSIFIED") for c in FOCUS_CODES}
FOCUS_CODE_NOTE = {
    "H0004": "Counselling and therapy. One unit is 15 minutes. On the published above 8 units a "
             "day documentation list for claims RECEIVED on or after 2023-05-03.",
    "T1016": "Case management. One unit is 15 minutes. On the same documentation list. Must not be "
             "billed on the same day as H0018.",
    "H0031": "Mental health assessment by a non physician. Counted as SESSIONS. AHCCCS does not "
             "state a unit, so units are never multiplied by any duration and this code "
             "contributes zero minutes to every day total in this workbook.",
    "H2038": "Treated as a PER DIEM with a five hour minimum, from FFS chapter 19. More than one "
             "unit in a day is a finding on the face of the claim. The basis is NOT verified: the "
             "code is absent from the FY21 fee schedule and from the 2023 rate study's behavioural "
             "health outpatient code list. A rate notice effective 2025-10-01 prices it at "
             "$488.88. CONFIRM THE BASIS AGAINST THE CURRENT FEE SCHEDULE before using any H2038 "
             "number here: at that price a unit error is expensive.",
}

print(f"Code dictionary: {len(BH_UNIVERSE)} behavioural health codes")
for _b in sorted(set(CODE_BASIS.values())):
    print(f"  {_b:<22} {sum(1 for v in CODE_BASIS.values() if v == _b):>3} codes")
print("\nFocus codes and their unit basis:")
for _c in FOCUS_CODES:
    print(f"  {_c}: {FOCUS_CODE_BASIS[_c]}")
    print(f"      {FOCUS_CODE_NOTE.get(_c, 'no note recorded for this code')}")
_unknown_focus = [c for c in FOCUS_CODES if c not in BH_UNIVERSE]
if _unknown_focus:
    print(f"\nWARNING: focus codes not in the behavioural health dictionary: "
          f"{', '.join(_unknown_focus)}. They are still reported, but with no unit basis, so they "
          f"contribute nothing to the minutes or per diem arithmetic.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. The group billing trap, and how this notebook handles it
# MAGIC
# MAGIC **Read this before trusting any hour count in the workbook.**
# MAGIC
# MAGIC AHCCCS counts group time per member for the whole length of the group, not divided between the
# MAGIC people in it. The billing manual's own worked example: eight members in a 60 minute group gives
# MAGIC four units each, billed on eight separate claims, so **32 units come out of 60 minutes of one
# MAGIC clinician's time**. Group size can run to 15. So a clinician running perfectly ordinary group
# MAGIC programming produces summed units that look impossible.
# MAGIC
# MAGIC The group marker on the claim is the `HQ` modifier. Codes whose own description already says
# MAGIC group, 90849 and 90853, do not carry `HQ`, and its absence there is correct rather than odd.
# MAGIC
# MAGIC **What this notebook does by default.** It takes `HQ` lines out of the minutes arithmetic
# MAGIC completely and reports the minutes it set aside in a column of its own, so nothing is hidden.
# MAGIC That is the conservative choice: it can only ever lower the number of long days reported, never
# MAGIC raise it. Two alternatives are available in the settings: divide a group line by the number of
# MAGIC members in that group on that provider day, or count it in full. Whichever is used is printed on
# MAGIC the Criteria sheet.

# COMMAND ----------

GROUP_MODIFIER = "HQ"
FAMILY_MODIFIERS = {"HR", "HS"}        # count per family, not per person
HQ_TREATMENT = _param("hq_treatment", "exclude",
                      "Group (HQ) lines in the minutes arithmetic: exclude / divide / full").lower()
if HQ_TREATMENT not in ("exclude", "divide", "full"):
    raise ValueError(f"hq_treatment must be exclude, divide or full, not {HQ_TREATMENT!r}")
HQ_TREATMENT_TEXT = {
    "exclude": "Group (HQ) lines are EXCLUDED from the clinician day minutes arithmetic and are "
               "reported separately in group_minutes_set_aside. Conservative: it can only lower "
               "the count of long days, never raise it.",
    "divide":  "Group (HQ) line minutes are DIVIDED by the number of distinct members in that "
               "provider's group, for that code and modifier set, on that day.",
    "full":    "Group (HQ) lines are counted IN FULL, which is what the AHCCCS group convention "
               "says the member received. This WILL make legitimate group programmes look like "
               "impossible days.",
}[HQ_TREATMENT]
print(f"HQ treatment: {HQ_TREATMENT}\n  {HQ_TREATMENT_TEXT}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Thresholds, limits and floors
# MAGIC
# MAGIC Two kinds of number live here. A **limit** comes from arithmetic or from a published AHCCCS
# MAGIC rule and needs no defence: 24 hours is 24 hours, and the above 8 units a day documentation
# MAGIC requirement is written down. A **floor** is a judgement about how much volume is needed before
# MAGIC a comparison means anything, and floors are the numbers to change if a peer comparison looks
# MAGIC thin.
# MAGIC
# MAGIC One mechanical point that is easy to get wrong and that this notebook gets right. The 2023-05-03
# MAGIC documentation controls key on the **claim receipt date**, not the service date. The published
# MAGIC wording is "claims received on and after". So the documentation test below uses the date the
# MAGIC claim arrived, and lines received before the cutoff are counted in a separate column that
# MAGIC cannot flag anybody.

# COMMAND ----------

# Clinician day hour thresholds, reported as counts of days, not used to flag.
HOUR_THRESHOLDS = [int(x) for x in _list("hour_thresholds", "8,12,16,24",
                                         "Clinician day hour thresholds")]

# Published documentation requirement: above 8 units a day, for claims RECEIVED on or after
# 2023-05-03. The 11 code list is as published.
DOC_8_CODES = ["H0004", "H0038", "H2011", "H2014", "H2015", "H2017", "H0025", "H2027", "S5150",
               "T1016", "T1019"]
DOC_8_THRESHOLD = int(_param("doc_8_threshold", "8", "Documentation threshold, units a day"))
DOC_4_CODES = ["H0034"]
DOC_4_THRESHOLD = int(_param("doc_4_threshold", "4",
                             "Documentation threshold for H0034, units a day"))
DOC_RECEIPT_CUTOFF = _param("doc_receipt_cutoff", "2023-05-03",
                            "Claim RECEIPT date the documentation rule starts (yyyy-mm-dd)")

# A per diem is one day. More than one unit of the same per diem code for one member on one day
# is a finding on the face of the claim.
PER_DIEM_UNITS_PER_DAY = float(_param("per_diem_units_per_day", "1",
                                      "Units of one per diem code allowed per member day"))

# A line whose service span covers several days puts all its units on the begin date in this
# table. Spans are spread across the days they cover. Anything longer than this is treated as a
# data error, left on the begin date and counted in its own column.
MAX_SPAN_DAYS = int(_param("max_span_days", "62", "Longest service span treated as real, in days"))
SPAN_TREATMENT = _param("span_treatment", "prorate",
                        "Multi day span lines in the day arithmetic: prorate / exclude").lower()
if SPAN_TREATMENT not in ("prorate", "exclude"):
    raise ValueError(f"span_treatment must be prorate or exclude, not {SPAN_TREATMENT!r}")
SPAN_TREATMENT_TEXT = {
    "prorate": "A line covering several days has its units, dollars and minutes spread evenly "
               "across the days it covers, so no single day inherits a fortnight of billing.",
    "exclude": "A line covering several days is kept out of the clinician day arithmetic entirely "
               "and counted in its own column.",
}[SPAN_TREATMENT]

# Peer comparison floors. A peer with three members and one service day produces an extreme
# looking ratio out of nothing, and letting those into a percentile makes the percentile noise.
PEER_MIN_MEMBERS = int(_param("peer_min_members", "10", "Peer floor: distinct members"))
PEER_MIN_DAYS = int(_param("peer_min_days", "10", "Peer floor: distinct service days"))
PEER_MIN_COUNT = int(_param("peer_min_count", "20",
                            "Smallest peer group a percentile is reported for"))
EXCLUDE_OTHER_TARGET_FROM_PEERS = _flag(
    "exclude_other_target", "true",
    "Keep the other target out of each target's peer group (true/false)")

# The contamination diagnostic from section 0 of the shared specification. Above this share of
# peers sitting beyond p75 plus three interquartile ranges, a percentile cut lands inside the
# cluster rather than outside it and the percentile means nothing.
CONTAMINATION_SHARE = float(_param("contamination_share", "0.10",
                                   "Share above p75+3*IQR that marks a measure contaminated"))

# Rows pulled back to the driver for any single detail sheet.
DETAIL_ROWS = int(_param("detail_rows", "50000", "Rows per detail sheet"))

print(f"Clinician day hour thresholds reported: {', '.join(str(h) for h in HOUR_THRESHOLDS)}")
print(f"Documentation rule: above {DOC_8_THRESHOLD} units a day on "
      f"{', '.join(DOC_8_CODES)}, and above {DOC_4_THRESHOLD} on {', '.join(DOC_4_CODES)}, "
      f"for claims RECEIVED on or after {DOC_RECEIPT_CUTOFF}")
print(f"Span treatment: {SPAN_TREATMENT}. {SPAN_TREATMENT_TEXT}")
print(f"Peer floors: at least {PEER_MIN_MEMBERS} members and {PEER_MIN_DAYS} service days, "
      f"and at least {PEER_MIN_COUNT} peers before a percentile is reported")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Which columns this table actually has
# MAGIC
# MAGIC This notebook uses a much richer set of columns than the earlier provider notebooks did:
# MAGIC diagnoses, billed and allowed dollars, the referring provider, the suspension indicator, the
# MAGIC claim receipt date, the revenue code, place of service and more. **Only seven columns are
# MAGIC treated as certain**, and every other one is looked for by name and reported as found or not.
# MAGIC The revenue code column is known to exist but its exact spelling is not, so it is detected
# MAGIC across the names this family of tables uses rather than hard coded.
# MAGIC
# MAGIC **Read the output of this cell.** Anything listed as missing means the part of the workbook
# MAGIC that depends on it says so rather than guessing. Nothing silently becomes a zero.

# COMMAND ----------

_src_cols = spark.table(SERVICE_TABLE).columns
_src_set = set(_src_cols)


def have(col):
    return col in _src_set


def first_present(*cands):
    for c in cands:
        if c in _src_set:
            return c
    return None


MODS = [c for c in MODIFIER_COLS if have(c)]
DIAGS = [c for c in DIAG_COLS if have(c)]
DIAG_DESCS = [c for c in DIAG_DESC_COLS if have(c)]
PRIMARY_DIAG_COL = DIAG_COLS[0] if have(DIAG_COLS[0]) else None
PRIMARY_DIAG_DESC_COL = DIAG_DESC_COLS[0] if have(DIAG_DESC_COLS[0]) else None

BILL_COL = first_present("BILL_AMT")
ALLOW_COL = first_present("ALLOW_AMT")
RECEIPT_KEY_COL = first_present("DATE_RECORD_ADDED_CALENDAR_KEY")
RECEIPT_DT_COL = first_present("Claim_Added_Dt")
PA_COL = first_present("EVENT_PA_NO", "PA_NO", "PRIOR_AUTH_NO")
DOB_COL = first_present("DATE_OF_BIRTH")
POS_CODE_COL = first_present("POS_CODE")
POS_DESC_COL = first_present("POS_DESC")
CLM_TYP_COL = first_present("CLM_TYP_DESC")
FORM_TYP_COL = first_present("FORM_TYP_DESC")
# A revenue code column is confirmed to exist but its exact name is not, so it is detected
# across the spellings this family of tables uses rather than hard coded.
REV_CODE_COL = first_present("Revenue_Cd", "REVENUE_CD", "REV_CD", "REVENUE_CODE", "Revenue_Code",
                             "REV_CODE", "UB_REV_CD", "RevenueCode", "REV_CD_CD")
REV_DESC_COL = first_present("Revenue_Cd_Desc", "REVENUE_CD_DESC", "REV_CD_DESC",
                             "REVENUE_CODE_DESC", "Revenue_Code_Desc", "RevenueCodeDesc",
                             "REV_DESC")
LOB_COL = first_present("line_of_business")
LOB_ORG_COL = first_present("lob_org_name")
IHS_638_COL = first_present("IHS_638")
IHS_PROV_COL = first_present("IHS_PROVIDER")
COS_CODE_COL = first_present("COS_CODE")
COS_DESC_COL = first_present("COS_DESC")
PROC_DESC_COL = first_present("PROC_CD_DESC")
CLAIM_ID_COL = first_present("ClaimID")
CLAIM_HDR_COL = first_present("Claim_Hdr_ID")
LINE_NO_COL = first_present("LN_NO")
SVC_BEG_DT_COL = first_present("Svc_Begin_Dt")
SVC_END_DT_COL = first_present("Svc_End_Dt")

SPROV_NAME_COL = first_present("SProv_Name")
SPROV_TYPE_COL = first_present("SProv_Type")
SPROV_TYPE_DESC_COL = first_present("SProv_Type_Desc")
SPROV_SUSP_COL = first_present("SProv_CAF_Susp_Ind")
BPROV_NAME_COL = first_present("BProv_Name")
BPROV_TYPE_DESC_COL = first_present("BProv_Type_Desc")
BPROV_SUSP_COL = first_present("BProv_CAF_Susp_Ind")
RPROV_NAME_COL = first_present("RProv_Name")
RPROV_SUSP_COL = first_present("RProv_CAF_Susp_Ind")

# ── WHAT THIS NOTEBOOK TREATS AS CERTAIN, AND WHAT IT ONLY HOPES FOR ────────────────────────
#
# Only these seven columns are required. Everything else is optional, is detected by name, and
# when it is absent the analysis that needed it is reported as NOT RUN with the reason rather
# than quietly left out of the workbook. The difference between "this provider did not do it"
# and "this extract could not show it" is the difference between a lead and a false accusation.
REQUIRED = [
    (SPROV_COL, "servicing provider identity"),
    (BPROV_COL, "billing provider identity"),
    (MEMBER_COL, "member identity"),
    (PROC_COL, "procedure code"),
    (QTY_COL, "units paid"),
    (PMT_COL, "dollars paid"),
    (SRV_BEG_COL, "service begin date"),
]
_missing_required = [c for c, _ in REQUIRED if not have(c)]
if _missing_required:
    raise ValueError(
        f"These columns are required and are not on {SERVICE_TABLE}: "
        f"{', '.join(_missing_required)}. Nothing in this notebook can run without them, so it "
        f"stops here rather than producing a workbook built on substitutes.")

# Every optional column, what it is for, and what is lost without it. The third element is the
# label used on the Data_Availability sheet, and it names the analysis rather than the column, so
# an investigator reading that sheet can see which section of the workbook to distrust.
OPTIONAL = [
    (SPROV_NAME_COL, "SProv_Name", "servicing provider name",
     "A_Provider_Profile cannot show the name on the claims, so the name in the request cannot be "
     "checked against the name the claims carry."),
    (SPROV_TYPE_DESC_COL, "SProv_Type_Desc", "provider type description",
     "WEAKENS THE WHOLE OF PART D. Without it the notebook cannot tell whether this provider ID "
     "is one person or an organisational NPI, so no hour total in a day can be read as one "
     "person's day. Part D still reports the hours and marks the question unanswered."),
    (SPROV_TYPE_COL, "SProv_Type", "provider type code",
     "The peer group in Part G falls back to the type DESCRIPTION; if that is missing too, Part G "
     "is NOT RUN and the member overlap in Part F loses its chance comparison."),
    (SPROV_SUSP_COL, "SProv_CAF_Susp_Ind", "servicing provider payment suspension indicator",
     "A_Provider_Profile cannot report whether a payment suspension is set on either provider."),
    (BPROV_NAME_COL, "BProv_Name", "billing provider name",
     "The billing entity sheets show IDs without names."),
    (BPROV_TYPE_DESC_COL, "BProv_Type_Desc", "billing provider type description",
     "The billing entity sheets cannot say whether an entity is a facility or a clinic, which is "
     "what the same day tests in Part E turn on when a row is reviewed."),
    (BPROV_SUSP_COL, "BProv_CAF_Susp_Ind", "billing provider payment suspension indicator",
     "The billing entity sheets cannot report a payment suspension on any entity."),
    (RPROV_COL if have(RPROV_COL) else None, "RProv_ID", "referring provider identity",
     "NOT RUN: the referring provider breakdown per provider, and the shared referring provider "
     "test in Part F. A referral is a relationship stated on the claim and is the best evidence "
     "of a connection between two providers that this table holds, so losing it is the single "
     "biggest loss in Part F."),
    (RPROV_NAME_COL, "RProv_Name", "referring provider name",
     "The referring provider sheets show IDs without names."),
    (PROC_DESC_COL, "PROC_CD_DESC", "procedure description",
     "Every code sheet shows bare codes, so the workbook cannot be read without a code lookup "
     "beside it."),
    (BILL_COL, "BILL_AMT", "dollars billed",
     "NOT RUN: paid over billed, on A_Provider_Profile and on the code sheets. Billed dollars are "
     "reported as blank rather than zero, because a zero would read as nothing billed."),
    (ALLOW_COL, "ALLOW_AMT", "dollars allowed",
     "Allowed dollars are blank on A_Provider_Profile."),
    (SRV_END_COL if have(SRV_END_COL) else None, "SRV_END_CALENDAR_KEY", "service end date",
     "WEAKENS PART D. Every line is treated as a single day line, so a line covering a fortnight "
     "puts a fortnight of units on one date and can create a day that looks impossible. The span "
     "columns on the day sheets are all zero and mean nothing."),
    (RECEIPT_KEY_COL or RECEIPT_DT_COL, "DATE_RECORD_ADDED_CALENDAR_KEY or Claim_Added_Dt",
     "claim receipt date",
     "NOT RUN: the above 8 units a day documentation test in Part E. That rule applies to claims "
     "RECEIVED on or after 2023-05-03 and there is no receipt date here. The notebook does NOT "
     "fall back to the service date: the claim lag would mis-date the test and could produce a "
     "false finding against a named individual, which is the worst outcome available here."),
    (PRIMARY_DIAG_COL, "DIAG_CD_1", "primary diagnosis",
     "NOT RUN: every diagnosis sheet, the focus diagnosis half of Part B, the code by diagnosis "
     "cross tabulation, and the focus diagnosis measure in Part G."),
    (PRIMARY_DIAG_DESC_COL, "DIAG_CD_1_DESC", "primary diagnosis description",
     "The diagnosis sheets show bare codes."),
    (PA_COL, "EVENT_PA_NO", "prior authorization number",
     "A_Provider_Profile cannot count lines carrying a prior authorization."),
    (DOB_COL, "DATE_OF_BIRTH", "member date of birth",
     "NOT RUN: member age at the date of service on A_Provider_Profile."),
    (POS_CODE_COL or POS_DESC_COL, "POS_CODE or POS_DESC", "place of service",
     "A_Provider_Profile cannot list the places of service, so a residential setting cannot be "
     "told apart from an office one when a Part E row is reviewed."),
    (CLM_TYP_COL, "CLM_TYP_DESC", "claim type",
     "A_Provider_Profile cannot show the claim type mix."),
    (FORM_TYP_COL, "FORM_TYP_DESC", "claim form",
     "Professional lines cannot be told apart from institutional ones on A_Provider_Profile."),
    (REV_CODE_COL or REV_DESC_COL, "a revenue code column, detected by name",
     "revenue code on institutional lines",
     "Institutional lines carry no revenue code on the sheets. Residential and subacute stays are "
     "priced by revenue code rather than by an H code, so without this an institutional line "
     "cannot be told apart from a professional one by its content."),
    (LOB_COL, "line_of_business", "line of business",
     "A_Provider_Profile cannot list the lines of business."),
    (LOB_ORG_COL, "lob_org_name", "plan or programme name", "No plan context on the sheets."),
    (IHS_638_COL, "IHS_638", "IHS or 638 indicator",
     "Programme context is missing, which matters because IHS and 638 providers are exempt from "
     "some AHCCCS controls."),
    (COS_CODE_COL or COS_DESC_COL, "COS_CODE or COS_DESC", "category of service",
     "A_Provider_Profile cannot list the categories of service."),
    (CLAIM_ID_COL or CLAIM_HDR_COL, "ClaimID or Claim_Hdr_ID", "claim identity",
     "WEAKENS EVERY PART E SHEET. Each row is supposed to be an instruction to pull a record, and "
     "without a claim identifier a row names only a provider, a member, a date and a code."),
    (LINE_NO_COL, "LN_NO", "line number", "A claim line cannot be named exactly."),
    (MODS[0] if MODS else None, ", ".join(MODIFIER_COLS), "procedure modifiers",
     "WEAKENS THE WHOLE OF PART D AND OVERSTATES IT. Group lines carry the HQ modifier and cannot "
     "be identified without these columns, so group time that AHCCCS counts per member for the "
     "whole group length is counted in full. A clinician running ordinary group programming will "
     "then show days that look arithmetically impossible. The modifier mix sheet is also empty."),
    (DIAGS[1] if len(DIAGS) > 1 else None, ", ".join(DIAG_COLS[1:]),
     "secondary diagnosis positions",
     "The workbook cannot tell a focus diagnosis that is the reason for the visit from one that "
     "is a secondary note on the claim."),
]

_av_rows = []
for found, name, what, consequence in OPTIONAL:
    _av_rows.append({
        "column_or_columns": name,
        "used_for": what,
        "found_in_this_table": "yes" if found else "NO",
        "column_name_actually_used": found or "",
        "what_is_lost_without_it": "" if found else consequence,
        "analysis_status": "available" if found else (
            "NOT RUN" if consequence.startswith("NOT RUN") else
            "WEAKENED" if consequence.startswith("WEAKENS") else "reduced detail"),
    })
data_availability = pd.DataFrame(
    [{"column_or_columns": c, "used_for": w, "found_in_this_table": "yes",
      "column_name_actually_used": c, "what_is_lost_without_it": "",
      "analysis_status": "required, and present"} for c, w in REQUIRED] + _av_rows)
_assert_unique(data_availability["column_or_columns"], "data availability rows")

MISSING_COLS = data_availability.loc[
    data_availability.found_in_this_table == "NO", "column_or_columns"].tolist()
NOT_RUN = data_availability.loc[data_availability.analysis_status == "NOT RUN",
                                "what_is_lost_without_it"].tolist()
WEAKENED = data_availability.loc[data_availability.analysis_status == "WEAKENED",
                                 "what_is_lost_without_it"].tolist()
CANNOT_DO = NOT_RUN + WEAKENED

print(f"Columns on {SERVICE_TABLE}: {len(_src_cols)}")
print(f"Required columns: all {len(REQUIRED)} present")
print(f"Modifier columns found: {MODS or 'NONE'}")
print(f"Diagnosis columns found: {DIAGS or 'NONE'}")
print(f"Revenue code column found: {REV_CODE_COL or REV_DESC_COL or 'NONE'}")
print(f"Claim receipt date column found: {RECEIPT_KEY_COL or RECEIPT_DT_COL or 'NONE'}")
if MISSING_COLS:
    print(f"\nOptional columns NOT found ({len(MISSING_COLS)}):")
    for c in MISSING_COLS:
        print(f"   {c}")
else:
    print("\nEvery optional column this notebook looks for is present.")
for _c in NOT_RUN:
    print(f"\n{_c}")
for _c in WEAKENED:
    print(f"\n{_c}")
if not NOT_RUN and not WEAKENED:
    print("\nNo analysis in this notebook had to be skipped or weakened.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Load every claim for the two providers
# MAGIC
# MAGIC This cell loads the claims. Read the first sentence of the Criteria sheet before reading any
# MAGIC number that comes out of it.
# MAGIC
# MAGIC **Every claim for these two providers is loaded, not only the focus codes.** The focus codes and
# MAGIC diagnoses are highlighted later. They are not used to restrict the data, for the four reasons
# MAGIC in the first cell of this notebook.
# MAGIC
# MAGIC Each line is given:
# MAGIC
# MAGIC - a unit basis from the code dictionary, and the minutes it represents if it is a timed code;
# MAGIC - a clean modifier set, with the placeholder values this extract uses for "no modifier"
# MAGIC   stripped out, so the modifier mix on a sheet is the real one;
# MAGIC - the days its service span covers, because a line spanning a fortnight would otherwise look
# MAGIC   like a fortnight of work on one Tuesday.

# COMMAND ----------

_basis_map = F.create_map(*[F.lit(x) for kv in CODE_BASIS.items() for x in kv])
_minutes_map = F.create_map(*[F.lit(x) for kv in MINUTES_PER_UNIT.items() for x in kv])


def _clean_token(col, nulls):
    """Upper case, trimmed, cast to string, with placeholder values turned into a real null.
    Modifiers are cast to string FIRST: when all four are null in a partition they can arrive
    typed numeric and comparing them with 'HQ' then throws CAST_INVALID_INPUT."""
    c = F.upper(F.trim(F.coalesce(F.col(col).cast("string"), F.lit(""))))
    return F.when(c.isin(sorted(nulls)), F.lit(None).cast("string")).otherwise(c)


# Modifier array with the placeholders removed. array_remove on a null is avoided deliberately:
# in this project F.array_remove(F.array(...), None) returned NULL once and a whole test then
# found nothing across every provider. Placeholders are turned into "" and filtered by a
# comparison that cannot return null instead.
if MODS:
    _mod_raw = F.array(*[_clean_token(c, MOD_NULL_TOKENS) for c in MODS])
    _mod_arr = F.array_sort(F.array_distinct(
        F.filter(F.transform(_mod_raw, lambda x: F.coalesce(x, F.lit(""))), lambda x: x != F.lit(""))))
else:
    _mod_arr = F.array().cast("array<string>")


def _money(col):
    return F.coalesce(F.col(col).cast("double"), F.lit(0.0)) if col else F.lit(None).cast("double")


def _txt(col):
    return F.coalesce(F.col(col).cast("string"), F.lit("")) if col else F.lit("").cast("string")


def _receipt_date():
    if RECEIPT_KEY_COL:
        return F.to_date(F.col(RECEIPT_KEY_COL).cast("string"), "yyyyMMdd")
    if RECEIPT_DT_COL:
        return F.to_date(F.col(RECEIPT_DT_COL).cast("string"))
    return F.lit(None).cast("date")


def prepare(sdf):
    """Turn raw claim lines into the shape every part of this notebook works from."""
    out = (
        sdf
        .withColumn(PROC_COL, F.upper(F.trim(F.coalesce(F.col(PROC_COL).cast("string"), F.lit("")))))
        .withColumn("srv_date", F.to_date(F.col(SRV_BEG_COL).cast("string"), "yyyyMMdd"))
        .withColumn("_end_raw", F.to_date(F.col(SRV_END_COL).cast("string"), "yyyyMMdd"))
        .filter(F.col("srv_date").isNotNull())
        .withColumn("end_date", F.when(F.col("_end_raw").isNull() |
                                       (F.col("_end_raw") < F.col("srv_date")), F.col("srv_date"))
                                 .otherwise(F.col("_end_raw")))
        .withColumn("span_days", F.datediff("end_date", "srv_date") + F.lit(1))
        .withColumn("span_too_long", F.col("span_days") > F.lit(MAX_SPAN_DAYS))
        .withColumn("eff_span", F.when(F.col("span_too_long"), F.lit(1)).otherwise(F.col("span_days")))
        .withColumn("month", F.trunc("srv_date", "month"))
        .withColumn("recv_date", _receipt_date())
        .withColumn("qty", F.coalesce(F.col(QTY_COL).cast("double"), F.lit(0.0)))
        .withColumn("paid", F.coalesce(F.col(PMT_COL).cast("double"), F.lit(0.0)))
        .withColumn("billed", _money(BILL_COL))
        .withColumn("allowed", _money(ALLOW_COL))
        .withColumn("mod_set", F.array_join(_mod_arr, ","))
        .withColumn("is_group", F.array_contains(_mod_arr, F.lit(GROUP_MODIFIER)))
        .withColumn("is_family", F.size(F.array_intersect(
            _mod_arr, F.array(*[F.lit(m) for m in sorted(FAMILY_MODIFIERS)]))) > 0)
        .withColumn("basis", F.coalesce(_basis_map[F.col(PROC_COL)], F.lit("UNCLASSIFIED")))
        .withColumn("min_per_unit", F.coalesce(_minutes_map[F.col(PROC_COL)], F.lit(0)))
        .withColumn("line_minutes", F.col("qty") * F.col("min_per_unit"))
        .withColumn("is_session", F.col("basis") == F.lit("PER_SESSION"))
        .withColumn("sessions", F.when(F.col("basis") == F.lit("PER_SESSION"), F.col("qty"))
                                 .otherwise(F.lit(0.0)))
        .withColumn("is_per_diem_any", F.col(PROC_COL).isin(sorted(PER_DIEM_ALL)))
        .withColumn("is_per_diem_testable", F.col(PROC_COL).isin(sorted(PER_DIEM_TESTABLE)))
        .withColumn("is_unverified_basis", F.col(PROC_COL).isin(sorted(UNVERIFIED_BASIS_CODES)))
        .withColumn("is_focus_code", F.col(PROC_COL).isin(FOCUS_CODES))
        .withColumn("proc_desc", _txt(PROC_DESC_COL))
    )
    # Primary diagnosis, cleaned the same way everywhere so a decimal point or a lower case
    # letter cannot make a focus diagnosis fail to match.
    if PRIMARY_DIAG_COL:
        out = out.withColumn("diag1", F.upper(F.trim(F.regexp_replace(
            F.coalesce(F.col(PRIMARY_DIAG_COL).cast("string"), F.lit("")), r"\.", ""))))
    else:
        out = out.withColumn("diag1", F.lit("").cast("string"))
    out = (out.withColumn("diag1", F.when(F.col("diag1") == F.lit(""), F.lit("(no diagnosis on the line)"))
                                    .otherwise(F.col("diag1")))
              .withColumn("diag1_desc", _txt(PRIMARY_DIAG_DESC_COL))
              .withColumn("is_focus_diag", F.col("diag1").isin(FOCUS_DIAGS)))
    # Any-position focus diagnosis, reported alongside the primary so the investigator can see
    # whether a focus diagnosis is the reason for the visit or a secondary note.
    if DIAGS:
        _dz = F.array(*[F.upper(F.trim(F.regexp_replace(
            F.coalesce(F.col(c).cast("string"), F.lit("")), r"\.", ""))) for c in DIAGS])
        out = out.withColumn("focus_diag_any_position",
                             F.size(F.array_intersect(
                                 _dz, F.array(*[F.lit(d) for d in FOCUS_DIAGS]))) > 0)
    else:
        out = out.withColumn("focus_diag_any_position", F.lit(False))
    # Context columns, each only if the table has it.
    out = (out
           .withColumn("sprov_name", _txt(SPROV_NAME_COL))
           .withColumn("sprov_type", _txt(SPROV_TYPE_COL))
           .withColumn("sprov_type_desc", _txt(SPROV_TYPE_DESC_COL))
           .withColumn("sprov_susp", _txt(SPROV_SUSP_COL))
           .withColumn("bprov_name", _txt(BPROV_NAME_COL))
           .withColumn("bprov_type_desc", _txt(BPROV_TYPE_DESC_COL))
           .withColumn("bprov_susp", _txt(BPROV_SUSP_COL))
           .withColumn("rprov_id", _clean_token(RPROV_COL, ID_NULL_TOKENS) if have(RPROV_COL)
                       else F.lit(None).cast("string"))
           .withColumn("rprov_name", _txt(RPROV_NAME_COL))
           .withColumn("rprov_susp", _txt(RPROV_SUSP_COL))
           .withColumn("pos_code", _txt(POS_CODE_COL))
           .withColumn("pos_desc", _txt(POS_DESC_COL))
           .withColumn("clm_typ", _txt(CLM_TYP_COL))
           .withColumn("form_typ", _txt(FORM_TYP_COL))
           .withColumn("rev_desc", _txt(REV_DESC_COL))
           .withColumn("lob", _txt(LOB_COL))
           .withColumn("lob_org", _txt(LOB_ORG_COL))
           .withColumn("ihs_638", _txt(IHS_638_COL))
           .withColumn("cos_code", _txt(COS_CODE_COL))
           .withColumn("cos_desc", _txt(COS_DESC_COL))
           .withColumn("pa_no", _clean_token(PA_COL, ID_NULL_TOKENS) if PA_COL
                       else F.lit(None).cast("string"))
           .withColumn("claim_id", _txt(CLAIM_ID_COL))
           .withColumn("claim_hdr", _txt(CLAIM_HDR_COL))
           .withColumn("line_no", _txt(LINE_NO_COL))
           .withColumn("rev_code", _clean_token(REV_CODE_COL, ID_NULL_TOKENS) if REV_CODE_COL
                       else F.lit(None).cast("string")))
    if DOB_COL:
        out = (out.withColumn("_dob", F.to_date(F.col(DOB_COL).cast("string")))
                  .withColumn("member_age", F.when(F.col("_dob").isNotNull(),
                                                   F.floor(F.months_between("srv_date", "_dob") / 12))))
    else:
        out = out.withColumn("member_age", F.lit(None).cast("double"))
    return out


_raw = spark.table(SERVICE_TABLE).withColumn(
    "_sp", F.trim(F.coalesce(F.col(SPROV_COL).cast("string"), F.lit(""))))
df_line = prepare(_raw.filter(F.col("_sp").isin(TARGETS)).withColumn(SPROV_COL, F.col("_sp")))

N_LINES_ALL = df_line.count()
if FOCUS_ONLY:
    df_line = df_line.filter(F.col("is_focus_code") | F.col("is_focus_diag"))
    print("focus_only is ON. The claims have been restricted to lines that carry a focus code or "
          "a focus primary diagnosis. Read the Criteria sheet: the clinician day hours are now "
          "too low, every share of book is measured against a denominator that holds almost "
          "nothing else so it sits near 100% by construction and tells you nothing, and the same "
          "day conflict tests cannot see the per diem lines they need.")
df_line = df_line.cache()
N_LINES = df_line.count()

print(f"\nClaim lines for the two providers: {N_LINES:,}"
      + (f" (of {N_LINES_ALL:,} before focus_only)" if FOCUS_ONLY else ""))
if N_LINES == 0:
    print("WARNING: neither provider has any claim line in this table. Check the IDs. Every sheet "
          "below will be a note instead of a table.")
_per_prov = (df_line.groupBy(SPROV_COL)
             .agg(F.count(F.lit(1)).alias("lines"),
                  F.round(F.sum("paid"), 2).alias("paid"),
                  F.countDistinct(MEMBER_COL).alias("members"),
                  F.countDistinct("srv_date").alias("service_days"))
             .toPandas())
print(_per_prov.to_string(index=False))
_absent_target = [t for t in TARGETS if t not in set(_per_prov[SPROV_COL].astype(str))]
if _absent_target:
    print(f"Targets with no claims in this table: {', '.join(_absent_target)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. One row per line per day
# MAGIC
# MAGIC Builds the day level copy of the claims. A line covering one day is unchanged. A line covering
# MAGIC several days becomes several rows with its units, dollars and minutes divided evenly between
# MAGIC them, so no single day inherits a fortnight of billing. Spans longer than the limit in the
# MAGIC settings are treated as a data error, left on the begin date and counted in their own column.
# MAGIC
# MAGIC This cell also applies the group modifier treatment chosen in the settings and keeps the
# MAGIC minutes it set aside in a column of its own, so a reader can always see what was left out.

# COMMAND ----------

_span_src = df_line if SPAN_TREATMENT == "prorate" else df_line.filter(F.col("span_days") == 1)

df_day = (
    _span_src
    .withColumn("srv_date", F.explode(F.expr(
        "sequence(srv_date, date_add(srv_date, cast(eff_span as int) - 1))")))
    .withColumn("month", F.trunc("srv_date", "month"))
    .withColumn("day_qty", F.col("qty") / F.col("eff_span"))
    .withColumn("day_paid", F.col("paid") / F.col("eff_span"))
    .withColumn("day_billed", F.col("billed") / F.col("eff_span"))
    .withColumn("day_minutes", F.col("line_minutes") / F.col("eff_span"))
    .withColumn("day_sessions", F.col("sessions") / F.col("eff_span"))
)

if HQ_TREATMENT == "divide":
    _keys = [SPROV_COL, "srv_date", PROC_COL, "mod_set"]
    _gs = (df_day.filter(F.col("is_group")).groupBy(*_keys)
           .agg(F.countDistinct(MEMBER_COL).alias("group_size")))
    df_day = (df_day.join(_gs, _keys, "left")
              .withColumn("group_size", F.coalesce(F.col("group_size"), F.lit(1)))
              .withColumn("clin_minutes",
                          F.col("day_minutes") / F.greatest(F.col("group_size"), F.lit(1))))
elif HQ_TREATMENT == "full":
    df_day = (df_day.withColumn("group_size", F.lit(1))
                    .withColumn("clin_minutes", F.col("day_minutes")))
else:
    df_day = (df_day.withColumn("group_size", F.lit(1))
                    .withColumn("clin_minutes", F.when(F.col("is_group"), F.lit(0.0))
                                                 .otherwise(F.col("day_minutes"))))

df_day = (df_day
          .withColumn("group_minutes_set_aside",
                      F.when(F.col("is_group"), F.col("day_minutes") - F.col("clin_minutes"))
                       .otherwise(F.lit(0.0)))
          .cache())

N_DAY_ROWS = df_day.count()
_multi = df_line.filter(F.col("span_days") > 1).count()
_long = df_line.filter(F.col("span_too_long")).count()
print(f"Day level rows: {N_DAY_ROWS:,} (from {N_LINES:,} claim lines)")
print(f"Lines whose service span covers more than one day: {_multi:,} "
      f"({'spread across the days they cover' if SPAN_TREATMENT == 'prorate' else 'EXCLUDED from the day arithmetic'})")
print(f"Lines with a span longer than {MAX_SPAN_DAYS} days, treated as a data error and left on "
      f"the begin date: {_long:,}")
if N_LINES:
    _rng = df_line.agg(F.min("srv_date").alias("a"), F.max("srv_date").alias("b"),
                       F.min("recv_date").alias("c"), F.max("recv_date").alias("d")).first()
    DATA_START, DATA_END = _rng["a"], _rng["b"]
    RECV_START, RECV_END = _rng["c"], _rng["d"]
    print(f"Service dates run {DATA_START} to {DATA_END}")
    print(f"Claim receipt dates run {RECV_START} to {RECV_END}")
else:
    DATA_START = DATA_END = RECV_START = RECV_END = None


def _safe(num, den):
    return F.when(F.col(den) > 0, F.col(num) / F.col(den))


def _rate(a, b):
    """Division for pandas that returns None rather than an infinity or a NaN."""
    try:
        a, b = float(a), float(b)
    except (TypeError, ValueError):
        return None
    return round(a / b, 4) if b else None


def _ensure_cols(df, cols):
    """Add any missing column as a blank rather than letting a sheet lose it. When a provider
    billed none of the focus codes, the zero rows built for them carry fewer columns than the
    real rows, and a sheet whose columns change shape between runs is a sheet nobody can read."""
    if df is None or not len(df):
        return df
    out = df.copy()
    for c in cols:
        if c not in out.columns:
            out[c] = None
    return out


def _present(df, cols):
    """The subset of cols that this frame actually has, for printing."""
    return [c for c in cols if df is not None and c in getattr(df, "columns", [])]

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Part A. Who these two providers are
# MAGIC
# MAGIC One row for each provider, side by side, so they can be compared at a glance. It answers the
# MAGIC questions an investigator asks before looking at any billing: what is this provider, is it a
# MAGIC person or an organisation, is a payment suspension set, how long have they been billing, how
# MAGIC much have they been paid, how many members, who submits their claims and who refers to them.
# MAGIC
# MAGIC **Why the person or organisation question comes first.** A servicing provider ID on this table
# MAGIC may be an organisational NPI rather than one human being. If it is an organisation, then adding
# MAGIC up the hours billed in a day and comparing the total against 24 is meaningless, because several
# MAGIC members of staff bill under one number. Both of these providers were named as individuals in
# MAGIC the request, so the clinician day arithmetic in Part D should be meaningful. This cell checks
# MAGIC that against the provider type description on the claims rather than assuming it.
# MAGIC
# MAGIC **About the suspension indicator.** The raw value is reported exactly as the claims carry it,
# MAGIC with no interpretation. If every provider in the extract carries the same value, the column is
# MAGIC not populated in a useful way and the Data_Availability sheet says so.

# COMMAND ----------

# Type descriptions that denote a natural person rather than an organisation. Matched as whole
# descriptions where possible and as a leading phrase otherwise, because this is the evidence the
# claims actually carry about whether a clinician day total means anything.
INDIVIDUAL_TYPE_PATTERNS = [
    "MD-PHYSICIAN", "DO-PHYSICIAN", "PHYSICIANS ASSISTANT", "PHYSICIAN ASSISTANT",
    "REGISTERED NURSE PRACTITIONER", "CLINICAL NURSE SPECIALIST", "CERTIFIED NURSE-MIDWIFE",
    "CERTIFIED REGISTERED NURSE ANESTHETIST", "PSYCHOLOGIST", "PSYCHIATRIST",
    "LICENSED PROFESSIONAL COUNSELOR", "LICENSED CLINICAL SOCIAL WORKER",
    "LICENSED MARRIAGE & FAMILY THERAPIST", "LICENSED MARRIAGE AND FAMILY THERAPIST",
    "LIC INDEP ADDICTION COUNS", "LICENSED INDEPENDENT SUBSTANCE ABUSE COUNSELOR",
    "BOARD CERTIFIED BEHAVIOR ANALYST", "DENTIST", "OPTOMETRIST", "PODIATRIST",
    "PHYSICAL THERAPIST", "OCCUPATIONAL THERAPIST", "SPEECH/HEARING THERAPIST",
    "SPEECH LANGUAGE PATHOLOGY ASSISTANT", "AUDIOLOGIST", "REGISTERED DIETICIAN",
    "AFFILIATED DENTAL HYGIENIST", "SCHOOL BASED NURSE", "NURSE PRACTITIONER",
]
ORGANISATION_TYPE_PATTERNS = [
    "HOSPITAL", "CLINIC", "PHARMACY", "FACILITY", "AGENCY", "CENTER", "CENTRE", "HOME",
    "LABORATORY", "SUPPLIER", "TRANSPORTATION", "SCHOOL BASED BUS", "FQHC", "NURSING HOME",
    "RESID TRMT CTR", "SUBACUTE", "SHELTER", "HOUSING", "ATTENDANT CARE", "HABILITATION",
    "DIALYSIS", "VISION CENTER", "TRAVEL SERVICES", "AMBULATORY SURGICAL",
]


def person_or_organisation(type_desc):
    """Decide from the TYPE DESCRIPTION, which is the only evidence on the claim about this.
    Returns the verdict and the reason, because the verdict decides whether the clinician day
    arithmetic in Part D means anything and a bare verdict would not be checkable."""
    t = (type_desc or "").strip().upper()
    if not t:
        return ("cannot tell: no provider type description on the claims",
                "SProv_Type_Desc is empty or missing")
    for p in INDIVIDUAL_TYPE_PATTERNS:
        if p in t:
            return ("individual practitioner",
                    f"the provider type description '{type_desc}' names a licensed individual")
    for p in ORGANISATION_TYPE_PATTERNS:
        if p in t:
            return ("organisation",
                    f"the provider type description '{type_desc}' names a facility or an "
                    f"organisation, so hours billed in a day are hours across its whole staff "
                    f"and must NOT be read as one person's day")
    return ("cannot tell from the type description",
            f"the provider type description '{type_desc}' is not on either list in this notebook")


def name_shape(name):
    n = (name or "").strip()
    if not n:
        return "no name on the claims"
    if re.match(r"^[A-Za-z'\-]+,\s*[A-Za-z]", n):
        return "looks like a person: surname, forename"
    return "does not look like a person's name in this extract"


def profile_for(pid):
    d = df_line.filter(F.col(SPROV_COL) == pid)
    n = d.count()
    row = {"provider_id": pid,
           "name_in_the_request": TARGET_NAMES_GIVEN.get(pid, "(none supplied)")}
    if n == 0:
        row["claim_lines"] = 0
        row["note"] = "this provider has no claim line in the table at all"
        return row
    ident = (d.groupBy(SPROV_COL)
             .agg(F.concat_ws(" | ", F.array_sort(F.collect_set("sprov_name"))).alias("name_on_claims"),
                  F.concat_ws(" | ", F.array_sort(F.collect_set("sprov_type"))).alias("type_code"),
                  F.concat_ws(" | ", F.array_sort(F.collect_set("sprov_type_desc"))).alias("type_desc"),
                  F.concat_ws(" | ", F.array_sort(F.collect_set("sprov_susp"))).alias("susp_ind"))
             .first())
    agg = d.agg(
        F.count(F.lit(1)).alias("lines"),
        F.round(F.sum("paid"), 2).alias("paid"),
        F.round(F.sum("billed"), 2).alias("billed"),
        F.round(F.sum("allowed"), 2).alias("allowed"),
        F.round(F.sum("qty"), 1).alias("units"),
        F.countDistinct(MEMBER_COL).alias("distinct_members"),
        F.countDistinct("srv_date").alias("distinct_service_days"),
        F.countDistinct(PROC_COL).alias("distinct_codes"),
        F.countDistinct("diag1").alias("distinct_primary_diagnoses"),
        F.countDistinct(BPROV_COL).alias("distinct_billing_entities"),
        F.countDistinct("rprov_id").alias("distinct_referring_providers"),
        F.countDistinct("claim_hdr").alias("distinct_claims"),
        F.min("srv_date").alias("first_service_date"),
        F.max("srv_date").alias("last_service_date"),
        F.min("recv_date").alias("first_claim_received"),
        F.max("recv_date").alias("last_claim_received"),
        F.concat_ws(", ", F.array_sort(F.collect_set(F.when(F.col("lob") != F.lit(""), F.col("lob"))))).alias("lines_of_business"),
        F.concat_ws(", ", F.array_sort(F.collect_set(F.when(F.col("pos_desc") != F.lit(""), F.col("pos_desc"))))).alias("places_of_service"),
        F.concat_ws(", ", F.array_sort(F.collect_set(F.when(F.col("form_typ") != F.lit(""), F.col("form_typ"))))).alias("claim_forms"),
        F.concat_ws(", ", F.array_sort(F.collect_set(F.when(F.col("cos_desc") != F.lit(""), F.col("cos_desc"))))).alias("categories_of_service"),
        F.sum(F.col("is_group").cast("int")).alias("group_modifier_lines"),
        F.sum(F.when(F.col("pa_no").isNotNull(), F.lit(1)).otherwise(F.lit(0))).alias("lines_with_a_prior_auth"),
        F.round(F.avg("member_age"), 1).alias("mean_member_age_at_service"),
    ).first()
    focus = d.filter(F.col("is_focus_code")).agg(
        F.count(F.lit(1)).alias("lines"), F.round(F.sum("paid"), 2).alias("paid"),
        F.countDistinct(MEMBER_COL).alias("members")).first()
    fdiag = d.filter(F.col("is_focus_diag")).agg(
        F.count(F.lit(1)).alias("lines"), F.round(F.sum("paid"), 2).alias("paid"),
        F.countDistinct(MEMBER_COL).alias("members")).first()

    verdict, why = person_or_organisation(ident["type_desc"])
    row.update({
        "name_on_claims": ident["name_on_claims"],
        "name_matches_the_request":
            "yes" if (TARGET_NAMES_GIVEN.get(pid, "").strip().upper().replace(" ", "")
                      in (ident["name_on_claims"] or "").upper().replace(" ", "")) else
            "NO: check you have the right provider",
        "name_shape": name_shape(ident["name_on_claims"]),
        "provider_type_code": ident["type_code"],
        "provider_type_description": ident["type_desc"],
        "person_or_organisation": verdict,
        "person_or_organisation_evidence": why,
        "clinician_day_arithmetic_meaningful":
            "yes" if verdict == "individual practitioner" else
            "NO: do not read a day total as one person's day" if verdict == "organisation" else
            "unknown: confirm with provider enrolment before using Part D",
        "payment_suspension_indicator_raw": ident["susp_ind"],
        "claim_lines": n,
        "paid": agg["paid"],
        "billed": agg["billed"],
        "allowed": agg["allowed"],
        "paid_over_billed": _rate(agg["paid"], agg["billed"]),
        "units": agg["units"],
        "distinct_claims": agg["distinct_claims"],
        "distinct_members": agg["distinct_members"],
        "distinct_service_days": agg["distinct_service_days"],
        "distinct_procedure_codes": agg["distinct_codes"],
        "distinct_primary_diagnoses": agg["distinct_primary_diagnoses"],
        "distinct_billing_entities": agg["distinct_billing_entities"],
        "distinct_referring_providers": agg["distinct_referring_providers"],
        "first_service_date": agg["first_service_date"],
        "last_service_date": agg["last_service_date"],
        "first_claim_received": agg["first_claim_received"],
        "last_claim_received": agg["last_claim_received"],
        "lines_of_business": agg["lines_of_business"],
        "places_of_service": agg["places_of_service"],
        "claim_forms": agg["claim_forms"],
        "categories_of_service": agg["categories_of_service"],
        "group_modifier_lines": agg["group_modifier_lines"],
        "lines_with_a_prior_auth": agg["lines_with_a_prior_auth"],
        "mean_member_age_at_service": agg["mean_member_age_at_service"],
        "focus_code_lines": focus["lines"],
        "focus_code_paid": focus["paid"],
        "focus_code_members": focus["members"],
        "focus_code_share_of_paid": _rate(focus["paid"], agg["paid"]),
        "focus_diagnosis_lines": fdiag["lines"],
        "focus_diagnosis_paid": fdiag["paid"],
        "focus_diagnosis_members": fdiag["members"],
        "focus_diagnosis_share_of_paid": _rate(fdiag["paid"], agg["paid"]),
        "focus_diagnosis_share_of_members": _rate(fdiag["members"], agg["distinct_members"]),
    })
    return row


PROFILES = {pid: profile_for(pid) for pid in TARGETS}
profile_df = pd.DataFrame([PROFILES[p] for p in TARGETS])
# Transposed, because side by side means one column per provider when there are only two of them
# and thirty something measures.
profile_side_by_side = profile_df.set_index("provider_id").T.reset_index()
profile_side_by_side.columns = ["measure"] + [f"provider_{p}" for p in profile_df["provider_id"]]
_assert_unique(profile_side_by_side.columns, "provider profile columns")
print(profile_side_by_side.to_string(index=False))

IS_PERSON = {p: PROFILES[p].get("person_or_organisation") == "individual practitioner"
             for p in TARGETS}
SUSP_VALUES = {p: PROFILES[p].get("payment_suspension_indicator_raw") for p in TARGETS}

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. Part B. The focus lens
# MAGIC
# MAGIC Three tables, and then the one the request is really about.
# MAGIC
# MAGIC 1. **Each focus code, for each provider.** Lines, units, paid dollars, members, service
# MAGIC    days, units per member day, dollars per unit, and what share of this provider's whole book
# MAGIC    the code represents. The share column is the important one: it is the difference between a
# MAGIC    code being this provider's practice and being a corner of it.
# MAGIC 2. **Each focus primary diagnosis, for each provider**, the same way.
# MAGIC 3. **Focus code by focus diagnosis**, with lines, members and dollars in each cell, plus what
# MAGIC    share of each code's volume carries one of the focus diagnoses.
# MAGIC
# MAGIC **How to read units per member day.** It is units divided by the number of distinct member days
# MAGIC on which the code was billed. For H0004 and T1016, one unit is 15 minutes, so 8 units is two
# MAGIC hours with one person in one day and that is where the published documentation requirement
# MAGIC starts. For H0031 the number is sessions per member day and 15 minutes does not come into it.
# MAGIC For H2038 the number should be at or below 1 if it really is a per diem.

# COMMAND ----------

def code_rows(pid):
    d = df_line.filter(F.col(SPROV_COL) == pid)
    tot = d.agg(F.round(F.sum("paid"), 2).alias("paid"),
                          F.count(F.lit(1)).alias("lines"),
                          F.round(F.sum("qty"), 1).alias("units"),
                          F.countDistinct(MEMBER_COL).alias("members")).first()
    if tot is None or not tot["lines"]:
        return pd.DataFrame()
    per_code = (d.groupBy(PROC_COL)
                .agg(F.count(F.lit(1)).alias("lines"),
                     F.round(F.sum("qty"), 2).alias("units"),
                     F.round(F.sum("paid"), 2).alias("paid"),
                     F.round(F.sum("billed"), 2).alias("billed"),
                     F.countDistinct(MEMBER_COL).alias("members"),
                     F.countDistinct("srv_date").alias("service_days"),
                     F.countDistinct(F.concat_ws("|", F.col(MEMBER_COL).cast("string"),
                                                 F.col("srv_date").cast("string"))).alias("member_days"),
                     F.countDistinct(BPROV_COL).alias("billing_entities"),
                     F.max("qty").alias("largest_units_on_one_line"),
                     F.concat_ws(" | ", F.array_sort(F.collect_set(
                         F.when(F.col("proc_desc") != F.lit(""), F.col("proc_desc"))))).alias("procedure_description"),
                     F.concat_ws(", ", F.array_sort(F.collect_set(
                         F.when(F.col("mod_set") != F.lit(""), F.col("mod_set"))))).alias("modifier_sets_seen"),
                     F.sum(F.col("is_group").cast("int")).alias("group_modifier_lines"),
                     F.sum(F.col("is_focus_diag").cast("int")).alias("lines_with_a_focus_diagnosis"),
                     F.countDistinct(F.when(F.col("is_focus_diag"), F.col(MEMBER_COL))).alias("members_with_a_focus_diagnosis"))
                .toPandas())
    # Per member day units, computed from the member day count so a provider billing one member
    # four times in a day is not flattered by a line count.
    mdu = (d.groupBy(PROC_COL, MEMBER_COL, "srv_date")
           .agg(F.sum("qty").alias("u"))
           .groupBy(PROC_COL)
           .agg(F.round(F.avg("u"), 3).alias("mean_units_per_member_day"),
                F.expr("percentile_approx(u, 0.5)").alias("median_units_per_member_day"),
                F.round(F.max("u"), 2).alias("max_units_in_one_member_day"))
           .toPandas())
    out = per_code.merge(mdu, on=PROC_COL, how="left")
    out.insert(0, "provider_id", pid)
    out["unit_basis"] = out[PROC_COL].map(lambda c: CODE_BASIS.get(c, "UNCLASSIFIED"))
    out["is_focus_code"] = out[PROC_COL].isin(FOCUS_CODES)
    out["paid_per_unit"] = [_rate(a, b) for a, b in zip(out.paid, out.units)]
    out["paid_per_member"] = [_rate(a, b) for a, b in zip(out.paid, out.members)]
    out["share_of_provider_paid"] = [_rate(a, tot["paid"]) for a in out.paid]
    out["share_of_provider_lines"] = [_rate(a, tot["lines"]) for a in out.lines]
    out["share_of_provider_units"] = [_rate(a, tot["units"]) for a in out.units]
    out["share_of_provider_members"] = [_rate(a, tot["members"]) for a in out.members]
    out["share_of_code_lines_with_a_focus_diagnosis"] = [
        _rate(a, b) for a, b in zip(out.lines_with_a_focus_diagnosis, out.lines)]
    out["minutes_per_unit_used_here"] = out[PROC_COL].map(lambda c: MINUTES_PER_UNIT.get(c, 0))
    out["basis_note"] = out[PROC_COL].map(
        lambda c: FOCUS_CODE_NOTE.get(c, "not a focus code; basis from the dictionary in section 2"))
    return out


ALL_CODE_TABLES = {pid: code_rows(pid) for pid in TARGETS}
focus_codes_df = pd.concat(
    [t[t.is_focus_code] for t in ALL_CODE_TABLES.values() if len(t)], ignore_index=True) \
    if any(len(t) for t in ALL_CODE_TABLES.values()) else pd.DataFrame()
if len(focus_codes_df):
    focus_codes_df["_o"] = focus_codes_df[PROC_COL].map({c: i for i, c in enumerate(FOCUS_CODES)})
    focus_codes_df = focus_codes_df.sort_values(["provider_id", "_o"]).drop(columns="_o").reset_index(drop=True)

# A focus code a provider never billed must appear as a zero row rather than be absent, because
# an absent row reads as "not checked" and a zero row reads as "checked, none found".
_zero_rows = []
for pid in TARGETS:
    seen = set(focus_codes_df.loc[focus_codes_df.provider_id == pid, PROC_COL]) if len(focus_codes_df) else set()
    for c in FOCUS_CODES:
        if c not in seen:
            _zero_rows.append({"provider_id": pid, PROC_COL: c, "lines": 0, "units": 0.0,
                               "paid": 0.0, "members": 0, "service_days": 0,
                               "member_days": 0, "is_focus_code": True,
                               "unit_basis": CODE_BASIS.get(c, "UNCLASSIFIED"),
                               "basis_note": FOCUS_CODE_NOTE.get(c, ""),
                               "procedure_description": "this provider never billed this code"})
if _zero_rows:
    focus_codes_df = pd.concat([focus_codes_df, pd.DataFrame(_zero_rows)], ignore_index=True) \
        if len(focus_codes_df) else pd.DataFrame(_zero_rows)
FOCUS_CODE_SHEET_COLS = ["provider_id", PROC_COL, "procedure_description", "unit_basis",
                         "basis_note", "is_focus_code", "lines", "units", "paid",
                         "billed", "members", "service_days", "member_days",
                         "billing_entities", "mean_units_per_member_day",
                         "median_units_per_member_day", "max_units_in_one_member_day",
                         "largest_units_on_one_line", "paid_per_unit", "paid_per_member",
                         "share_of_provider_paid", "share_of_provider_lines",
                         "share_of_provider_units", "share_of_provider_members",
                         "lines_with_a_focus_diagnosis", "members_with_a_focus_diagnosis",
                         "share_of_code_lines_with_a_focus_diagnosis", "modifier_sets_seen",
                         "group_modifier_lines", "minutes_per_unit_used_here"]
_assert_unique(FOCUS_CODE_SHEET_COLS, "focus code sheet columns")
focus_codes_df = _ensure_cols(focus_codes_df, FOCUS_CODE_SHEET_COLS)
print("Focus codes:")
if len(focus_codes_df):
    print(focus_codes_df[_present(focus_codes_df,
                                  ["provider_id", PROC_COL, "unit_basis", "lines", "units",
                                   "paid", "members", "member_days",
                                   "mean_units_per_member_day", "max_units_in_one_member_day",
                                   "share_of_provider_paid"])].to_string(index=False))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 10. Part B continued. The focus diagnoses, and the code by diagnosis cross tabulation
# MAGIC
# MAGIC The diagnosis used here is the **primary** diagnosis on the line, `DIAG_CD_1`. A separate
# MAGIC column counts lines where a focus diagnosis appears in any of the four diagnosis positions, so
# MAGIC a reader can tell the difference between a focus diagnosis being the reason for the visit and
# MAGIC being a secondary note on the claim.
# MAGIC
# MAGIC The cross tabulation is the table the request is really asking for: of this provider's work on
# MAGIC these four codes, how much of it carries these four diagnoses. Read the final column first.
# MAGIC If a code's volume is almost entirely on one of the focus diagnoses that is a different picture
# MAGIC from the same code spread across a wide range of conditions.

# COMMAND ----------

def diag_rows(pid):
    d = df_line.filter(F.col(SPROV_COL) == pid)
    tot = d.agg(F.round(F.sum("paid"), 2).alias("paid"),
                          F.count(F.lit(1)).alias("lines"),
                          F.countDistinct(MEMBER_COL).alias("members")).first()
    if tot is None or not tot["lines"]:
        return pd.DataFrame()
    per = (d.groupBy("diag1")
           .agg(F.count(F.lit(1)).alias("lines"),
                F.round(F.sum("qty"), 2).alias("units"),
                F.round(F.sum("paid"), 2).alias("paid"),
                F.countDistinct(MEMBER_COL).alias("members"),
                F.countDistinct("srv_date").alias("service_days"),
                F.countDistinct(PROC_COL).alias("distinct_codes"),
                F.concat_ws(" | ", F.array_sort(F.collect_set(
                    F.when(F.col("diag1_desc") != F.lit(""), F.col("diag1_desc"))))).alias("diagnosis_description"),
                F.concat_ws(", ", F.array_sort(F.collect_set(
                    F.when(F.col("is_focus_code"), F.col(PROC_COL))))).alias("focus_codes_on_this_diagnosis"),
                F.round(F.sum(F.when(F.col("is_focus_code"), F.col("paid")).otherwise(0.0)), 2).alias("paid_on_focus_codes"))
           .toPandas())
    per.insert(0, "provider_id", pid)
    per["is_focus_diagnosis"] = per["diag1"].isin(FOCUS_DIAGS)
    per["share_of_provider_paid"] = [_rate(a, tot["paid"]) for a in per.paid]
    per["share_of_provider_lines"] = [_rate(a, tot["lines"]) for a in per.lines]
    per["share_of_provider_members"] = [_rate(a, tot["members"]) for a in per.members]
    per["share_of_this_diagnosis_on_focus_codes"] = [
        _rate(a, b) for a, b in zip(per.paid_on_focus_codes, per.paid)]
    return per.rename(columns={"diag1": "primary_diagnosis"})


FOCUS_DIAG_SHEET_COLS = ["provider_id", "primary_diagnosis", "diagnosis_description",
                         "is_focus_diagnosis", "lines", "units", "paid", "members",
                         "service_days", "distinct_codes", "focus_codes_on_this_diagnosis",
                         "paid_on_focus_codes", "share_of_provider_paid",
                         "share_of_provider_lines", "share_of_provider_members",
                         "share_of_this_diagnosis_on_focus_codes"]
_assert_unique(FOCUS_DIAG_SHEET_COLS, "focus diagnosis sheet columns")

ALL_DIAG_TABLES = {pid: diag_rows(pid) for pid in TARGETS}
focus_diags_df = pd.concat(
    [t[t.is_focus_diagnosis] for t in ALL_DIAG_TABLES.values() if len(t)], ignore_index=True) \
    if any(len(t) for t in ALL_DIAG_TABLES.values()) else pd.DataFrame()
_zero_d = []
for pid in TARGETS:
    seen = set(focus_diags_df.loc[focus_diags_df.provider_id == pid, "primary_diagnosis"]) \
        if len(focus_diags_df) else set()
    for dx in FOCUS_DIAGS:
        if dx not in seen:
            _zero_d.append({"provider_id": pid, "primary_diagnosis": dx, "lines": 0,
                            "paid": 0.0, "members": 0, "is_focus_diagnosis": True,
                            "diagnosis_description": "this provider never billed a line with this "
                                                     "as the primary diagnosis"})
if _zero_d:
    focus_diags_df = pd.concat([focus_diags_df, pd.DataFrame(_zero_d)], ignore_index=True) \
        if len(focus_diags_df) else pd.DataFrame(_zero_d)
if len(focus_diags_df):
    focus_diags_df["_o"] = focus_diags_df["primary_diagnosis"].map(
        {c: i for i, c in enumerate(FOCUS_DIAGS)})
    focus_diags_df = focus_diags_df.sort_values(["provider_id", "_o"]).drop(columns="_o").reset_index(drop=True)
    focus_diags_df = _ensure_cols(focus_diags_df, FOCUS_DIAG_SHEET_COLS)
    print("Focus primary diagnoses:")
    print(focus_diags_df[_present(focus_diags_df,
                                  ["provider_id", "primary_diagnosis", "lines", "paid",
                                   "members", "share_of_provider_paid"])]
          .to_string(index=False))

# The cross tabulation. Every focus code by focus diagnosis cell is written, including the empty
# ones, so a gap in the grid cannot be mistaken for a cell that was not computed.
_cross = (df_line.filter(F.col("is_focus_code"))
          .groupBy(SPROV_COL, PROC_COL, "diag1")
          .agg(F.count(F.lit(1)).alias("lines"),
               F.round(F.sum("qty"), 2).alias("units"),
               F.round(F.sum("paid"), 2).alias("paid"),
               F.countDistinct(MEMBER_COL).alias("members"),
               F.countDistinct("srv_date").alias("service_days"),
               F.concat_ws(" | ", F.array_sort(F.collect_set(
                   F.when(F.col("diag1_desc") != F.lit(""), F.col("diag1_desc"))))).alias("diagnosis_description"))
          .toPandas())
grid = []
for pid in TARGETS:
    code_tot = {}
    if len(ALL_CODE_TABLES.get(pid, pd.DataFrame())):
        t = ALL_CODE_TABLES[pid]
        code_tot = {r[PROC_COL]: (r["lines"], r["paid"], r["members"])
                    for _, r in t.iterrows()}
    for c in FOCUS_CODES:
        cl, cp, cm = code_tot.get(c, (0, 0.0, 0))
        hit_lines = hit_paid = 0.0
        for dx in FOCUS_DIAGS:
            m = _cross[(_cross[SPROV_COL] == pid) & (_cross[PROC_COL] == c) & (_cross.diag1 == dx)]
            r = m.iloc[0] if len(m) else None
            hit_lines += 0 if r is None else float(r["lines"])
            hit_paid += 0.0 if r is None else float(r["paid"])
            grid.append({
                "provider_id": pid, "procedure_code": c,
                "unit_basis": CODE_BASIS.get(c, "UNCLASSIFIED"),
                "primary_diagnosis": dx,
                "diagnosis_description": "" if r is None else r["diagnosis_description"],
                "lines": 0 if r is None else int(r["lines"]),
                "units": 0.0 if r is None else float(r["units"]),
                "paid": 0.0 if r is None else float(r["paid"]),
                "members": 0 if r is None else int(r["members"]),
                "service_days": 0 if r is None else int(r["service_days"]),
                "cell_share_of_this_code_lines": _rate(0 if r is None else r["lines"], cl),
                "cell_share_of_this_code_paid": _rate(0.0 if r is None else r["paid"], cp),
            })
        grid.append({
            "provider_id": pid, "procedure_code": c,
            "unit_basis": CODE_BASIS.get(c, "UNCLASSIFIED"),
            "primary_diagnosis": "ALL FOUR FOCUS DIAGNOSES TOGETHER",
            "diagnosis_description": "subtotal row, not a diagnosis",
            "lines": int(hit_lines), "units": None, "paid": round(hit_paid, 2),
            "members": None, "service_days": None,
            "cell_share_of_this_code_lines": _rate(hit_lines, cl),
            "cell_share_of_this_code_paid": _rate(hit_paid, cp),
        })
        grid.append({
            "provider_id": pid, "procedure_code": c,
            "unit_basis": CODE_BASIS.get(c, "UNCLASSIFIED"),
            "primary_diagnosis": "EVERY OTHER DIAGNOSIS",
            "diagnosis_description": "the rest of this code's volume, for comparison",
            "lines": int(cl - hit_lines), "units": None,
            "paid": round(float(cp) - hit_paid, 2) if cp else 0.0,
            "members": None, "service_days": None,
            "cell_share_of_this_code_lines": _rate(cl - hit_lines, cl),
            "cell_share_of_this_code_paid": _rate((float(cp) - hit_paid) if cp else 0.0, cp),
        })
code_by_diag_df = pd.DataFrame(grid)

# The modifier mix on the focus codes, with the placeholder values this extract uses for "no
# modifier" already stripped, so an empty modifier_set really means no modifier.
mod_mix_df = (df_line.filter(F.col("is_focus_code"))
              .groupBy(SPROV_COL, PROC_COL, "mod_set")
              .agg(F.count(F.lit(1)).alias("lines"),
                   F.round(F.sum("qty"), 2).alias("units"),
                   F.round(F.sum("paid"), 2).alias("paid"),
                   F.countDistinct(MEMBER_COL).alias("members"),
                   F.max(F.col("is_group")).alias("is_a_group_line"))
              .orderBy(SPROV_COL, PROC_COL, F.desc("lines"))
              .toPandas())
if len(mod_mix_df):
    mod_mix_df["modifier_set"] = mod_mix_df["mod_set"].replace("", "(no modifier on the line)")
    mod_mix_df = mod_mix_df.drop(columns=["mod_set"])
print(f"\nFocus code by focus diagnosis cells: {len(code_by_diag_df)}")
print(f"Focus code modifier combinations: {len(mod_mix_df)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 11. Part C. The whole book, so the lens has a denominator
# MAGIC
# MAGIC Two sheets per provider: every procedure code they billed, ranked by paid dollars, and
# MAGIC every primary diagnosis they billed, ranked the same way. The focus codes and focus diagnoses
# MAGIC are marked with a column rather than moved to the top, so their position in the ranking is
# MAGIC itself visible.
# MAGIC
# MAGIC **This is the sheet that answers "is this their practice or a corner of it".** If the four
# MAGIC focus codes sit at the top of the ranking and carry most of the dollars, the Part B numbers
# MAGIC describe the provider. If they sit halfway down a long list, the Part B numbers describe a
# MAGIC fraction of the provider and should be read that way.

# COMMAND ----------

ALL_CODES_SHEETS, ALL_DIAGS_SHEETS = {}, {}
for pid in TARGETS:
    t = ALL_CODE_TABLES.get(pid, pd.DataFrame())
    if len(t):
        t = t.sort_values("paid", ascending=False).reset_index(drop=True)
        t.insert(0, "rank_by_paid", np.arange(1, len(t) + 1))
    ALL_CODES_SHEETS[pid] = t
    dt = ALL_DIAG_TABLES.get(pid, pd.DataFrame())
    if len(dt):
        dt = dt.sort_values("paid", ascending=False).reset_index(drop=True)
        dt.insert(0, "rank_by_paid", np.arange(1, len(dt) + 1))
    ALL_DIAGS_SHEETS[pid] = dt
    if len(t):
        _f = t[t.is_focus_code]
        print(f"\nProvider {pid}: {len(t)} procedure codes billed. Focus codes rank "
              f"{', '.join(f'{r[PROC_COL]} at {int(r.rank_by_paid)}' for _, r in _f.iterrows()) or 'nowhere, none billed'}"
              f" of {len(t)}.")
        print(f"   The four focus codes are "
              f"{(_f.paid.sum() / max(t.paid.sum(), 1e-9)):.1%} of this "
              f"provider's paid dollars and {(_f.lines.sum() / max(t.lines.sum(), 1)):.1%} of "
              f"their lines.")
    if len(dt):
        _fd = dt[dt.is_focus_diagnosis]
        print(f"   {len(dt)} primary diagnoses billed. The four focus diagnoses are "
              f"{(_fd.paid.sum() / max(dt.paid.sum(), 1e-9)):.1%} of paid "
              f"dollars and cover {_fd.members.sum()} of this provider's member appearances.")

# Codes outside the AHCCCS behavioural health set, which is a finding in its own right rather
# than a classification problem.
absent_found = (df_line.filter(F.col("basis") == F.lit("NOT_IN_AHCCCS_BH_SET"))
                .groupBy(SPROV_COL, PROC_COL)
                .agg(F.count(F.lit(1)).alias("lines"),
                     F.round(F.sum("qty"), 2).alias("units"),
                     F.round(F.sum("paid"), 2).alias("paid"),
                     F.countDistinct(MEMBER_COL).alias("members"))
                .orderBy(F.desc("paid")).toPandas())
unverified_found = (df_line.filter(F.col("basis") == F.lit("UNVERIFIED"))
                    .groupBy(SPROV_COL, PROC_COL)
                    .agg(F.count(F.lit(1)).alias("lines"),
                         F.round(F.sum("qty"), 2).alias("units"),
                         F.round(F.sum("paid"), 2).alias("paid"),
                         F.countDistinct(MEMBER_COL).alias("members"))
                    .orderBy(F.desc("paid")).toPandas())
basis_cov = (df_line.groupBy(SPROV_COL, "basis")
             .agg(F.count(F.lit(1)).alias("lines"),
                  F.round(F.sum("paid"), 2).alias("paid"),
                  F.round(F.sum("qty"), 1).alias("units"))
             .orderBy(SPROV_COL, F.desc("paid")).toPandas())
print(f"\nCodes AHCCCS does not include in its behavioural health set, billed by these providers: "
      f"{len(absent_found)} rows")
print(f"Codes whose unit basis could not be verified, billed by these providers: "
      f"{len(unverified_found)} rows")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 12. Part D. The day
# MAGIC
# MAGIC One row for every calendar day on which each provider billed anything. It adds up the timed
# MAGIC minutes they billed that day across **every** procedure code and **every** billing entity, and
# MAGIC reports the days that go past 8, 12, 16 and 24 hours together with the dollars paid on them.
# MAGIC
# MAGIC **Four rules this arithmetic follows, and each one lowers the total rather than raising it.**
# MAGIC
# MAGIC 1. Only timed codes contribute minutes. Per session codes, H0031 among them, count as sessions
# MAGIC    and add zero minutes. Multiplying a session by a nominal length is the single most common way
# MAGIC    to manufacture an impossible day that never happened.
# MAGIC 2. Codes whose unit basis AHCCCS has never published contribute zero minutes. H2038 is one of
# MAGIC    them, so its units never become hours here.
# MAGIC 3. Group lines are handled as the setting says, and by default they are taken out completely.
# MAGIC    AHCCCS counts group time per member for the whole group, so eight members in a one hour group
# MAGIC    is 32 units from one hour of work. Leaving group lines in would turn ordinary group
# MAGIC    programming into an impossible day.
# MAGIC 4. A line spanning several days has its minutes spread over those days.
# MAGIC
# MAGIC **What this total does NOT include.** Only behavioural health codes in the dictionary in section
# MAGIC 2 have a unit basis. If one of these providers also bills timed services outside that set, those
# MAGIC minutes are not in the total, so a day total here is a floor and not a ceiling.
# MAGIC
# MAGIC **Who the hours belong to.** If Part A says this provider ID is an organisation, these hours are
# MAGIC its whole staff's hours and comparing them against one person's day means nothing. Part A says
# MAGIC which of the two providers that applies to.
# MAGIC
# MAGIC Two more day level tests sit here because they are arithmetic rather than policy: a per diem
# MAGIC code billed for more than one unit for one member in one day, and H2038 above one unit in a day.

# COMMAND ----------

day_src = df_day

clinician_day = (
    day_src.groupBy(SPROV_COL, "srv_date")
    .agg(F.round(F.sum("clin_minutes"), 1).alias("timed_minutes"),
         F.round(F.sum("clin_minutes") / 60.0, 2).alias("timed_hours"),
         F.round(F.sum("group_minutes_set_aside"), 1).alias("group_minutes_set_aside"),
         F.round(F.sum("day_sessions"), 2).alias("sessions_billed"),
         F.round(F.sum(F.when(F.col("is_per_diem_any"), F.col("day_qty")).otherwise(0.0)), 2).alias("per_diem_units"),
         F.round(F.sum("day_qty"), 2).alias("units_all_codes"),
         F.round(F.sum("day_paid"), 2).alias("paid_that_day"),
         F.countDistinct(MEMBER_COL).alias("members_that_day"),
         F.countDistinct(BPROV_COL).alias("billing_entities_that_day"),
         F.countDistinct(PROC_COL).alias("codes_that_day"),
         F.count(F.lit(1)).alias("day_rows"),
         F.sum(F.col("is_group").cast("int")).alias("group_lines"),
         F.sum(F.when(F.col("span_days") > 1, F.lit(1)).otherwise(F.lit(0))).alias("multi_day_span_lines"),
         F.concat_ws(", ", F.array_sort(F.collect_set(PROC_COL))).alias("codes_billed_that_day"),
         F.concat_ws(", ", F.array_sort(F.collect_set(
             F.when(F.col("is_focus_code"), F.col(PROC_COL))))).alias("focus_codes_that_day"),
         F.round(F.sum(F.when(F.col("is_focus_code"), F.col("day_paid")).otherwise(0.0)), 2).alias("paid_on_focus_codes_that_day"))
)
for h in HOUR_THRESHOLDS:
    clinician_day = clinician_day.withColumn(f"over_{h}_hours", F.col("timed_hours") > F.lit(h))
clinician_day = clinician_day.cache()

CLIN_DAY_SHEETS, DAY_SUMMARY = {}, []
for pid in TARGETS:
    cd = (clinician_day.filter(F.col(SPROV_COL) == pid)
          .orderBy(F.desc("timed_hours"), F.desc("paid_that_day"))
          .limit(DETAIL_ROWS).toPandas())
    CLIN_DAY_SHEETS[pid] = cd
    row = {"provider_id": pid,
           "person_or_organisation": PROFILES[pid].get("person_or_organisation", "unknown"),
           "read_hours_as_one_persons_day":
               PROFILES[pid].get("clinician_day_arithmetic_meaningful", "unknown"),
           "active_days": len(cd),
           "total_timed_hours": round(float(cd.timed_hours.sum()), 1) if len(cd) else 0.0,
           "group_minutes_set_aside": round(float(cd.group_minutes_set_aside.sum()), 1) if len(cd) else 0.0,
           "mean_timed_hours_per_active_day": round(float(cd.timed_hours.mean()), 2) if len(cd) else None,
           "median_timed_hours_per_active_day": round(float(cd.timed_hours.median()), 2) if len(cd) else None,
           "max_timed_hours_in_one_day": round(float(cd.timed_hours.max()), 2) if len(cd) else None,
           "mean_members_per_active_day": round(float(cd.members_that_day.mean()), 2) if len(cd) else None,
           "max_members_in_one_day": int(cd.members_that_day.max()) if len(cd) else None,
           "total_sessions_billed": round(float(cd.sessions_billed.sum()), 1) if len(cd) else 0.0}
    for h in HOUR_THRESHOLDS:
        c = f"over_{h}_hours"
        sel = cd[cd[c]] if len(cd) and c in cd.columns else pd.DataFrame()
        row[f"days_over_{h}_hours"] = len(sel)
        row[f"share_of_active_days_over_{h}_hours"] = _rate(len(sel), len(cd))
        row[f"paid_on_days_over_{h}_hours"] = round(float(sel.paid_that_day.sum()), 2) if len(sel) else 0.0
    DAY_SUMMARY.append(row)
day_summary_df = pd.DataFrame(DAY_SUMMARY)
_assert_unique(day_summary_df.columns, "day summary columns")
print(day_summary_df.T.to_string())

# A per diem is one day. More than one unit of the same per diem code for one member on one day
# is a finding on the face of the claim, and no peer group is needed to say so. H2038 is reported
# in its own test below because its basis is unverified and it is a focus code.
per_diem_units_day = (
    df_line
    .filter(F.col("is_per_diem_any"))
    .groupBy(SPROV_COL, MEMBER_COL, "srv_date", PROC_COL)
    .agg(F.round(F.sum("qty"), 2).alias("units_that_day"),
         F.count(F.lit(1)).alias("lines"),
         F.round(F.sum("paid"), 2).alias("paid"),
         F.concat_ws(", ", F.array_sort(F.collect_set(BPROV_COL))).alias("billing_entities"),
         F.concat_ws(", ", F.array_sort(F.collect_set("claim_id"))).alias("claim_ids"),
         F.max("span_days").alias("longest_span_days"))
    .filter(F.col("units_that_day") > F.lit(PER_DIEM_UNITS_PER_DAY))
)
per_diem_units_df = (per_diem_units_day
                     .withColumn("unit_basis", F.coalesce(_basis_map[F.col(PROC_COL)], F.lit("UNCLASSIFIED")))
                     .withColumn("basis_is_verified", ~F.col(PROC_COL).isin(sorted(UNVERIFIED_BASIS_CODES)))
                     .withColumn("is_focus_code", F.col(PROC_COL).isin(FOCUS_CODES))
                     .orderBy(F.desc("paid"))
                     .limit(DETAIL_ROWS).toPandas())

h2038_units_df = (per_diem_units_df[per_diem_units_df[PROC_COL] == "H2038"].copy()
                  if len(per_diem_units_df) else pd.DataFrame())
if len(h2038_units_df):
    h2038_units_df["what_to_check"] = (
        "H2038 is documented in FFS chapter 19 as a per diem with a five hour minimum, so more "
        "than one unit in a day does not fit that description. The basis is NOT verified against "
        "the fee schedule. Confirm the basis before treating this as an error.")
print(f"\nPer diem codes billed above {PER_DIEM_UNITS_PER_DAY:g} unit for one member in one day: "
      f"{len(per_diem_units_df)} member days")
print(f"   of which H2038: {len(h2038_units_df)} member days")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 13. Part E. The published rules, tested
# MAGIC
# MAGIC These five tests are anchored on published AHCCCS policy or on arithmetic. None of them needs a
# MAGIC peer group and none of them needs a statistical defence, which is why they come before the peer
# MAGIC comparison in this workbook. Each row names a member, a date, the codes, the units and the
# MAGIC dollars, so each row is an instruction to pull a record.
# MAGIC
# MAGIC 1. **Above 8 units in a day on the documentation list.** AHCCCS requires documentation above 8
# MAGIC    units a day on eleven codes, H0004 and T1016 among them, and above 4 units on H0034. The rule
# MAGIC    applies to **claims received** on or after 2023-05-03, so this test uses the date the claim
# MAGIC    arrived and not the date of service. Lines received before the cutoff are counted in a
# MAGIC    separate column that cannot flag anybody. This is a documentation requirement, not a cap: a
# MAGIC    day above 8 units is a day where documentation should exist, not a day that was not allowed.
# MAGIC 2. **H0004 on the same day as a residential or intensive outpatient per diem.** Verified
# MAGIC    prohibition. The test keys on the **billing entity**, because AHCCCS explicitly permits
# MAGIC    therapy billed by a separate provider on the same day as a residential per diem. To do that
# MAGIC    properly this test looks at every per diem billed under the same billing entity, including
# MAGIC    per diems billed by other servicing providers under that entity, not only the two targets.
# MAGIC 3. **T1016 on the same day as H0018.** Verified prohibition, keyed on the billing entity for
# MAGIC    the same reason: case management by a provider other than the residential facility is
# MAGIC    permitted.
# MAGIC 4. **More than one per diem behavioural health code for one member in one day.** Published rule,
# MAGIC    effective February 2024 and restated in the Covered Behavioral Health Services Guide.
# MAGIC    Collisions that involve a code with an unverified unit basis are reported in a column of
# MAGIC    their own and cannot flag, because nobody has published how many days one of those units
# MAGIC    covers.
# MAGIC 5. **A per diem code above one unit in a day**, and **H2038 above one unit in a day**, from
# MAGIC    Part D above.
# MAGIC
# MAGIC **One limitation stated plainly.** Tests 1, 4 and 5 can only see what these two providers and
# MAGIC their billing entities billed. A member who received a per diem from a completely unrelated
# MAGIC provider on the same day is invisible here. The member journey screen is the notebook that does
# MAGIC that arithmetic.

# COMMAND ----------

# Every per diem line, from any provider and any billing entity, for the members these two
# providers treated. That scope is what makes the same day tests honour the published permission
# for a SEPARATE provider to bill therapy on a residential day: the test needs to see the per
# diem whoever billed it, and then decide on the billing entity. Scoping it to the targets' own
# billing entities would make the permitted cross entity version invisible rather than counted,
# and an invisible permitted pattern reads as an absent one.
TARGET_ENTITIES = sorted({r[0] for r in df_line.select(BPROV_COL).distinct().collect()
                          if r[0] is not None})
print(f"Billing entities appearing on these providers' claims: {len(TARGET_ENTITIES)}")

_members_df = df_line.select(MEMBER_COL).distinct()
N_TARGET_MEMBERS = _members_df.count()
if N_TARGET_MEMBERS:
    _pd_raw = spark.table(SERVICE_TABLE).filter(
        F.upper(F.trim(F.coalesce(F.col(PROC_COL).cast("string"), F.lit(""))))
         .isin(sorted(PER_DIEM_ALL)))
    df_entity_per_diem = (prepare(_pd_raw)
                          .join(F.broadcast(_members_df), MEMBER_COL, "inner")
                          .cache())
    N_ENTITY_PER_DIEM = df_entity_per_diem.count()
else:
    df_entity_per_diem = df_line.filter(F.lit(False))
    N_ENTITY_PER_DIEM = 0
print(f"Members treated by these providers: {N_TARGET_MEMBERS:,}")
print(f"Per diem lines for those members, from any provider: {N_ENTITY_PER_DIEM:,}")

rule_src = df_line
entity_pd_src = df_entity_per_diem

# ── Rule 1. Above the documentation threshold, keyed on the CLAIM RECEIPT date ────────────────
#
# This rule applies to claims RECEIVED on or after 2023-05-03. If the table carries no receipt
# date the test is NOT RUN. There is deliberately no fallback to the service date: the claim lag
# would mis-date the test, and a mis-dated documentation test produces a false finding against a
# named individual. That is the worst outcome available in this notebook, and it is worse than
# having no answer.
RECEIPT_AVAILABLE = bool(RECEIPT_KEY_COL or RECEIPT_DT_COL)
DOC_TEST_STATUS = ("run, keyed on the claim receipt date" if RECEIPT_AVAILABLE else
                   "NOT RUN: this table carries no claim receipt date column and the published "
                   "rule applies to claims RECEIVED on or after " + DOC_RECEIPT_CUTOFF +
                   ". There is no fallback to the service date, because the claim lag would "
                   "mis-date the test and could produce a false finding against a named person.")
_doc_codes = {c: DOC_8_THRESHOLD for c in DOC_8_CODES}
_doc_codes.update({c: DOC_4_THRESHOLD for c in DOC_4_CODES})
_thr_map = F.create_map(*[F.lit(x) for kv in _doc_codes.items() for x in kv])

if RECEIPT_AVAILABLE:
    _doc = (rule_src.filter(F.col(PROC_COL).isin(sorted(_doc_codes)))
            .withColumn("doc_threshold_units", _thr_map[F.col(PROC_COL)])
            .withColumn("received_on_or_after_cutoff",
                        F.col("recv_date").isNotNull() &
                        (F.col("recv_date") >= F.lit(DOC_RECEIPT_CUTOFF)))
            .groupBy(SPROV_COL, MEMBER_COL, "srv_date", PROC_COL, "doc_threshold_units")
            .agg(F.round(F.sum("qty"), 2).alias("units_that_day"),
                 F.round(F.sum(F.when(F.col("received_on_or_after_cutoff"), F.col("qty")).otherwise(0.0)), 2)
                  .alias("units_on_lines_received_after_cutoff"),
                 F.round(F.sum(F.when(~F.col("received_on_or_after_cutoff"), F.col("qty")).otherwise(0.0)), 2)
                  .alias("units_on_lines_received_before_cutoff"),
                 F.count(F.lit(1)).alias("lines"),
                 F.round(F.sum("paid"), 2).alias("paid"),
                 F.min("recv_date").alias("earliest_claim_received"),
                 F.max("recv_date").alias("latest_claim_received"),
                 F.concat_ws(", ", F.array_sort(F.collect_set(BPROV_COL))).alias("billing_entities"),
                 F.concat_ws(", ", F.array_sort(F.collect_set(
                     F.when(F.col("mod_set") != F.lit(""), F.col("mod_set"))))).alias("modifier_sets"),
                 F.max(F.col("is_group")).alias("any_group_line"),
                 F.concat_ws(", ", F.array_sort(F.collect_set("diag1"))).alias("primary_diagnoses"),
                 F.max(F.col("is_focus_diag")).alias("any_focus_diagnosis"),
                 F.concat_ws(", ", F.array_sort(F.collect_set("claim_id"))).alias("claim_ids")))
    doc_over_df = (_doc
                   .withColumn("over_threshold_on_lines_received_after_cutoff",
                               F.col("units_on_lines_received_after_cutoff") > F.col("doc_threshold_units"))
                   .withColumn("over_threshold_counting_every_line",
                               F.col("units_that_day") > F.col("doc_threshold_units"))
                   .filter(F.col("over_threshold_on_lines_received_after_cutoff") |
                           F.col("over_threshold_counting_every_line"))
                   .withColumn("is_focus_code", F.col(PROC_COL).isin(FOCUS_CODES))
                   .withColumn("minutes_implied_if_not_group",
                               F.col("units_that_day") * F.coalesce(_minutes_map[F.col(PROC_COL)], F.lit(0)))
                   .withColumn("keyed_on", F.lit("claim receipt date, which is what the published "
                                                 "rule says"))
                   .withColumn("receipt_date_source", F.lit(RECEIPT_KEY_COL or RECEIPT_DT_COL))
                   .orderBy(F.desc("units_that_day"))
                   .limit(DETAIL_ROWS).toPandas())
    _doc_counted = int(doc_over_df["over_threshold_on_lines_received_after_cutoff"].sum()) \
        if len(doc_over_df) else 0
    _doc_excluded = len(doc_over_df) - _doc_counted
    print(f"\nRule 1. Member days above the documentation threshold: {_doc_counted} counted "
          f"(claims received on or after {DOC_RECEIPT_CUTOFF}), {_doc_excluded} listed but NOT "
          f"counted because the claim was received before the cutoff.")
else:
    doc_over_df = pd.DataFrame()
    _doc_counted = _doc_excluded = None
    print(f"\nRule 1. NOT RUN. {DOC_TEST_STATUS}")

# ── Rule 2. H0004 on the same day as a residential or IOP per diem, SAME billing entity ──────
_h0004 = (rule_src.filter(F.col(PROC_COL) == F.lit("H0004"))
          .groupBy(SPROV_COL, BPROV_COL, MEMBER_COL, "srv_date")
          .agg(F.round(F.sum("qty"), 2).alias("h0004_units"),
               F.round(F.sum("paid"), 2).alias("h0004_paid"),
               F.count(F.lit(1)).alias("h0004_lines"),
               F.concat_ws(", ", F.array_sort(F.collect_set("claim_id"))).alias("h0004_claim_ids"),
               F.concat_ws(", ", F.array_sort(F.collect_set("diag1"))).alias("h0004_primary_diagnoses")))
_resiop = (entity_pd_src.filter(F.col(PROC_COL).isin(sorted(RES_OR_IOP_PER_DIEM)))
           .groupBy(BPROV_COL, MEMBER_COL, "srv_date")
           .agg(F.concat_ws(", ", F.array_sort(F.collect_set(PROC_COL))).alias("per_diem_codes"),
                F.round(F.sum("qty"), 2).alias("per_diem_units"),
                F.round(F.sum("paid"), 2).alias("per_diem_paid"),
                F.concat_ws(", ", F.array_sort(F.collect_set(SPROV_COL))).alias("per_diem_servicing_providers"),
                F.concat_ws(", ", F.array_sort(F.collect_set("claim_id"))).alias("per_diem_claim_ids")))
h0004_perdiem_df = (_h0004.join(_resiop, [BPROV_COL, MEMBER_COL, "srv_date"], "inner")
                    .withColumn("same_billing_entity", F.lit(True))
                    .withColumn("rule", F.lit("H0004 must not be billed the same day as a "
                                              "residential or intensive outpatient per diem"))
                    .orderBy(F.desc("h0004_paid"))
                    .limit(DETAIL_ROWS).toPandas())
# The permitted version, counted so a reader can see how much of it there is. It cannot flag.
_resiop_other = (entity_pd_src.filter(F.col(PROC_COL).isin(sorted(RES_OR_IOP_PER_DIEM)))
                 .groupBy(MEMBER_COL, "srv_date")
                 .agg(F.concat_ws(", ", F.array_sort(F.collect_set(BPROV_COL))).alias("per_diem_entities")))
h0004_cross_entity_n = (_h0004.join(_resiop_other, [MEMBER_COL, "srv_date"], "inner")
                        .filter(~F.expr(f"array_contains(split(per_diem_entities, ', '), {BPROV_COL})"))
                        .count())
print(f"Rule 2. H0004 on a residential or IOP per diem day, SAME billing entity: "
      f"{len(h0004_perdiem_df)} member days (a finding). "
      f"Different entity, which AHCCCS PERMITS: {h0004_cross_entity_n} member days (not a finding).")

# ── Rule 3. T1016 on the same day as H0018, SAME billing entity ──────────────────────────────
_t1016 = (rule_src.filter(F.col(PROC_COL) == F.lit("T1016"))
          .groupBy(SPROV_COL, BPROV_COL, MEMBER_COL, "srv_date")
          .agg(F.round(F.sum("qty"), 2).alias("t1016_units"),
               F.round(F.sum("paid"), 2).alias("t1016_paid"),
               F.count(F.lit(1)).alias("t1016_lines"),
               F.concat_ws(", ", F.array_sort(F.collect_set("claim_id"))).alias("t1016_claim_ids"),
               F.concat_ws(", ", F.array_sort(F.collect_set("diag1"))).alias("t1016_primary_diagnoses")))
_h0018 = (entity_pd_src.filter(F.col(PROC_COL) == F.lit("H0018"))
          .groupBy(BPROV_COL, MEMBER_COL, "srv_date")
          .agg(F.round(F.sum("qty"), 2).alias("h0018_units"),
               F.round(F.sum("paid"), 2).alias("h0018_paid"),
               F.concat_ws(", ", F.array_sort(F.collect_set(SPROV_COL))).alias("h0018_servicing_providers"),
               F.concat_ws(", ", F.array_sort(F.collect_set("claim_id"))).alias("h0018_claim_ids")))
t1016_h0018_df = (_t1016.join(_h0018, [BPROV_COL, MEMBER_COL, "srv_date"], "inner")
                  .withColumn("same_billing_entity", F.lit(True))
                  .withColumn("rule", F.lit("T1016 must not be billed the same day as H0018"))
                  .orderBy(F.desc("t1016_paid"))
                  .limit(DETAIL_ROWS).toPandas())
_h0018_any = (entity_pd_src.filter(F.col(PROC_COL) == F.lit("H0018"))
              .groupBy(MEMBER_COL, "srv_date")
              .agg(F.concat_ws(", ", F.array_sort(F.collect_set(BPROV_COL))).alias("h0018_entities")))
t1016_cross_entity_n = (_t1016.join(_h0018_any, [MEMBER_COL, "srv_date"], "inner")
                        .filter(~F.expr(f"array_contains(split(h0018_entities, ', '), {BPROV_COL})"))
                        .count())
print(f"Rule 3. T1016 on an H0018 day, SAME billing entity: {len(t1016_h0018_df)} member days "
      f"(a finding). Different entity, which AHCCCS permits for case management: "
      f"{t1016_cross_entity_n} member days (not a finding).")

# ── Rule 4. More than one per diem behavioural health code for one member in one day ─────────
_pd_day = (df_day
           .filter(F.col("is_per_diem_any"))
           .groupBy(SPROV_COL, MEMBER_COL, "srv_date")
           .agg(F.countDistinct(PROC_COL).alias("distinct_per_diem_codes"),
                F.countDistinct(F.when(F.col("is_per_diem_testable"), F.col(PROC_COL)))
                 .alias("distinct_per_diem_codes_with_a_published_unit"),
                F.concat_ws(", ", F.array_sort(F.collect_set(PROC_COL))).alias("per_diem_codes"),
                F.concat_ws(", ", F.array_sort(F.collect_set(
                    F.when(F.col("is_unverified_basis"), F.col(PROC_COL))))).alias("codes_with_an_unverified_unit"),
                F.round(F.sum("day_qty"), 2).alias("per_diem_units"),
                F.round(F.sum("day_paid"), 2).alias("per_diem_paid"),
                F.round(F.max("day_paid"), 2).alias("largest_single_per_diem_paid"),
                F.concat_ws(", ", F.array_sort(F.collect_set(BPROV_COL))).alias("billing_entities"),
                F.countDistinct(BPROV_COL).alias("distinct_billing_entities"),
                F.concat_ws(", ", F.array_sort(F.collect_set("claim_id"))).alias("claim_ids"))
           .filter(F.col("distinct_per_diem_codes") >= F.lit(2)))
multi_per_diem_df = (_pd_day
                     .withColumn("can_be_tested_against_the_published_rule",
                                 F.col("distinct_per_diem_codes_with_a_published_unit") >= F.lit(2))
                     .withColumn("dollars_beyond_the_largest_per_diem",
                                 F.round(F.col("per_diem_paid") -
                                         F.col("largest_single_per_diem_paid"), 2))
                     .withColumn("rule", F.lit("only one per diem behavioural health service per "
                                               "member per day, effective 2024-02-06"))
                     .orderBy(F.desc("dollars_beyond_the_largest_per_diem"))
                     .limit(DETAIL_ROWS).toPandas())
_pd_testable = int(multi_per_diem_df["can_be_tested_against_the_published_rule"].sum()) if len(multi_per_diem_df) else 0
print(f"Rule 4. Member days with two or more per diem codes: {len(multi_per_diem_df)}, "
      f"of which {_pd_testable} involve only codes with a published unit basis and can be tested "
      f"against the rule. The remainder involve a code AHCCCS has never published a unit for and "
      f"are listed without a finding attached.")

RULE_COUNTS = {
    "Rule 1: above the documentation threshold, claims received on or after the cutoff":
        _doc_counted if _doc_counted is not None else "NOT RUN",
    "Rule 1 decoy: above the threshold but the claim was received BEFORE the cutoff":
        _doc_excluded if _doc_excluded is not None else "NOT RUN",
    "Rule 2: H0004 on a residential or IOP per diem day, same billing entity": len(h0004_perdiem_df),
    "Rule 2 permitted: the per diem was billed by a different entity": h0004_cross_entity_n,
    "Rule 3: T1016 on an H0018 day, same billing entity": len(t1016_h0018_df),
    "Rule 3 permitted: the H0018 was billed by a different entity": t1016_cross_entity_n,
    "Rule 4: two or more per diem codes for one member in one day": len(multi_per_diem_df),
    "Rule 4 testable against the published rule": _pd_testable,
    "Rule 5: a per diem code above one unit in one member day": len(per_diem_units_df),
    "Rule 5: H2038 above one unit in one member day": len(h2038_units_df),
}
_assert_unique(RULE_COUNTS, "rule count labels")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 14. Part F. Who they work with, and whether the two are connected
# MAGIC
# MAGIC These two providers were named together, so this section answers the obvious question directly.
# MAGIC
# MAGIC **Do they share members, and is the overlap bigger than chance?** The shared member count is
# MAGIC exact. The comparison against chance is a rough model and should be read as a rough model. It
# MAGIC assumes members are assigned to providers independently and at random from a shared pool, which
# MAGIC is not true: geography, health plan, language, specialism and referral habits all concentrate
# MAGIC members, and two providers in the same building would share members with no arrangement between
# MAGIC them at all. So a ratio above 1 is a reason to look, never evidence of an arrangement. The
# MAGIC member pool used as the denominator is stated on the sheet.
# MAGIC
# MAGIC **The other three questions have no model in them and are simply facts off the claims:** which
# MAGIC billing entities both providers bill under, which referring providers send to both, and whether
# MAGIC the shared members were seen on the same days or at different times.
# MAGIC
# MAGIC **Why the referring provider matters.** It is a relationship stated on the claim. Every other
# MAGIC way of inferring that two providers are connected is guesswork by comparison.
# MAGIC
# MAGIC Each provider also gets its own billing entity sheet and referring provider sheet, with dollars
# MAGIC and the payment suspension indicator for every entity that appears.

# COMMAND ----------

ap_line = df_line

# ── Member overlap ───────────────────────────────────────────────────────────────────────────
_mem = {pid: {r[0] for r in ap_line.filter(F.col(SPROV_COL) == pid)
              .select(MEMBER_COL).distinct().collect()} for pid in TARGETS}
overlap_rows = []
for i, a in enumerate(TARGETS):
    for b in TARGETS[i + 1:]:
        shared = _mem[a] & _mem[b]
        na, nb = len(_mem[a]), len(_mem[b])
        # The denominator for the chance comparison: every member seen by any provider of either
        # target's own provider type, which is the nearest thing to a plausible shared pool that
        # this table supports. Stated on the sheet because it drives the ratio.
        types = sorted({PROFILES[p].get("provider_type_code") for p in (a, b)
                        if PROFILES[p].get("provider_type_code")})
        pool = None
        if types and SPROV_TYPE_COL:
            pool = (spark.table(SERVICE_TABLE)
                    .filter(F.col(SPROV_TYPE_COL).cast("string").isin(types))
                    .select(MEMBER_COL).distinct().count())
        expected = (na * nb / pool) if (pool and pool > 0) else None
        overlap_rows.append({
            "provider_a": a, "provider_b": b,
            "members_seen_by_a": na, "members_seen_by_b": nb,
            "members_seen_by_both": len(shared),
            "share_of_a_members_also_seen_by_b": _rate(len(shared), na),
            "share_of_b_members_also_seen_by_a": _rate(len(shared), nb),
            "member_pool_used_as_the_denominator": pool,
            "pool_definition": ("every member seen by any servicing provider whose provider type "
                                f"code is one of {', '.join(types)}") if types else
                               "could not be built: no provider type column",
            "expected_overlap_if_members_were_assigned_at_random":
                round(expected, 2) if expected is not None else None,
            "observed_over_expected": _rate(len(shared), expected) if expected else None,
            "how_to_read_this": "The shared member count is exact. The expected figure assumes "
                                "members are assigned independently and at random from the pool, "
                                "which is NOT true in practice: two providers in one building, "
                                "one health plan or one small town share members with no "
                                "arrangement between them. A ratio above 1 is a reason to look, "
                                "not evidence of an arrangement.",
        })
member_overlap_df = pd.DataFrame(overlap_rows)
if len(member_overlap_df):
    print(member_overlap_df.T.to_string())

SHARED_MEMBERS = sorted(set.intersection(*[_mem[p] for p in TARGETS])) if len(TARGETS) > 1 else []
print(f"\nMembers seen by every one of the {len(TARGETS)} providers: {len(SHARED_MEMBERS)}")

# One row per shared member per date, so the investigator can see whether the two providers saw
# the same person on the same day or months apart.
if SHARED_MEMBERS:
    _sm = ap_line.filter(F.col(MEMBER_COL).isin(SHARED_MEMBERS))
    overlap_days_df = (_sm.groupBy(MEMBER_COL, "srv_date")
                       .agg(F.countDistinct(SPROV_COL).alias("targets_billing_that_day"),
                            F.concat_ws(", ", F.array_sort(F.collect_set(SPROV_COL))).alias("providers"),
                            F.concat_ws(", ", F.array_sort(F.collect_set(PROC_COL))).alias("codes"),
                            F.concat_ws(", ", F.array_sort(F.collect_set(BPROV_COL))).alias("billing_entities"),
                            F.concat_ws(", ", F.array_sort(F.collect_set("diag1"))).alias("primary_diagnoses"),
                            F.round(F.sum("qty"), 2).alias("units"),
                            F.round(F.sum("paid"), 2).alias("paid"),
                            F.max(F.col("is_focus_code")).alias("any_focus_code"),
                            F.max(F.col("is_focus_diag")).alias("any_focus_diagnosis"))
                       .orderBy(F.desc("targets_billing_that_day"), MEMBER_COL, "srv_date")
                       .limit(DETAIL_ROWS).toPandas())
    _same_day = (overlap_days_df.targets_billing_that_day >= 2).sum() if len(overlap_days_df) else 0
    _range = (_sm.groupBy(MEMBER_COL)
              .agg(F.min("srv_date").alias("first_date_either_provider"),
                   F.max("srv_date").alias("last_date_either_provider"),
                   F.countDistinct("srv_date").alias("service_days_either_provider"),
                   F.round(F.sum("paid"), 2).alias("paid_either_provider"))
              .orderBy(F.desc("paid_either_provider")).toPandas())
    overlap_members_df = _range
    print(f"Dates on which a shared member was billed by both providers on the SAME day: {_same_day}")
else:
    overlap_days_df = pd.DataFrame()
    overlap_members_df = pd.DataFrame()
    _same_day = 0

# ── Billing entities and referring providers, per provider and shared ────────────────────────
def entity_table(pid):
    d = df_line.filter(F.col(SPROV_COL) == pid)
    tot = d.agg(F.round(F.sum("paid"), 2).alias("p")).first()["p"] or 0.0
    t = (d.groupBy(BPROV_COL)
         .agg(F.concat_ws(" | ", F.array_sort(F.collect_set(
                  F.when(F.col("bprov_name") != F.lit(""), F.col("bprov_name"))))).alias("billing_provider_name"),
              F.concat_ws(" | ", F.array_sort(F.collect_set(
                  F.when(F.col("bprov_type_desc") != F.lit(""), F.col("bprov_type_desc"))))).alias("billing_provider_type"),
              F.concat_ws(", ", F.array_sort(F.collect_set(
                  F.when(F.col("bprov_susp") != F.lit(""), F.col("bprov_susp"))))).alias("payment_suspension_indicator_raw"),
              F.count(F.lit(1)).alias("lines"),
              F.round(F.sum("qty"), 2).alias("units"),
              F.round(F.sum("paid"), 2).alias("paid"),
              F.countDistinct(MEMBER_COL).alias("members"),
              F.countDistinct("srv_date").alias("service_days"),
              F.countDistinct(PROC_COL).alias("codes"),
              F.min("srv_date").alias("first_service_date"),
              F.max("srv_date").alias("last_service_date"),
              F.round(F.sum(F.when(F.col("is_focus_code"), F.col("paid")).otherwise(0.0)), 2).alias("paid_on_focus_codes"),
              F.concat_ws(", ", F.array_sort(F.collect_set(
                  F.when(F.col("is_focus_code"), F.col(PROC_COL))))).alias("focus_codes_billed"))
         .orderBy(F.desc("paid")).toPandas())
    if len(t):
        t.insert(0, "provider_id", pid)
        t["share_of_provider_paid"] = [_rate(a, tot) for a in t.paid]
    return t


def referrer_table(pid):
    d = df_line.filter(F.col(SPROV_COL) == pid)
    tot = d.agg(F.round(F.sum("paid"), 2).alias("p")).first()["p"] or 0.0
    t = (d.groupBy("rprov_id")
         .agg(F.concat_ws(" | ", F.array_sort(F.collect_set(
                  F.when(F.col("rprov_name") != F.lit(""), F.col("rprov_name"))))).alias("referring_provider_name"),
              F.concat_ws(", ", F.array_sort(F.collect_set(
                  F.when(F.col("rprov_susp") != F.lit(""), F.col("rprov_susp"))))).alias("payment_suspension_indicator_raw"),
              F.count(F.lit(1)).alias("lines"),
              F.round(F.sum("paid"), 2).alias("paid"),
              F.countDistinct(MEMBER_COL).alias("members"),
              F.countDistinct("srv_date").alias("service_days"),
              F.min("srv_date").alias("first_service_date"),
              F.max("srv_date").alias("last_service_date"),
              F.round(F.sum(F.when(F.col("is_focus_code"), F.col("paid")).otherwise(0.0)), 2).alias("paid_on_focus_codes"))
         .orderBy(F.desc("paid")).toPandas())
    if len(t):
        t.insert(0, "provider_id", pid)
        t["referring_provider_id"] = t["rprov_id"].fillna("(no referring provider on the line)")
        t = t.drop(columns=["rprov_id"])
        t["share_of_provider_paid"] = [_rate(a, tot) for a in t.paid]
    return t


ENTITY_SHEETS = {pid: entity_table(pid) for pid in TARGETS}
REFERRER_SHEETS = {pid: referrer_table(pid) for pid in TARGETS}

_ent_sets = {pid: set(ENTITY_SHEETS[pid][BPROV_COL]) if len(ENTITY_SHEETS[pid]) else set()
             for pid in TARGETS}
_ref_sets = {pid: set(REFERRER_SHEETS[pid]["referring_provider_id"]) - {"(no referring provider on the line)"}
             if len(REFERRER_SHEETS[pid]) else set() for pid in TARGETS}
SHARED_ENTITIES = sorted(set.intersection(*[_ent_sets[p] for p in TARGETS])) if len(TARGETS) > 1 else []
SHARED_REFERRERS = sorted(set.intersection(*[_ref_sets[p] for p in TARGETS])) if len(TARGETS) > 1 else []

shared_entities_df = (pd.concat([ENTITY_SHEETS[p][ENTITY_SHEETS[p][BPROV_COL].isin(SHARED_ENTITIES)]
                                 for p in TARGETS if len(ENTITY_SHEETS[p])], ignore_index=True)
                      if SHARED_ENTITIES else pd.DataFrame())
shared_referrers_df = (pd.concat(
    [REFERRER_SHEETS[p][REFERRER_SHEETS[p]["referring_provider_id"].isin(SHARED_REFERRERS)]
     for p in TARGETS if len(REFERRER_SHEETS[p])], ignore_index=True)
    if SHARED_REFERRERS else pd.DataFrame())

print(f"Billing entities both providers bill under: {len(SHARED_ENTITIES)}"
      + (f" ({', '.join(SHARED_ENTITIES[:10])})" if SHARED_ENTITIES else ""))
print(f"Referring providers who send to both: {len(SHARED_REFERRERS)}"
      + (f" ({', '.join(SHARED_REFERRERS[:10])})" if SHARED_REFERRERS else ""))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 15. Part G. Peer comparison, and why to distrust it
# MAGIC
# MAGIC Each provider is compared against other providers of the **same provider type**, on four
# MAGIC measures restricted to the focus codes: units per member day, dollars per member, what share of
# MAGIC their whole book sits on the focus codes, and what share of their members carry one of the focus
# MAGIC diagnoses. Small peers are excluded by the floors in the settings, because a peer with three
# MAGIC members and one service day produces an extreme ratio out of nothing.
# MAGIC
# MAGIC **Read the contamination column before reading the percentile.** Arizona's outpatient
# MAGIC behavioural health spend went from roughly $53 million in 2019 to $668 million in 2022 on a
# MAGIC scheme that saturated the market. When a large share of a peer group is doing the same thing,
# MAGIC the 75th percentile is a scheme inflated number and an outlier test lands inside the cluster
# MAGIC rather than outside it. The published literature says the same: cross provider comparison
# MAGIC favours detecting uncommon behaviour, and widespread behaviour goes undetected.
# MAGIC
# MAGIC So this notebook reports, for every measure, the share of peers sitting beyond the 75th
# MAGIC percentile plus three interquartile ranges. Above 10% the measure is marked contaminated and
# MAGIC **the percentile on that measure should not be used on its own**. The published rule tests in
# MAGIC Part E need no peer group at all, which is why they come first.

# COMMAND ----------

PEER_TYPES = sorted({PROFILES[p].get("provider_type_code") for p in TARGETS
                     if PROFILES[p].get("provider_type_code")})
PEER_TYPE_DESCS = sorted({PROFILES[p].get("provider_type_description") for p in TARGETS
                          if PROFILES[p].get("provider_type_description")})
PEER_KEY_COL = SPROV_TYPE_COL or SPROV_TYPE_DESC_COL
PEER_KEY_VALUES = PEER_TYPES if SPROV_TYPE_COL else PEER_TYPE_DESCS


def contamination(series):
    """Share of the population beyond p75 plus three interquartile ranges. Above the configured
    share the top is heavy enough that a percentile cut lands inside the cluster."""
    s = pd.to_numeric(pd.Series(series), errors="coerce").dropna()
    if len(s) < 20:
        return None, "too few peers to judge"
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    fence = q3 + 3 * (q3 - q1)
    share = float((s > fence).mean())
    return round(share, 4), ("CONTAMINATED: the top of this distribution is heavy enough that a "
                             "percentile cut would land inside the cluster. Do not let this "
                             "measure decide anything on its own."
                             if share > CONTAMINATION_SHARE else
                             "the distribution looks ordinary enough for a percentile to mean "
                             "something")


PEER_MEASURES = [
    ("focus_units_per_member_day", "Units of the focus codes per member day"),
    ("focus_paid_per_member", "Approved dollars on the focus codes per member"),
    ("focus_share_of_book", "Share of all paid dollars sitting on the focus codes"),
    ("focus_diagnosis_share_of_members", "Share of members carrying a focus primary diagnosis"),
]
_assert_unique([m for m, _ in PEER_MEASURES], "peer measures")

peer_rows, contam_rows, peer_pool_df = [], [], pd.DataFrame()
if not PEER_KEY_COL or not PEER_KEY_VALUES:
    print("No provider type column or no type value for the targets, so no peer group can be "
          "built. The peer sheets are a note instead of a table.")
else:
    _peer_raw = (spark.table(SERVICE_TABLE)
                 .withColumn("_sp", F.trim(F.coalesce(F.col(SPROV_COL).cast("string"), F.lit(""))))
                 .filter(F.col(PEER_KEY_COL).cast("string").isin([str(v) for v in PEER_KEY_VALUES])))
    df_peer = prepare(_peer_raw.withColumn(SPROV_COL, F.col("_sp"))).cache()
    _n_peer_lines = df_peer.count()
    print(f"Peer scope: providers whose {PEER_KEY_COL} is one of "
          f"{', '.join(str(v) for v in PEER_KEY_VALUES)}. {_n_peer_lines:,} claim lines.")

    peer_agg = (df_peer.groupBy(SPROV_COL, "sprov_type", "sprov_type_desc")
                .agg(F.round(F.sum("paid"), 2).alias("total_paid"),
                     F.countDistinct(MEMBER_COL).alias("total_members"),
                     F.countDistinct("srv_date").alias("total_service_days"),
                     F.count(F.lit(1)).alias("total_lines"),
                     F.round(F.sum(F.when(F.col("is_focus_code"), F.col("paid")).otherwise(0.0)), 2).alias("focus_paid"),
                     F.round(F.sum(F.when(F.col("is_focus_code"), F.col("qty")).otherwise(0.0)), 2).alias("focus_units"),
                     F.sum(F.when(F.col("is_focus_code"), F.lit(1)).otherwise(F.lit(0))).alias("focus_lines"),
                     F.countDistinct(F.when(F.col("is_focus_code"), F.col(MEMBER_COL))).alias("focus_members"),
                     F.countDistinct(F.when(F.col("is_focus_code"), F.concat_ws(
                         "|", F.col(MEMBER_COL).cast("string"), F.col("srv_date").cast("string"))))
                      .alias("focus_member_days"),
                     F.countDistinct(F.when(F.col("is_focus_diag"), F.col(MEMBER_COL)))
                      .alias("members_with_a_focus_diagnosis"),
                     F.sum(F.col("is_group").cast("int")).alias("group_lines"))
                .toPandas())
    peer_agg["focus_units_per_member_day"] = [
        _rate(a, b) for a, b in zip(peer_agg.focus_units, peer_agg.focus_member_days)]
    peer_agg["focus_paid_per_member"] = [
        _rate(a, b) for a, b in zip(peer_agg.focus_paid, peer_agg.focus_members)]
    peer_agg["focus_share_of_book"] = [
        _rate(a, b) for a, b in zip(peer_agg.focus_paid, peer_agg.total_paid)]
    peer_agg["focus_diagnosis_share_of_members"] = [
        _rate(a, b) for a, b in zip(peer_agg.members_with_a_focus_diagnosis, peer_agg.total_members)]
    peer_agg["meets_the_volume_floor"] = ((peer_agg.total_members >= PEER_MIN_MEMBERS) &
                                          (peer_agg.total_service_days >= PEER_MIN_DAYS))
    peer_pool_df = peer_agg.copy()

    for pid in TARGETS:
        own = peer_agg[peer_agg[SPROV_COL] == pid]
        if own.empty:
            peer_rows.append({"provider_id": pid, "measure": "(not in the peer scope)",
                              "note": "this provider did not appear in the peer scope, so no "
                                      "percentile could be computed"})
            continue
        own_type = own.iloc[0]["sprov_type"] if SPROV_TYPE_COL else own.iloc[0]["sprov_type_desc"]
        pool = peer_agg[peer_agg["sprov_type" if SPROV_TYPE_COL else "sprov_type_desc"] == own_type]
        pool = pool[pool[SPROV_COL] != pid]
        if EXCLUDE_OTHER_TARGET_FROM_PEERS:
            pool = pool[~pool[SPROV_COL].isin(TARGETS)]
        pool_all = pool.copy()
        pool = pool[pool.meets_the_volume_floor]
        for measure, label in PEER_MEASURES:
            tv = own.iloc[0][measure]
            vals = pd.to_numeric(pool[measure], errors="coerce").dropna()
            share, verdict = contamination(vals)
            pct = None
            if len(vals) >= PEER_MIN_COUNT and tv is not None and not pd.isna(tv):
                pct = round(float((vals < float(tv)).mean()) * 100, 1)
            peer_rows.append({
                "provider_id": pid, "measure": measure, "what_it_means": label,
                "provider_value": tv,
                "peer_group_type": str(own_type),
                "peers_in_scope_before_the_floor": len(pool_all),
                "peers_above_the_volume_floor": len(pool),
                "peer_median": round(float(vals.median()), 4) if len(vals) else None,
                "peer_p75": round(float(vals.quantile(0.75)), 4) if len(vals) else None,
                "peer_p90": round(float(vals.quantile(0.90)), 4) if len(vals) else None,
                "peer_p99": round(float(vals.quantile(0.99)), 4) if len(vals) else None,
                "provider_percentile_among_peers": pct,
                "ratio_to_peer_median": _rate(tv, vals.median()) if len(vals) else None,
                "share_of_peers_beyond_p75_plus_3_iqr": share,
                "contamination_verdict": verdict,
                "percentile_reported":
                    "yes" if pct is not None else
                    f"NO: fewer than {PEER_MIN_COUNT} peers above the volume floor, so a "
                    f"percentile would be noise",
                "safe_to_lean_on_this_percentile":
                    "no, this measure is contaminated" if (share is not None and share > CONTAMINATION_SHARE)
                    else ("no percentile was reported" if pct is None else "yes, with the usual care"),
            })
            contam_rows.append({
                "provider_id": pid, "measure": measure, "what_it_means": label,
                "peers_measured": len(vals),
                "share_of_peers_beyond_p75_plus_3_iqr": share,
                "threshold_that_marks_it_contaminated": CONTAMINATION_SHARE,
                "verdict": verdict,
            })

peer_comparison_df = pd.DataFrame(peer_rows)
peer_contamination_df = pd.DataFrame(contam_rows)
if len(peer_comparison_df):
    _show = [c for c in ["provider_id", "measure", "provider_value", "peer_median",
                         "provider_percentile_among_peers", "share_of_peers_beyond_p75_plus_3_iqr",
                         "peers_above_the_volume_floor"] if c in peer_comparison_df.columns]
    print(peer_comparison_df[_show].to_string(index=False))
CONTAMINATED_MEASURES = (
    peer_contamination_df.loc[
        pd.to_numeric(peer_contamination_df.get("share_of_peers_beyond_p75_plus_3_iqr"),
                      errors="coerce").fillna(0) > CONTAMINATION_SHARE, "measure"].unique().tolist()
    if len(peer_contamination_df) else [])
if CONTAMINATED_MEASURES:
    print(f"\nCONTAMINATED measures, do not let these percentiles decide anything on their own: "
          f"{', '.join(sorted(set(CONTAMINATED_MEASURES)))}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 17. What is new since the last run
# MAGIC
# MAGIC Compares the rule findings in Part E against every finding from earlier runs of this notebook
# MAGIC for these providers, so a reviewer coming back to it can see what has appeared since they last
# MAGIC looked. The first run marks everything as unknown, because there is nothing to compare against.

# COMMAND ----------

def _load_history():
    try:
        h = pd.read_csv(HISTORY_PATH, dtype=str)
        return h[h["run_tag"] != RUN_TAG]
    except Exception:
        return pd.DataFrame(columns=["run_tag", "finding_key"])


_hist = _load_history()


def finding_keys(df, test, prov_col, mem_col, date_col, code_col=None):
    if df is None or not len(df):
        return pd.Series(dtype=str)
    parts = [df[prov_col].astype(str), df[mem_col].astype(str), df[date_col].astype(str)]
    if code_col and code_col in df.columns:
        parts.append(df[code_col].astype(str))
    out = pd.Series([test] * len(df), index=df.index)
    for p in parts:
        out = out.str.cat(p, sep="|")
    return out


def mark_new(df, test, prov_col, mem_col, date_col, code_col=None):
    """Add new_this_run against the prior run's findings. Nothing here is ever shortened: a
    truncated key collides and a collision silently marks a real finding as already seen."""
    if df is None or not len(df):
        return df, pd.Series(dtype=str)
    keys = finding_keys(df, test, prov_col, mem_col, date_col, code_col)
    seen = set(_hist["finding_key"]) if len(_hist) else set()
    df = df.copy()
    df.insert(0, "new_this_run",
              "Unknown (no earlier run)" if _hist.empty
              else np.where(keys.isin(seen), "No", "Yes"))
    return df, keys


_all_keys = []
doc_over_df, _k = mark_new(doc_over_df, "doc_threshold", SPROV_COL, MEMBER_COL, "srv_date", PROC_COL)
_all_keys.append(_k)
h0004_perdiem_df, _k = mark_new(h0004_perdiem_df, "h0004_per_diem", SPROV_COL, MEMBER_COL, "srv_date")
_all_keys.append(_k)
t1016_h0018_df, _k = mark_new(t1016_h0018_df, "t1016_h0018", SPROV_COL, MEMBER_COL, "srv_date")
_all_keys.append(_k)
multi_per_diem_df, _k = mark_new(multi_per_diem_df, "multi_per_diem", SPROV_COL, MEMBER_COL, "srv_date")
_all_keys.append(_k)
per_diem_units_df, _k = mark_new(per_diem_units_df, "per_diem_units", SPROV_COL, MEMBER_COL, "srv_date", PROC_COL)
_all_keys.append(_k)
if len(h2038_units_df):
    h2038_units_df, _ = mark_new(h2038_units_df, "per_diem_units", SPROV_COL, MEMBER_COL, "srv_date", PROC_COL)

KEYS_THIS_RUN = pd.concat([k for k in _all_keys if len(k)], ignore_index=True) \
    if any(len(k) for k in _all_keys) else pd.Series(dtype=str)
NEW_FINDINGS = 0
for _d in (doc_over_df, h0004_perdiem_df, t1016_h0018_df, multi_per_diem_df, per_diem_units_df):
    if _d is not None and len(_d) and "new_this_run" in _d.columns:
        NEW_FINDINGS += int((_d["new_this_run"] == "Yes").sum())
try:
    pd.concat([_hist, pd.DataFrame({"run_tag": RUN_TAG, "finding_key": KEYS_THIS_RUN})],
              ignore_index=True).to_csv(HISTORY_PATH, index=False)
except Exception as e:
    print(f"Could not update the history file ({e}). The next run will treat everything as new.")
print(f"Rule findings new since the last run: {NEW_FINDINGS}"
      + (" (first run, so everything is marked unknown)" if _hist.empty else ""))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 18. The Criteria sheet, the Data_Availability sheet and the workbook
# MAGIC
# MAGIC Three sheets matter most to a reader who did not build this:
# MAGIC
# MAGIC - **Start_Here** explains every other sheet in plain language;
# MAGIC - **Criteria** records every threshold, every code list, every unit basis and every assumption,
# MAGIC   so a number can be checked without reading any code;
# MAGIC - **Data_Availability** lists every optional column, whether it was found, and exactly which
# MAGIC   analysis was skipped or weakened because it was absent. It also records how group lines and
# MAGIC   multi day spans were treated.
# MAGIC
# MAGIC No sheet is ever skipped for being empty. An empty test writes a single explanatory row
# MAGIC instead, so a missing sheet can never be mistaken for a test that found nothing.

# COMMAND ----------

criteria = pd.DataFrame([
    ["What this workbook is", "An individual investigation of two named servicing providers, with a "
                              "short list of procedure codes and primary diagnoses used as a lens."],
    ["Run date", str(TODAY)],
    ["Providers under review", "; ".join(f"{p} ({TARGET_NAMES_GIVEN.get(p, 'no name supplied')})"
                                         for p in TARGETS)],
    ["Source table", SERVICE_TABLE],
    ["Focus procedure codes", ", ".join(FOCUS_CODES)],
    ["Focus primary diagnoses", ", ".join(FOCUS_DIAGS) + ", matched on DIAG_CD_1 with any decimal "
                               "point removed and upper cased"],
    ["THE LENS IS NOT A FILTER",
     "Every claim line for these two providers is loaded, whatever the code and whatever the "
     "diagnosis. The focus codes and diagnoses are highlighted, counted separately and compared "
     "against the rest of the book. They are never used to restrict the data."],
    ["What narrowing to the lens would cost",
     "Setting focus_only to true restricts the claims to the focus codes and focus diagnoses. Four "
     "things break. One, the clinician day hours come out too low, because a person's day contains "
     "every timed service they billed and not four of them. Two, every share of book is then "
     "measured against a denominator that holds almost nothing but the lens itself, so it sits "
     "near 100% by construction and no longer tells you what fraction of the provider's work "
     "these codes are. Three, the same day conflict tests become impossible, because H0004 against a "
     "residential or IOP per diem needs the per diem lines, and T1016 against H0018 needs the "
     "H0018 lines. Four, you cannot tell whether the focus diagnoses are the whole caseload or a "
     "sliver of it. Use it only to read a narrow view of something you have already understood "
     "from the full one."],
    ["focus_only setting in this run", "ON, with the four costs above in force" if FOCUS_ONLY
                                       else "OFF, which is the default and the right setting"],
    ["What the source table holds",
     f"{SERVICE_TABLE} holds PAID claim lines only. There are no denied lines in it. So every "
     f"dollar figure in this workbook is money paid, no figure here is an approval or denial "
     f"rate, and nothing in this workbook can say anything about what was refused. A pattern that "
     f"was denied and never paid is invisible to this notebook by construction."],
    ["Columns are detected, never assumed",
     "Only the servicing provider, the billing provider, the member, the procedure code, the "
     "units, the dollars paid and the service dates are treated as certain. Every other column is "
     "looked for by name, and where it is absent the analysis that needed it is reported as NOT "
     "RUN on the Data_Availability sheet with the reason. Nothing becomes a quiet zero."],
    ["Unit basis, H0004", FOCUS_CODE_NOTE["H0004"]],
    ["Unit basis, T1016", FOCUS_CODE_NOTE["T1016"]],
    ["Unit basis, H0031", FOCUS_CODE_NOTE["H0031"]],
    ["Unit basis, H2038", FOCUS_CODE_NOTE["H2038"]],
    ["CONFIRM H2038 BEFORE QUOTING ANY H2038 NUMBER",
     "H2038 is treated here as a per diem, so more than one unit in a day is reported as a finding "
     "on the face of the claim. The basis is NOT verified: FFS chapter 19 describes it as a per "
     "diem with a five hour minimum, but the code is absent from the FY21 fee schedule and from "
     "the 2023 rate study's behavioural health outpatient code list. A rate notice effective "
     "2025-10-01 prices it at $488.88. At that price a unit basis error is expensive in both "
     "directions, so check the current fee schedule before using any H2038 figure from this "
     "workbook."],
    ["Codes left unclassified on purpose",
     "Codes whose unit basis AHCCCS has never published keep their dollars and are kept out of "
     "every minutes calculation: " + ", ".join(sorted(UNVERIFIED_BASIS_CODES)) + ". A day count or "
     "an hour count built on an unpublished unit is a guess dressed up as arithmetic."],
    ["Codes AHCCCS does not include in its behavioural health set",
     ", ".join(sorted(ABSENT_CODES)) + ". Each confirmed absent against four independent AHCCCS "
     "sources. If one appears on a provider's claims that is itself worth a look, so it is counted "
     "on its own sheet rather than quietly classified."],
    ["Per session codes are never multiplied",
     "One session is one unit whatever its length. H0031 is one of these and AHCCCS states no "
     "duration for it, so it contributes ZERO minutes to every day total in this workbook. "
     "Treating 90834 as an hour overstates by a third; treating any session code as 15 minutes "
     "understates by two to four times."],
    ["Group (HQ) treatment", HQ_TREATMENT_TEXT],
    ["The group convention this guards against",
     "AHCCCS counts group time per member for the whole length of the group. Eight members in a "
     "one hour group produce 32 units from one hour of clinician time, and group size runs to 15. "
     "So a provider running ordinary group programming produces summed units that look impossible. "
     "A screen that sums units into hours without partitioning group lines generates false "
     "positives against real group programmes."],
    ["Multi day span lines", SPAN_TREATMENT_TEXT +
     f" Spans longer than {MAX_SPAN_DAYS} days are treated as a data error, left on the begin date "
     f"and counted in their own column."],
    ["Clinician day hour thresholds reported", ", ".join(f"{h} hours" for h in HOUR_THRESHOLDS) +
     ". These are reported as counts of days with the dollars on them. None of them is an AHCCCS "
     "published maximum and none of them flags anybody. The 96 unit and 40 unit figures quoted in "
     "the state's forensic audit are the auditors' own arithmetic, not an AHCCCS limit, and are "
     "not used here."],
    ["What the clinician day total leaves out",
     "Only behavioural health codes in the dictionary have a unit basis, so timed services outside "
     "that set contribute no minutes. A day total here is a floor, not a ceiling. Group minutes "
     "set aside are reported in their own column."],
    ["Whether a day total means one person's day",
     "; ".join(f"{p}: {PROFILES[p].get('person_or_organisation', 'unknown')}, so "
               f"{PROFILES[p].get('clinician_day_arithmetic_meaningful', 'unknown')}"
               for p in TARGETS) +
     ". Decided from SProv_Type_Desc on the claims, not assumed from the name in the request."],
    ["Documentation rule, Part E test 1",
     f"AHCCCS requires documentation above {DOC_8_THRESHOLD} units a day on "
     f"{', '.join(DOC_8_CODES)}, and above {DOC_4_THRESHOLD} units on {', '.join(DOC_4_CODES)}, "
     f"for claims RECEIVED on or after {DOC_RECEIPT_CUTOFF}. This test uses the claim receipt date "
     f"and NOT the date of service, because the published wording is 'claims received on and "
     f"after'. Lines received before the cutoff are listed in their own column and cannot flag."],
    ["Documentation rule status in this run", DOC_TEST_STATUS],
    ["Why there is no fallback to the service date",
     "If the table carries no claim receipt date this test does not run at all. Keying it on the "
     "service date instead would mis-date it by the claim lag, and a mis-dated documentation test "
     "produces a false finding against a named individual. No answer is better than a wrong one "
     "here."],
    ["A documentation requirement is not a cap",
     "A day above 8 units is a day where documentation should exist. It is not a day that was not "
     "allowed and it is not an overpayment."],
    ["Same day rules, Part E tests 2 and 3",
     "H0004 must not be billed the same day as a residential or intensive outpatient per diem, and "
     "T1016 must not be billed the same day as H0018. Both tests key on the BILLING ENTITY, "
     "because AHCCCS explicitly permits therapy billed by a separate provider on the same day as a "
     "residential per diem, and permits case management by a provider other than the facility. "
     "Getting that wrong is the main false positive risk in this whole workbook. To test it "
     "properly the per diem side loads every per diem line for the members these two providers "
     "treated, from any provider and any billing entity, and then decides on the billing entity. "
     "Scoping the per diem side to the targets' own entities instead would make the permitted "
     "cross entity pattern invisible rather than counted."],
    ["One per diem per member per day, Part E test 4",
     "Published rule, effective 2024-02-06 and restated in the Covered Behavioral Health Services "
     "Guide dated 2024-10-01. Collisions involving a code with an unverified unit basis are listed "
     "but carry no finding, because nobody has published how many days one of those units covers."],
    ["What Part E cannot see",
     "It only sees what these two providers and their billing entities billed. A member who "
     "received a per diem from an unrelated provider on the same day is invisible here. The member "
     "journey screen is the notebook that does cross provider member day arithmetic."],
    ["Per diem above one unit in a day",
     f"A per diem is one day, so more than {PER_DIEM_UNITS_PER_DAY:g} unit of the same per diem "
     f"code for one member in one day does not fit the description of the code. Reported for every "
     f"per diem code, and separately for H2038."],
    ["Member overlap and chance, Part F",
     "The shared member count is exact. The expected figure assumes members are assigned to "
     "providers independently and at random from a shared pool, which is NOT true: geography, "
     "health plan, language, specialism and referral habits all concentrate members, and two "
     "providers in one building share members with no arrangement between them. A ratio above 1 "
     "is a reason to look, not evidence of an arrangement. The pool used is named on the sheet."],
    ["Peer group, Part G", (f"Providers whose {PEER_KEY_COL} matches the target's, excluding the "
                            f"target itself"
                            + (", and excluding the other target" if EXCLUDE_OTHER_TARGET_FROM_PEERS else "")
                            + f". Peer floors: at least {PEER_MIN_MEMBERS} members and "
                              f"{PEER_MIN_DAYS} service days. A percentile is reported only when "
                              f"at least {PEER_MIN_COUNT} peers clear the floor.")
                           if PEER_KEY_COL else "Could not be built: no provider type column."],
    ["Why the peer comparison comes last",
     "The peer benchmark in this market is contaminated. Arizona's outpatient behavioural health "
     "spend went from roughly $53 million in 2019 to $668 million in 2022 on a scheme that "
     "saturated the market, so the 75th percentile is a scheme inflated number and an outlier "
     "fence lands inside the scheme cluster. Every peer measure therefore carries the share of "
     f"peers beyond p75 plus three interquartile ranges; above {CONTAMINATION_SHARE:.0%} the "
     "measure is marked contaminated and must not decide anything on its own. The Part E tests "
     "need no peer group at all, which is why they come first."],
    ["Contaminated measures in this run",
     ", ".join(sorted(set(CONTAMINATED_MEASURES))) if CONTAMINATED_MEASURES
     else "none of the peer measures crossed the contamination threshold in this run"],
    ["Modifier placeholders",
     "This extract uses placeholder values in the modifier columns for 'no modifier on the line'. "
     "These values are treated as no modifier rather than as a real modifier: "
     + ", ".join(sorted(t for t in MOD_NULL_TOKENS if t)) +
     ". Treating them as real would invent a modifier mix that does not exist. Every modifier is "
     "cast to string before it is compared, because when all four are null in a partition they can "
     "arrive typed numeric and comparing them with 'HQ' then throws a cast error."],
    ["Dollars", "paid is PMT_AMT, billed is BILL_AMT and allowed is ALLOW_AMT. Paid over billed is "
                "reported because it is informative on its own, and is blank rather than zero when "
                "BILL_AMT is not on the table. None of these figures is an overpayment finding and "
                "none is a recovery estimate."],
    ["What this workbook does not establish",
     "It is based only on claims data. It does not establish fraud, intent, or whether a service "
     "was medically necessary. Everything here is something to look at."],
], columns=["Item", "What it is"])

_susp_uniform = {p: SUSP_VALUES.get(p) for p in TARGETS}
data_availability_notes = pd.DataFrame([
    ["Claim lines loaded for the two providers", f"{N_LINES:,}"
     + (f" (restricted from {N_LINES_ALL:,} by focus_only)" if FOCUS_ONLY else "")],
    ["Day level rows built from them", f"{N_DAY_ROWS:,}"],
    ["Service date range", f"{DATA_START} to {DATA_END}" if DATA_START else "no claims found"],
    ["Claim receipt date range", f"{RECV_START} to {RECV_END}" if RECV_START else "not available"],
    ["Columns this notebook asked for and did NOT find",
     ", ".join(MISSING_COLS) if MISSING_COLS else "none, every column asked for is present"],
    ["Analyses NOT RUN because a column was absent",
     " | ".join(NOT_RUN) if NOT_RUN else "none"],
    ["Analyses WEAKENED because a column was absent",
     " | ".join(WEAKENED) if WEAKENED else "none"],
    ["Modifier columns used", ", ".join(MODS) if MODS else "NONE, so group lines cannot be "
                                                           "identified and the day totals may "
                                                           "include group time"],
    ["Diagnosis columns used", ", ".join(DIAGS) if DIAGS else "NONE, so the diagnosis dimension "
                                                              "is unavailable"],
    ["What the source table holds",
     "Paid claim lines only. There are no denied lines, so no figure here is an approval rate and "
     "nothing in this workbook speaks to what was refused."],
    ["Group (HQ) treatment", HQ_TREATMENT_TEXT],
    ["Multi day span treatment", SPAN_TREATMENT_TEXT],
    ["Lines with a service span over one day", f"{_multi:,}"],
    ["Lines with a span longer than the limit", f"{_long:,} (treated as a data error)"],
    ["Payment suspension indicator values seen on the targets",
     "; ".join(f"{p}: {_susp_uniform[p]!r}" for p in TARGETS) +
     ". The raw value is reported with no interpretation. If every provider in the extract carries "
     "the same value, this column is not populated in a way that tells you anything."],
    ["Billing entities on the targets' claims", f"{len(TARGET_ENTITIES)}"],
    ["Members treated by the targets", f"{N_TARGET_MEMBERS:,}"],
    ["Per diem lines loaded for the same day tests",
     f"{N_ENTITY_PER_DIEM:,}: every per diem line for those members, from any provider and any "
     f"billing entity. Scoped by member rather than by the targets' own entities so that the "
     f"permitted cross entity pattern is counted rather than invisible."],
    ["Peer scope", f"{len(peer_pool_df):,} providers aggregated" if len(peer_pool_df)
                   else "no peer group could be built"],
    ["Targets absent from the table", ", ".join(_absent_target) if _absent_target else "none"],
], columns=["Item", "What it is"])

# One row per section of the workbook, so an investigator can see at a glance what this run
# could and could not do rather than finding a section quietly thin.
def _status(ok, weakened_if=False, reason=""):
    if not ok:
        return "NOT RUN", reason
    if weakened_if:
        return "WEAKENED", reason
    return "ran", ""


rule_status_rows = []
for _section, _ok, _weak, _why in [
    ("Part A, provider profile", True, not SPROV_TYPE_DESC_COL,
     "no provider type description, so whether this ID is a person or an organisation is unknown"),
    ("Part A, payment suspension indicator", bool(SPROV_SUSP_COL), False,
     "no SProv_CAF_Susp_Ind column"),
    ("Part B, focus codes", True, False, ""),
    ("Part B, focus diagnoses", bool(PRIMARY_DIAG_COL), False, "no primary diagnosis column"),
    ("Part B, code by diagnosis cross tabulation", bool(PRIMARY_DIAG_COL), False,
     "no primary diagnosis column"),
    ("Part B, modifier mix", bool(MODS), False, "no modifier columns"),
    ("Part C, every code ranked", True, False, ""),
    ("Part C, every diagnosis ranked", bool(PRIMARY_DIAG_COL), False,
     "no primary diagnosis column"),
    ("Part D, clinician day minutes", True, not MODS,
     "no modifier columns, so group (HQ) lines cannot be partitioned and the hours are OVERSTATED"),
    ("Part D, multi day span handling", True, not have(SRV_END_COL),
     "no service end date, so every line is treated as a single day line"),
    ("Part E rule 1, documentation threshold", RECEIPT_AVAILABLE, False,
     "no claim receipt date, and there is deliberately no fallback to the service date"),
    ("Part E rule 2, H0004 with a per diem", True, False, ""),
    ("Part E rule 3, T1016 with H0018", True, False, ""),
    ("Part E rule 4, one per diem per member day", True, False, ""),
    ("Part E rule 5, per diem units above one", True, False, ""),
    ("Part F, member overlap", True, not PEER_KEY_COL,
     "no provider type column, so the overlap has no chance comparison"),
    ("Part F, shared billing entities", True, False, ""),
    ("Part F, shared referring providers", have(RPROV_COL), False, "no referring provider column"),
    ("Part G, peer comparison", bool(PEER_KEY_COL and len(peer_comparison_df)), False,
     "no provider type column, or the targets were not in the peer scope"),
    ("Part G, focus diagnosis peer measure", bool(PRIMARY_DIAG_COL and PEER_KEY_COL), False,
     "no primary diagnosis column"),
    ("Paid over billed, everywhere", bool(BILL_COL), False, "no BILL_AMT column"),
    ("Member age at service", bool(DOB_COL), False, "no DATE_OF_BIRTH column"),
    ("Claim identifiers on the Part E sheets", bool(CLAIM_ID_COL or CLAIM_HDR_COL), False,
     "no claim identifier, so a row names a provider, a member, a date and a code but no claim"),
]:
    st, why = _status(_ok, _weak, _why)
    rule_status_rows.append({"section_of_the_workbook": _section, "status": st,
                             "reason_if_not_fully_available": why})
_assert_unique([r["section_of_the_workbook"] for r in rule_status_rows], "analysis status rows")
_bad = [r for r in rule_status_rows if r["status"] != "ran"]
print("\nAnalysis status by section:")
if _bad:
    for r in _bad:
        print(f"   {r['status']:<9} {r['section_of_the_workbook']}: {r['reason_if_not_fully_available']}")
else:
    print("   every section of the workbook ran with every column it wanted")

rule_summary = pd.DataFrame(
    [{"test": k, "rows": v} for k, v in RULE_COUNTS.items()])
rule_summary["rows_numeric"] = pd.to_numeric(rule_summary["rows"], errors="coerce")
rule_summary["note"] = [
    "the test could not run at all; see the Data_Availability sheet" if str(v) == "NOT RUN"
    else "a finding" if k.startswith(("Rule 1:", "Rule 2:", "Rule 3:", "Rule 4:", "Rule 5:"))
    else "a decoy or a permitted pattern, counted so you can see how much of it there is"
    for k, v in zip(rule_summary.test, rule_summary.rows)]
if (rule_summary["rows_numeric"].fillna(0) == 0).all():
    rule_summary.loc[len(rule_summary)] = [
        "EVERY TEST FOUND NOTHING OR COULD NOT RUN", 0, 0,
        "A test that silently finds nothing is the failure mode to fear. Before concluding there "
        "is nothing here, check the Data_Availability sheet for missing columns and for tests "
        "marked NOT RUN, check that the provider IDs are right, and check that focus_only is off."]
print(rule_summary.to_string(index=False))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 19. Write the workbook
# MAGIC
# MAGIC One file covering both providers. Start_Here first, then Criteria, Data_Availability and Run_Notes, then the
# MAGIC two provider profiles side by side, then the focus lens, then the whole book, then the day, then
# MAGIC the published rules, then the connection between the two providers, then the peer comparison.
# MAGIC Sheets whose rows belong to one provider carry that provider's ID at the front of the sheet
# MAGIC name.
# MAGIC
# MAGIC Sheet names are checked for collisions before anything is written. Excel caps a name at 31
# MAGIC characters and two names that shorten to the same thing would silently overwrite each other,
# MAGIC which has happened in this project before and destroyed two tests without any error.

# COMMAND ----------

_SHEET_REGISTRY = {}


def _sheet(name):
    s = str(name)[:31]
    if s in _SHEET_REGISTRY and _SHEET_REGISTRY[s] != name:
        raise ValueError(f"Excel sheet name collision: '{name}' and '{_SHEET_REGISTRY[s]}' both "
                         f"shorten to '{s}'. Rename one; do not truncate.")
    _SHEET_REGISTRY[s] = name
    return s


def _write(w, df, name, note):
    d = df if (df is not None and len(df)) else pd.DataFrame({"note": [note]})
    d.to_excel(w, sheet_name=_sheet(name), index=False)


def pfx(pid, name):
    """A sheet name for a sheet whose rows all belong to one provider."""
    return f"P{pid}_{name}"


start_here = pd.DataFrame([
    ["What this is",
     "An individual investigation of two named servicing providers: "
     + "; ".join(f"{p} ({TARGET_NAMES_GIVEN.get(p, 'no name supplied')})" for p in TARGETS)
     + ". Four procedure codes and four primary diagnoses are used as a lens."],
    ["Run date", str(TODAY)],
    ["Period covered", f"{DATA_START} to {DATA_END}" if DATA_START else "no claims found"],
    ["Where to start",
     "Open 'A_Provider_Profile'. It has one column for each provider so you can compare them at a "
     "glance. Read the person_or_organisation row first: it decides whether the hours in a day on "
     "the day sheets mean one person's day or a whole organisation's. Then open "
     "'E_Rule_Summary' for the published rule tests, because those need no statistical argument."],
    ["The most important thing about these numbers",
     "The four focus codes and the four focus diagnoses are a LENS, not a filter. Every claim for "
     "these two providers is in this workbook, whatever the code. That is on purpose. If the data "
     "were cut down to the four codes, the hours in a day would come out too low, every percentage "
     "would be 100% by construction, and the same day conflict tests could not run at all. The "
     "Criteria sheet spells this out."],
    ["Every dollar here is money paid",
     "The source table holds paid claim lines only. There are no denied lines in it, so nothing in "
     "this workbook tells you what was refused, and no percentage here is an approval rate."],
    ["Analysis_Status", "One row per section of this workbook saying whether it ran, ran "
                        "weakened, or did not run at all, and why. Read it with "
                        "Data_Availability."],
    ["Read Data_Availability before you read anything as absent",
     "Most of the columns this workbook uses are optional: the notebook looks for them and says "
     "plainly when one is not there. That sheet lists every optional column, whether it was found, "
     "and which analysis was skipped or weakened without it. If a section looks thin, check there "
     "first, because the difference between 'this provider did not do it' and 'this extract could "
     "not show it' matters."],
    ["A_Provider_Profile", "One column per provider: name, type, whether a payment suspension "
                           "indicator is set, date range, paid dollars, members, billing "
                           "entities, referring providers, lines of business, places of service."],
    ["B_Focus_Codes", "Each of the four focus codes for each provider: lines, units, dollars, "
                      "members, service days, units per member day, dollars per unit, and what "
                      "share of this provider's whole book the code represents. A code the "
                      "provider never billed appears as a row of zeros rather than being missing, "
                      "so you can tell 'checked, none found' from 'not checked'."],
    ["B_Focus_Diagnoses", "The same for the four focus primary diagnoses."],
    ["B_Code_By_Diagnosis", "The cross tabulation. Every focus code against every focus diagnosis, "
                            "with lines, members and dollars in each cell, plus two subtotal rows "
                            "per code: all four focus diagnoses together, and every other "
                            "diagnosis. Read the subtotals first."],
    ["B_Focus_Modifier_Mix", "Which modifiers appear on the focus code lines. HQ means a group "
                             "line. The placeholder values this extract uses for 'no modifier' "
                             "have been stripped out, so a blank really means no modifier."],
    ["The sheets starting with P and a provider number",
     "These belong to one provider only. All_Codes and All_Diagnoses are that provider's whole "
     "book ranked by dollars, with the focus items marked. That is how you learn whether the four "
     "codes are the provider's practice or a corner of it. Clinician_Days is one row per day they "
     "billed. Billing_Entities and Referring_Prov are who submits their claims and who refers to "
     "them, with the payment suspension indicator for each."],
    ["D_Day_Summary", "How many days went past 8, 12, 16 and 24 hours of billed timed service, and "
                      "the dollars paid on those days. None of those hour figures is an AHCCCS "
                      "published maximum. They are reported, not used to flag."],
    ["The one thing to understand about hours",
     "AHCCCS counts group time per member for the whole length of the group. Eight members in a "
     "one hour group produce 32 units from one hour of one clinician's time, and groups run to 15 "
     "members. So a provider running ordinary group programming can look impossible. By default "
     "this workbook takes group lines out of the hours arithmetic entirely and reports the minutes "
     "it set aside in its own column. That understates a day on purpose rather than risk naming "
     "somebody over a group programme billed exactly as the manual instructs."],
    ["The one thing to understand about H0031",
     "H0031 is billed per session and AHCCCS states no duration for it. It is counted as sessions "
     "and contributes ZERO minutes to every hour total here. A provider with a lot of H0031 will "
     "show a lot of sessions and few hours, and that is correct, not a gap."],
    ["The one thing to understand about H2038",
     "H2038 is treated here as a per diem, so more than one unit in a day is reported as something "
     "to look at. But the unit basis is NOT confirmed: the billing manual calls it a per diem with "
     "a five hour minimum while the code is absent from the fee schedule, and a rate notice prices "
     "it at $488.88. Check the current fee schedule before quoting any H2038 number from this "
     "workbook. At that price a unit error is expensive in both directions."],
    ["E_Rule_Summary", "How many rows each published rule test produced, with the decoys and the "
                       "permitted patterns counted alongside so you can see how much of each there "
                       "is. If every number here is zero, read the Data_Availability sheet before "
                       "concluding there is nothing to find."],
    ["E_Doc_Threshold_Over_8", "Member days above the published documentation threshold. This test "
                               "uses the date the CLAIM ARRIVED, not the date of service, because "
                               "the rule says 'claims received on and after 2023-05-03'. Days "
                               "whose claims arrived before that date are in the table with a "
                               "column saying so, and they are not findings. A documentation "
                               "requirement is not a cap: a day above 8 units is a day where "
                               "documentation should exist. If this table has no claim receipt "
                               "date the test does NOT run and the sheet says so: keying it on "
                               "the service date instead would mis-date it by the claim lag and "
                               "could produce a false finding against a named person."],
    ["Status of the documentation test in this run", DOC_TEST_STATUS],
    ["E_H0004_With_Per_Diem", "H0004 billed on the same day as a residential or intensive "
                              "outpatient per diem by the SAME billing entity. AHCCCS PERMITS "
                              "therapy billed by a separate provider on a residential day, so the "
                              "cross entity version is not a finding and is counted on the summary "
                              "sheet instead."],
    ["E_T1016_With_H0018", "T1016 billed on the same day as H0018 by the same billing entity. "
                           "Case management by a provider other than the facility is permitted, so "
                           "again only the same entity version is a finding."],
    ["E_Multi_Per_Diem_Days", "Member days with two or more per diem codes. AHCCCS allows one per "
                              "diem behavioural health service per member per day. Rows involving "
                              "a code whose unit basis has never been published are listed with a "
                              "column saying they cannot be tested against the rule."],
    ["E_Per_Diem_Units_Over_1", "A per diem code billed above one unit for one member in one day. "
                                "A per diem is one day, so this does not fit the description of "
                                "the code."],
    ["E_H2038_Units_Over_1", "The H2038 rows from that sheet on their own, because H2038 is one of "
                             "the focus codes and because its unit basis is unconfirmed."],
    ["F_Member_Overlap", "Whether the two providers share members, and how that compares with "
                         "chance. Read the how_to_read_this column. The shared count is exact; the "
                         "chance comparison is a rough model that assumes members are assigned at "
                         "random, which they are not. Two providers in one building share members "
                         "with no arrangement between them."],
    ["F_Overlap_Member_Days", "For each shared member and date, which of the two providers billed "
                              "and what they billed, so you can see whether they saw the same "
                              "person on the same day or months apart."],
    ["F_Shared_Billing_Entities", "Billing entities both providers bill under."],
    ["F_Shared_Referring_Prov", "Referring providers who send members to both. A referral is a "
                                "relationship stated on the claim, which is better evidence of a "
                                "connection than anything inferred."],
    ["G_Peer_Comparison", "Each provider against other providers of the same type, on the focus "
                          "codes. Read the contamination column BEFORE the percentile column."],
    ["Why to distrust a percentile here",
     "Arizona's outpatient behavioural health spend went from roughly $53 million in 2019 to $668 "
     "million in 2022 on a scheme that saturated the market. When much of a peer group is doing "
     "the same thing, the 75th percentile is a scheme inflated number and an outlier test lands "
     "inside the cluster. Every peer measure here carries the share of peers sitting beyond the "
     "75th percentile plus three interquartile ranges, and above 10% the measure is marked "
     "contaminated. The published rule tests on the E sheets need no peer group at all, which is "
     "why they come first."],
    ["new_this_run", "On the rule sheets, Yes means this member and date did not appear on an "
                     "earlier run of this notebook for these providers."],
    ["Before naming anybody",
     "Check the person_or_organisation row on A_Provider_Profile. Check the Criteria sheet to "
     "confirm the unit basis of the codes involved is still right, especially H2038. For the same "
     "day tests confirm that the billing entity really is the facility. And remember that these "
     "tests only see what these two providers and their billing entities billed."],
    ["What this is not",
     "It is based only on claims data. It does not establish fraud, intent, or whether a service "
     "was medically necessary. Everything here is something to look at."],
], columns=["Item", "Explanation"])

_EMPTY_WARN = ("no rows. If that is unexpected, read the Data_Availability sheet: a test that silently "
               "finds nothing is the failure mode to fear in this project.")

with pd.ExcelWriter(OUTPUT_PATH, engine="openpyxl") as w:
    _write(w, start_here, "Start_Here", "")
    _write(w, criteria, "Criteria", "")
    _write(w, data_availability, "Data_Availability", "")
    _write(w, data_availability_notes, "Run_Notes", "")
    _write(w, pd.DataFrame(rule_status_rows), "Analysis_Status", "")
    _write(w, profile_side_by_side, "A_Provider_Profile",
           "Neither provider has any claim line in this table. Check the IDs.")
    _write(w, focus_codes_df, "B_Focus_Codes", _EMPTY_WARN)
    _write(w, focus_diags_df, "B_Focus_Diagnoses", _EMPTY_WARN)
    _write(w, code_by_diag_df, "B_Code_By_Diagnosis", _EMPTY_WARN)
    _write(w, mod_mix_df, "B_Focus_Modifier_Mix",
           "No modifiers on any focus code line, or no focus code lines at all.")
    _write(w, day_summary_df, "D_Day_Summary", _EMPTY_WARN)
    _write(w, rule_summary, "E_Rule_Summary", "")
    _write(w, doc_over_df, "E_Doc_Threshold_Over_8",
           DOC_TEST_STATUS if not RECEIPT_AVAILABLE else
           f"No member day went above the documentation threshold on "
           f"{', '.join(DOC_8_CODES + DOC_4_CODES)}. {_EMPTY_WARN}")
    _write(w, h0004_perdiem_df, "E_H0004_With_Per_Diem",
           "No day where H0004 and a residential or intensive outpatient per diem were billed by "
           "the same billing entity for the same member.")
    _write(w, t1016_h0018_df, "E_T1016_With_H0018",
           "No day where T1016 and H0018 were billed by the same billing entity for the same "
           "member.")
    _write(w, multi_per_diem_df, "E_Multi_Per_Diem_Days",
           "No member day carried two or more per diem behavioural health codes.")
    _write(w, per_diem_units_df, "E_Per_Diem_Units_Over_1",
           f"No per diem code was billed above {PER_DIEM_UNITS_PER_DAY:g} unit for one member in "
           f"one day.")
    _write(w, h2038_units_df, "E_H2038_Units_Over_1",
           "H2038 was never billed above one unit for one member in one day, or was never billed "
           "at all. Check B_Focus_Codes to see which.")
    _write(w, member_overlap_df, "F_Member_Overlap",
           "Only one provider was analysed, so there is no overlap to report.")
    _write(w, overlap_members_df, "F_Overlap_Members",
           "The two providers share no members at all.")
    _write(w, overlap_days_df, "F_Overlap_Member_Days",
           "The two providers share no members, so there are no shared member days.")
    _write(w, shared_entities_df, "F_Shared_Billing_Entities",
           "The two providers have no billing entity in common.")
    _write(w, shared_referrers_df, "F_Shared_Referring_Prov",
           "No referring provider sends members to both, or the referring provider column is not "
           "populated. Check the Data_Availability sheet.")
    _write(w, peer_comparison_df, "G_Peer_Comparison",
           "No peer group could be built. Check the Data_Availability sheet for the provider type "
           "column.")
    _write(w, peer_contamination_df, "G_Peer_Contamination",
           "No peer group could be built, so no contamination diagnostic was computed.")
    _write(w, basis_cov, "Unit_Basis_Coverage", _EMPTY_WARN)
    _write(w, unverified_found, "Codes_Unverified_Basis",
           "Neither provider billed a code whose unit basis AHCCCS has never published.")
    _write(w, absent_found, "Codes_Not_In_AHCCCS_BH_Set",
           "Neither provider billed a code confirmed absent from the AHCCCS behavioural health "
           "set.")
    for pid in TARGETS:
        _write(w, ALL_CODES_SHEETS.get(pid), pfx(pid, "All_Codes"),
               f"Provider {pid} billed nothing in this table.")
        _write(w, ALL_DIAGS_SHEETS.get(pid), pfx(pid, "All_Diagnoses"),
               f"Provider {pid} has no diagnosis on any line, or billed nothing.")
        _write(w, CLIN_DAY_SHEETS.get(pid), pfx(pid, "Clinician_Days"),
               f"Provider {pid} has no service day in this table.")
        _write(w, ENTITY_SHEETS.get(pid), pfx(pid, "Billing_Entities"),
               f"Provider {pid} has no billing entity on any line.")
        _write(w, REFERRER_SHEETS.get(pid), pfx(pid, "Referring_Prov"),
               f"Provider {pid} has no referring provider on any line.")

print(f"\nSaved to {OUTPUT_PATH}")
print(f"  sheets written: {len(_SHEET_REGISTRY)}")
print(f"  providers covered: {', '.join(TARGETS)}")
print(f"  rule findings new since the last run: {NEW_FINDINGS}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 20. Email the workbook
# MAGIC
# MAGIC Sends the workbook to the recipients in the settings with a short summary in the body. If the
# MAGIC email fails the job still succeeds, because the workbook has already been saved.

# COMMAND ----------

import os, smtplib, ssl
from email.message import EmailMessage


def _secret(key):
    try:
        return dbutils.secrets.get(scope=SMTP_SCOPE, key=key)       # noqa: F821
    except Exception:
        return None


def _profile_rows():
    rows = ""
    for p in TARGETS:
        pr = PROFILES[p]
        rows += (f"<tr><td>{p}</td><td>{pr.get('name_on_claims', '')}</td>"
                 f"<td>{pr.get('provider_type_description', '')}</td>"
                 f"<td>{pr.get('person_or_organisation', '')}</td>"
                 f"<td align='right'>${(pr.get('paid') or 0):,.0f}</td>"
                 f"<td align='right'>{pr.get('distinct_members') or 0:,}</td>"
                 f"<td align='right'>"
                 f"{(pr.get('focus_code_share_of_paid') or 0) * 100:.0f}%</td></tr>")
    return ("<table border='1' cellpadding='4' cellspacing='0'>"
            "<tr><th>Provider</th><th>Name on claims</th><th>Type</th><th>Person or "
            "organisation</th><th>Approved paid</th><th>Members</th><th>Share of dollars on the "
            "focus codes</th></tr>" + rows + "</table>")


def _rule_rows():
    def _fmt(v):
        return f"{v:,}" if isinstance(v, (int, float)) else str(v)
    return ("<ul>" + "".join(f"<li>{k}: <b>{_fmt(v)}</b></li>" for k, v in RULE_COUNTS.items())
            + "</ul>")


body = f"""
<p>Individual deep dive on two servicing providers, run {RUN_TAG}.</p>
{_profile_rows()}
<p>The four focus codes ({', '.join(FOCUS_CODES)}) and the four focus diagnoses
({', '.join(FOCUS_DIAGS)}) are used as a lens. Every claim for both providers is in the workbook,
not only those codes, so that the hours in a day are right, every share has a denominator and the
same day conflict tests can run at all.</p>
<p>Published rule tests, which need no peer comparison:</p>
{_rule_rows()}
<ul>
  <li>Members seen by both providers: <b>{len(SHARED_MEMBERS)}</b>. Days when both billed the same
      member: <b>{_same_day}</b>. Billing entities in common: <b>{len(SHARED_ENTITIES)}</b>.
      Referring providers in common: <b>{len(SHARED_REFERRERS)}</b>.</li>
  <li>{'Peer measures marked contaminated: <b>' + ', '.join(sorted(set(CONTAMINATED_MEASURES))) + '</b>. Do not let those percentiles decide anything on their own.' if CONTAMINATED_MEASURES else 'No peer measure crossed the contamination threshold this run.'}</li>
</ul>
<p>Open the <b>Start_Here</b> sheet first; it explains every other sheet in plain language.</p>
<p style="color:#666">Three cautions worth repeating. H0031 is billed per session and AHCCCS states
no duration for it, so it adds no hours to any day total and that is correct. H2038 is treated as a
per diem but its unit basis is not confirmed against the fee schedule, so confirm it before quoting
any H2038 number. And AHCCCS permits therapy billed by a separate provider on the same day as a
residential per diem, so the same day tests only report the same billing entity.</p>
<p style="color:#666">These are leads for review based only on claims data. They do not establish
fraud, intent, or whether a service was medically necessary.</p>
<p style="color:#666">File: {OUTPUT_PATH}</p>
"""


def send_workbook():
    if not EMAIL_TO:
        return "skipped: no recipients configured"
    user, pwd = _secret(SMTP_USER_KEY), _secret(SMTP_PASS_KEY)
    msg = EmailMessage()
    _n_rows = sum(v for k, v in RULE_COUNTS.items()
                  if k.startswith("Rule") and ":" in k and isinstance(v, (int, float)))
    msg["Subject"] = (f"Provider deep dive - {', '.join(TARGETS)} - {RUN_TAG} - "
                      f"{_n_rows} rule test rows")
    msg["From"], msg["To"] = EMAIL_FROM, ", ".join(EMAIL_TO)
    msg.set_content("This message is formatted in HTML.")
    msg.add_alternative(body, subtype="html")
    size_mb = os.path.getsize(OUTPUT_PATH) / 1e6
    if size_mb <= MAX_ATTACH_MB:
        with open(OUTPUT_PATH, "rb") as fh:
            msg.add_attachment(fh.read(), maintype="application",
                               subtype="vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                               filename=os.path.basename(OUTPUT_PATH))
    else:
        msg.add_alternative(body + f"<p><b>The workbook is {size_mb:.0f} MB, too large to attach. "
                                   f"Open it at {OUTPUT_PATH}.</b></p>", subtype="html")
    with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=60) as s:
        s.starttls(context=ssl.create_default_context())
        if user and pwd:
            s.login(user, pwd)
        s.send_message(msg)
    return f"sent to {', '.join(EMAIL_TO)} ({size_mb:.1f} MB)"


try:
    print("Email:", send_workbook())
except Exception as e:
    print(f"Email NOT sent ({type(e).__name__}: {e}). The workbook is saved at {OUTPUT_PATH}.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 21. How to point this at other providers, and how to schedule it
# MAGIC
# MAGIC 1. Click **Schedule** at the top right of this notebook, or just run it with different
# MAGIC    parameters.
# MAGIC 2. Under Parameters set `targets` to the servicing provider IDs you want, comma separated. Set
# MAGIC    `focus_codes` and `focus_diags` to the codes and diagnoses you want highlighted. Set
# MAGIC    `email_to`.
# MAGIC 3. If you add a provider ID, add its name to `TARGET_NAMES_GIVEN` in section 1 as well, so the
# MAGIC    workbook can tell you whether the name on the claims matches the name you were given. That
# MAGIC    is the only code edit this notebook needs.
# MAGIC
# MAGIC **Four things to do before relying on the output.**
# MAGIC
# MAGIC 1. Read the **Data_Availability** and **Analysis_Status** sheets. Anything listed as a
# MAGIC    missing column means a part of this workbook could not be computed, and those sheets say
# MAGIC    exactly which part, whether it was skipped outright or only weakened, and why.
# MAGIC 2. Read the **person_or_organisation** row on A_Provider_Profile. If a provider turns out to be
# MAGIC    an organisational NPI rather than a person, every hour figure on the day sheets is that
# MAGIC    organisation's whole staff and must not be read as one person's day.
# MAGIC 3. Confirm the **H2038** unit basis against the current fee schedule. The notebook treats it as
# MAGIC    a per diem on the strength of the billing manual, but the code is absent from the fee
# MAGIC    schedule and a rate notice prices it at $488.88, so a unit error costs real money.
# MAGIC 4. Read **E_Rule_Summary** twice. Once for a test producing more rows than anybody can work,
# MAGIC    and once for a test producing none at all. A zero is not a pass; in this project a zero has
# MAGIC    twice meant a broken test rather than a clean provider.
# MAGIC
# MAGIC **One thing worth doing by hand, once.** Take three rows from E_Doc_Threshold_Over_8 and three
# MAGIC from E_H0004_With_Per_Diem and pull the actual records. That is the only way to find out whether
# MAGIC the unit basis and the billing entity logic behave the way this notebook assumes on real
# MAGIC Arizona claims, and it costs an afternoon.
