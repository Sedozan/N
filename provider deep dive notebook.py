# Databricks notebook source
# MAGIC %md
# MAGIC # Servicing provider deep dive
# MAGIC
# MAGIC **What this notebook does.** It looks at one or more named servicing providers one at a time
# MAGIC and describes everything they billed. A short list of procedure codes, and optionally a short
# MAGIC list of primary diagnoses, are used as a lens: they are highlighted everywhere, counted
# MAGIC separately, and compared against the rest of the provider's work. It writes one Excel workbook
# MAGIC covering every provider named and emails it to the review team.
# MAGIC
# MAGIC **This notebook changes shape to match the focus codes, and that is the first thing to
# MAGIC understand about it.** A procedure code is either billed by the clock or billed by the day, and
# MAGIC the two need completely different arithmetic.
# MAGIC
# MAGIC - If most of the money on the focus codes sits on **timed** codes, the spine of the analysis is
# MAGIC   the working day: minutes billed, hours in a day, days past 8, 12, 16 and 24 hours.
# MAGIC - If most of it sits on **per diem** codes, that spine says nothing at all. A per diem carries
# MAGIC   no minutes, so the hours come out near zero no matter how much is being billed, and a reader
# MAGIC   who sees near zero hours concludes wrongly that nothing is happening. The spine is instead
# MAGIC   the **daily census** (how many people were on a per diem each day) and the **episode** (how
# MAGIC   long each person stayed).
# MAGIC
# MAGIC The notebook works out which shape it is looking at, says so in plain language, leads with the
# MAGIC right section, and marks the other one as not the right instrument with the reason. Both paths
# MAGIC live in this one notebook; nothing is forked.
# MAGIC
# MAGIC **The second thing to understand is who the provider is.** A servicing provider ID on this
# MAGIC table may be an organisation rather than a person. An organisation's day is not one person's
# MAGIC day: several members of staff bill under one number, so a day total for an organisation can
# MAGIC never be compared against twenty four hours. The notebook decides this from the provider type
# MAGIC description on the claims and refuses to describe an organisation's day total as one person's
# MAGIC day anywhere in the workbook.
# MAGIC
# MAGIC **The most important thing to understand about the numbers.** The focus codes and the focus
# MAGIC diagnoses are a lens, not a filter. Every claim for these providers is loaded, whatever the
# MAGIC code and whatever the diagnosis. That is deliberate, and there are four reasons for it:
# MAGIC
# MAGIC 1. A working day contains every timed service billed that day, not just the focus ones. Filter
# MAGIC    the data to the focus codes and the hours in a day come out too low.
# MAGIC 2. Every rate needs a denominator. Without the whole book you cannot say what share of this
# MAGIC    provider's work the focus codes are, which is usually the question being asked.
# MAGIC 3. The same day conflict rules compare one code against another. H0004 must not be billed on
# MAGIC    the same day as a residential or intensive outpatient per diem, a member cannot be in
# MAGIC    residential care and attending an intensive outpatient programme on the same day, and you
# MAGIC    cannot see either of those unless the other lines are loaded too.
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
# MAGIC Everything a reviewer might want to change lives here: the providers, the focus codes, the
# MAGIC focus diagnoses, the thresholds and the email recipients. Each setting is printed onto the
# MAGIC Criteria sheet of the workbook so a result can be checked without reading any code.
# MAGIC
# MAGIC **The focus diagnosis list may be left empty.** When no diagnoses are supplied, every
# MAGIC diagnosis sheet and every diagnosis measure reports that no list was given rather than
# MAGIC producing an empty table that reads as "checked, none found". The rest of the workbook is
# MAGIC unaffected.
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
from datetime import date, timedelta

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
# The source table is a setting rather than a constant so the same notebook can be pointed at a
# different catalog or at a test table without a code edit.
SERVICE_TABLE = _param("service_table", "main.prod_input.all_paid_services", "Source table")

# The providers under review. IDs and names are kept as a pair so the workbook can show the
# name the investigator used and the name the claims carry, and flag a mismatch. A provider whose
# name was not supplied with the request is handled explicitly: the workbook says there was
# nothing to check rather than reporting a match against an empty string.
TARGETS = _list("targets", "158633", "Servicing provider IDs (comma separated)")
TARGET_NAMES_GIVEN = {
    "123531": "Knudson, Rachel Elizabeth",
    "120020": "Etukudoh Akaninyene J",
}

# The lens. Codes and primary diagnoses that are highlighted everywhere but never used to
# restrict the data. The diagnosis list may be empty, and when it is, every diagnosis sheet says
# no list was supplied instead of showing an empty table.
FOCUS_CODES = [c.strip().upper() for c in
               _list("focus_codes",
                     "S9480,S9485,H0035,H2012,H0018,H0019,98960,H0001,H0002,H0031",
                     "Focus procedure codes")]
FOCUS_DIAGS = [d.strip().upper().replace(".", "") for d in
               _list("focus_diags", "", "Focus primary diagnoses (may be left empty)")]
HAVE_FOCUS_DIAGS = bool(FOCUS_DIAGS)

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

OUTPUT_DIR = _param(
    "output_dir", f"/Workspace/Users/{ROOT}/Projects/provider_statistics/peer_analysis/results",
    "Folder the workbook is written to")
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
print(f"Focus primary diagnoses: "
      + (", ".join(FOCUS_DIAGS) if HAVE_FOCUS_DIAGS else
         "NONE SUPPLIED. Every diagnosis sheet will say so rather than show an empty table, and "
         "the diagnosis measure is dropped from the peer comparison."))
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
# MAGIC - **A session is one unit no matter how long it lasted.** H0031 is one of these. It
# MAGIC   is one of these. AHCCCS does not state a duration for it. It is counted as sessions and is
# MAGIC   never multiplied by a nominal length, so it contributes zero minutes to any day total.
# MAGIC - **The unit changes between tiers of the same service**, so the basis is held per code and
# MAGIC   never per service family.
# MAGIC - **Where AHCCCS has not published a unit, we do not guess.** Those codes keep their dollars
# MAGIC   and stay out of every minutes and days calculation.
# MAGIC
# MAGIC **The codes that come up most often in these requests, and what each one is.** The notebook
# MAGIC prints the basis of whichever codes were actually asked for, and this table is the reference.
# MAGIC
# MAGIC | Code | Service | One unit means | Note |
# MAGIC |---|---|---|---|
# MAGIC | `H0004` | counselling and therapy | 15 minutes | on the published above 8 units a day documentation list |
# MAGIC | `T1016` | case management | 15 minutes | on the same documentation list; must not be billed the same day as H0018 |
# MAGIC | `H0031` | mental health assessment by a non physician | one session | AHCCCS does not state a unit. Never multiplied by a duration |
# MAGIC | `H2038` | rehabilitation programme, per diem | one day, with a five hour minimum | basis NOT verified. See the warning below |
# MAGIC | `S9480` | intensive outpatient psychiatric | one day | per diem. Minimum 3 hours a day and 2 days a week. Once a day. Prior authorization from 2025-07-01 for fee for service |
# MAGIC | `S9485` | crisis intervention, per diem | one day | per diem. The hourly tier below it is S9484 |
# MAGIC | `H0035` | partial hospitalisation | one day | per diem. Minimum 20 hours a week |
# MAGIC | `H2012` | behavioural health day treatment | **60 minutes, up to 5 hours** | **NOT a per diem.** Above 5 hours the service belongs on H2015, and above that on H2016 |
# MAGIC | `H0018` | short term residential, BHRF | one day | per diem, **services only**: room and board is excluded. Place of service 56 from 2023-12-01. Provider type B8 bills it |
# MAGIC | `H0019` | long term residential | **unknown** | per diem asserted from the national HCPCS descriptor only. No AHCCCS statement, not repriced since 2014, absent from the billing manual. Dollars only |
# MAGIC | `98960` | education and training for patient self management | 30 minutes in the national descriptor | hard AHCCCS limits of 4 units a day and 60 units in any 30 days from 2023-04-01. Both are unit counts, so both tests are exact |
# MAGIC | `H0001` | alcohol and drug assessment | one session, any length | 12 a year. Which year is not stated, so both the calendar year and the benefit year are reported |
# MAGIC | `H0002` | screening for eligibility for admission to a treatment programme | one session, any length | 24 a year for members aged 11 and over, on the same unstated year basis |
# MAGIC | `H0031` | mental health assessment by a non physician | one session, any length | AHCCCS states no duration, so it is never multiplied and adds zero minutes to any day |
# MAGIC
# MAGIC **The two halves of a focus set like this one.** The first six codes are levels of care: where
# MAGIC a member is being treated and how intensively. The last four are the front door: who is
# MAGIC assessed, who is screened as eligible for admission, and who is given education and training.
# MAGIC They answer different questions, so the focus lens reports them as two blocks rather than one
# MAGIC flat list, and Part M describes the order in which a provider billed them.
# MAGIC
# MAGIC **H2012 is the trap in this set.** Four of the codes above it in the table are per diems and
# MAGIC H2012 sits among them, but it is billed **per hour**. Treating it as a per diem would turn five
# MAGIC perfectly ordinary hours of day treatment into five days of care. It is held as a 60 minute
# MAGIC timed code everywhere in this notebook.
# MAGIC
# MAGIC **H0019 keeps its dollars and nothing else.** Nobody at AHCCCS has published what one H0019
# MAGIC unit covers. It is counted in every dollar figure, named on every sheet it appears on, and kept
# MAGIC out of every day count, every census figure and every length of stay. A census built partly on
# MAGIC an unpublished unit would be a guess wearing the clothes of arithmetic.
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
TIMED_30_CODES = {                       # one paid unit = 30 minutes
    "98960",   # education and training for patient self management, individual   FOCUS CODE
    "98961",   # the same, 2 to 4 patients
    "98962",   # the same, 5 to 8 patients
}
# Where the 30 minutes comes from, and what that costs. The "each 30 minutes" is the national
# HCPCS descriptor of the code. AHCCCS has published a UNIT limit for this family, 4 units a day
# and 60 units in 30 days from 2023-04-01, but it has not published a minutes a unit statement.
# The two published limits are unit counts, so the tests built on them need no duration at all
# and are exact. The minutes are used only to put the code into the day hours total, and every
# day sheet carries a column showing how many of that day's minutes came from a code whose
# duration is the national descriptor rather than an AHCCCS statement.
DURATION_FROM_NATIONAL_DESCRIPTOR_ONLY = set(TIMED_30_CODES)
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

BH_UNIVERSE = sorted(TIMED_15_CODES | TIMED_60_CODES | TIMED_30_CODES | PER_DIEM_CODES |
                     PER_SESSION_CODES | PER_DOSE_CODES | PER_MONTH_CODES | ABSENT_CODES |
                     UNVERIFIED_BASIS_CODES)


def _basis_of(code):
    if code in UNVERIFIED_BASIS_CODES:
        return "UNVERIFIED"
    if code in ABSENT_CODES:
        return "NOT_IN_AHCCCS_BH_SET"
    if code in TIMED_15_CODES:
        return "TIMED_15"
    if code in TIMED_30_CODES:
        return "TIMED_30"
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
                        30 if CODE_BASIS[c] == "TIMED_30" else
                        60 if CODE_BASIS[c] == "TIMED_60" else 0) for c in BH_UNIVERSE}

# Level of care, used by the conflict matrix in Part E. A member cannot be in twenty four hour
# residential care and attending a day programme or an intensive outpatient programme on the same
# day, and the matrix that says so is printed on the Criteria sheet so every finding can be
# checked against the rule that produced it.
RESIDENTIAL_CODES = {"H0018", "H0019"}
DAY_OR_PARTIAL_CODES = {"H0035", "H2012"}
INTENSIVE_OUTPATIENT_CODES = {"H0015", "S9480"}
CPT_PSYCHOTHERAPY_CODES = {"90832", "90834", "90837", "90845", "90846", "90847", "90849", "90853"}
CASE_MANAGEMENT_CODES = {"T1016"}
COUNSELLING_CODES = {"H0004"}

LEVEL_OF_CARE_SETS = {
    "residential": RESIDENTIAL_CODES,
    "day or partial hospitalisation": DAY_OR_PARTIAL_CODES,
    "intensive outpatient": INTENSIVE_OUTPATIENT_CODES,
}


def _level_of_care(code):
    for lvl, codes in LEVEL_OF_CARE_SETS.items():
        if code in codes:
            return lvl
    return "not a level of care code"


CODE_LEVEL_OF_CARE = {c: _level_of_care(c) for c in BH_UNIVERSE}

# ── The front door and the treatment it leads to ─────────────────────────────────────────────
#
# Two different kinds of service sit in these focus sets and they answer different questions.
# Assessment, screening for eligibility for admission, and education and training are the FRONT
# DOOR: they decide who gets admitted. The per diems and day treatment are the TREATMENT people
# are admitted to. A provider that bills both halves is doing both jobs, and the funnel section
# of this workbook describes the order in which it billed them. That is a description of billing
# order and nothing more.
FRONT_DOOR_CODES = {"H0001", "H0002", "H0031"}
EDUCATION_CODES = set(TIMED_30_CODES)
CRISIS_PER_DIEM_CODES = {"S9485"}
TREATMENT_LEVEL_CODES = (RESIDENTIAL_CODES | DAY_OR_PARTIAL_CODES | INTENSIVE_OUTPATIENT_CODES |
                         CRISIS_PER_DIEM_CODES | {"H2016"})


def _focus_group_of(code):
    """Which block of the focus lens a code belongs in. The lens is grouped rather than listed
    flat because a flat list mixing levels of care with assessments is hard to read and invites a
    reader to compare two numbers that answer different questions."""
    if code in TREATMENT_LEVEL_CODES:
        return "treatment and level of care"
    if code in FRONT_DOOR_CODES or code in EDUCATION_CODES:
        return "assessment, screening and education"
    return "other"


CODE_FOCUS_GROUP = {c: _focus_group_of(c) for c in BH_UNIVERSE}
FOCUS_GROUP_ORDER = ["treatment and level of care", "assessment, screening and education", "other"]

# The focus codes, with their basis spelled out for the Criteria sheet. The notes cover every code
# this family of requests has asked for so far, and a code with no note still gets its basis from
# the dictionary above.
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
    "S9480": "Intensive outpatient psychiatric. One unit is ONE DAY. Published rules attached to "
             "it: once a day; a minimum of 3 hours a day and at least 2 days a week; prior "
             "authorization for dates of service on and after 2025-07-01 for fee for service "
             "(AIHP, DDD-THP, TRBHA, ALTCS) with IHS and 638 facilities exempt, and from "
             "2024-10-15 for Arizona Complete Health managed care. Paid at 58.66% of covered "
             "billed charges under By Report pricing until a fixed $370.41 took effect 2022-10-01.",
    "S9485": "Crisis intervention, PER DIEM. One unit is one day. The hourly tier below it is "
             "S9484, which is 60 minutes a unit up to 5 hours. The unit basis changes between the "
             "tiers, so the basis is held per code and never per service family.",
    "H0035": "Partial hospitalisation. One unit is ONE DAY. Published minimum of 20 hours a week. "
             "Claims cannot show hours for a per diem, so the weekly minimum is tested as a "
             "question about how many days in a week the programme was billed, never as an "
             "assertion that the minimum was missed.",
    "H2012": "Behavioural health day treatment. One unit is SIXTY MINUTES, up to 5 hours a day. "
             "THIS IS NOT A PER DIEM, even though it sits in the middle of a per diem focus set. "
             "Above 5 hours the service belongs on H2015 (15 minutes a unit, above 5 hours to 10 "
             "hours) and above that on H2016 (per diem), so more than 5 units of H2012 in one "
             "member day means the service exceeded the tier it was billed under. H2012 is also "
             "on the 2023-07-17 expanded documentation list in the 'above 2 hourly units' group, "
             "which keys on the claim RECEIPT date.",
    "H0018": "Behavioural health residential facility, SHORT TERM, per diem. One unit is one day, "
             "and it is a SERVICES ONLY per diem: room and board is excluded, so it must not be "
             "compared against a residential cost of care figure that includes them. Counselling "
             "and therapy, medication education, relapse prevention, substance use treatment, "
             "nursing, behavioural health professional oversight, case management by the facility "
             "and counselling by a behavioural health technician are all bundled into it and are "
             "not separately billable by the facility. Place of service 56 is required from "
             "2023-12-01. Provider type B8 is the provider type that bills it. Because the code "
             "says short term, a long unbroken run of it is a question on the face of the code.",
    "98960": "Education and training for patient self management, individual, by a qualified non "
             "physician professional. A community health worker and community health "
             "representative code. One unit is THIRTY MINUTES in the national HCPCS descriptor; "
             "AHCCCS has not published a minutes a unit statement, so that duration is used for "
             "the day hours total and is reported in a column of its own there. The two AHCCCS "
             "limits on it are UNIT counts and need no duration at all, so the tests built on "
             "them are exact: 4 units a day and 60 units in any 30 days, both effective "
             "2023-04-01 and both keyed on the date of SERVICE.",
    "H0001": "Alcohol and drug assessment. PER SESSION: one session is one unit whatever its "
             "length, and it is never multiplied by a duration. Limited to 12 a year by the "
             "Covered Behavioral Health Services Guide effective 2024-10-01. Which year is NOT "
             "stated, so this notebook counts both the calendar year and the 1 October to 30 "
             "September benefit year and reports them side by side.",
    "H0002": "Behavioural health screening to determine eligibility for admission to a treatment "
             "programme. PER SESSION. Limited to 24 a year for members aged 11 and over by the "
             "same guide, on the same unstated year basis, so both years are reported. Because "
             "this code decides who is admitted, the funnel section reports how often the same "
             "entity billed the screening and then the treatment it screened for. That is a "
             "description of billing order and nothing more.",
    "H0019": "Long term residential. UNIT BASIS NOT VERIFIED. Per diem is asserted from the "
             "national HCPCS descriptor only: there is no AHCCCS statement, the code has not been "
             "repriced since 2014 and it is absent from the billing manual, which is itself worth "
             "a look. It keeps its dollars everywhere in this workbook and is kept OUT of every "
             "day count, every census figure and every length of stay, because a day count built "
             "on an unpublished unit is a guess dressed up as arithmetic.",
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

# ── The 2023-07-17 expanded documentation list ───────────────────────────────────────────────
#
# A SECOND documentation control, separate from the 2023-05-03 one above and with its own code
# list, its own thresholds and its own cutoff. It keys on the claim RECEIPT date for the same
# reason: the published wording is "claims received on and after". The threshold is not one
# number, because the list mixes hourly codes with quarter hour codes and a code whose unit
# AHCCCS has never published:
#   any units at all          T1503, S5131, T2020, T2026, S5130, S9484
#   above 2 HOURLY units      H2012
#   above 4 QUARTER HOUR units  H0036, H2010, T1002, T1003
#   above 8 units             H2019, H2025
#   not testable              H0006, whose unit basis is unverified, so neither the two hour nor
#                             the four quarter hour threshold can be applied to it
DOC_2023_07_17_CUTOFF = _param("doc_2023_07_17_cutoff", "2023-07-17",
                               "Claim RECEIPT date the expanded documentation list starts")
DOC_EXPANDED_THRESHOLDS = {
    "T1503": 0, "S5131": 0, "T2020": 0, "T2026": 0, "S5130": 0, "S9484": 0,
    "H2012": 2,
    "H0036": 4, "H2010": 4, "T1002": 4, "T1003": 4,
    "H2019": 8, "H2025": 8,
}
DOC_EXPANDED_NOT_TESTABLE = {"H0006"}

# ── H2012 and the day treatment tier family ──────────────────────────────────────────────────
# FFS chapter 19 sets the family as H2012 per hour up to 5 hours, then H2015 per 15 minutes above
# 5 hours to 10 hours, then H2016 per diem. More than 5 units of H2012 in one member day means
# the service exceeded the tier it was billed under. Published, clean and needing no judgement.
H2012_TIER_MAX_UNITS = float(_param("h2012_tier_max_units", "5",
                                    "Units of H2012 the per hour tier covers in a day"))
DAY_TREATMENT_TIERS = [("H2012", "per hour, up to 5 hours"),
                       ("H2015", "per 15 minutes, above 5 hours to 10 hours"),
                       ("H2016", "per diem, above 10 hours")]

# ── S9480 ────────────────────────────────────────────────────────────────────────────────────
S9480_UNITS_PER_DAY = float(_param("s9480_units_per_day", "1",
                                   "Units of S9480 allowed in one member day"))
# Prior authorization. Two dates, because fee for service and managed care moved separately.
S9480_PA_FFS_DATE = _param("s9480_pa_ffs_date", "2025-07-01",
                           "Date S9480 needs prior authorization, fee for service")
S9480_PA_MCO_DATE = _param("s9480_pa_mco_date", "2024-10-15",
                           "Date S9480 needs prior authorization, Arizona Complete Health")
FFS_LOB_TOKENS = [t.strip().upper() for t in
                  _list("ffs_lob_tokens", "AIHP,DDD-THP,DDD THP,TRBHA,ALTCS,FFS,FEE FOR SERVICE",
                        "Line of business values that mean fee for service") if t.strip()]
MCO_PA_LOB_TOKENS = [t.strip().upper() for t in
                     _list("mco_pa_lob_tokens",
                           "ARIZONA COMPLETE HEALTH,AZ COMPLETE HEALTH,AZCH,COMPLETE HEALTH",
                           "Line of business values the 2024-10-15 managed care date applies to")
                     if t.strip()]
# ── Minimum intensity, and the line between a published rule and our own arithmetic ──────────
#
# S9480 has a published minimum of at least 2 DAYS a week. A claim can count days, so that rule
# can be tested as published and it is a rule.
#
# H0035 has a published minimum of 20 HOURS a week. A per diem claim carries no hours at all, so
# that one CANNOT be tested as published. An earlier version of this notebook turned the 20 hours
# into a number of days by dividing by the longest day of programming it was willing to believe.
# That divisor was ours, not AHCCCS's, and the threshold it produced was therefore ours too. It
# has been removed. A row built on our own divisor would sit on the rule sheets beside nine
# published rules, and the note saying it was a judgement would not travel with the row when
# somebody pasted it into a referral.
#
# What replaced it: the DISTRIBUTION of H0035 days per member per week, on its own sheet, with no
# threshold, no flag and no finding count, and the published 20 hours printed beside it as
# context for reading it.
S9480_MIN_DAYS_PER_WEEK = int(_param("s9480_min_days_per_week", "2",
                                     "S9480 published minimum days a week"))
H0035_MIN_HOURS_PER_WEEK = float(_param("h0035_min_hours_per_week", "20",
                                        "H0035 published minimum hours a week, reported as "
                                        "context only: a per diem claim cannot carry hours"))

# Calendar edges on the S9480 weekly minimum. A programme that starts on a Friday shows one day
# in its first week and that is where the calendar falls, not a short programme. Those weeks are
# EXCLUDED by default rather than merely labelled, because a label does not travel with a row
# that has been pasted into a referral. They are kept on a sheet of their own so nothing is
# hidden, and the count removed is reported.
S9480_INCLUDE_EDGE_WEEKS = _flag(
    "s9480_include_edge_weeks", "false",
    "Include the weeks at the edge of a run of S9480 activity in the weekly minimum test")

# ── 98960 and the education and training family ──────────────────────────────────────────────
#
# These are HARD CAPS, not documentation thresholds, and they key on the date of SERVICE rather
# than the claim receipt date. Both are unit counts, so neither needs a minutes a unit statement
# and both tests are exact whatever anyone thinks the 30 minutes means.
EDU_DAILY_UNIT_CAP = float(_param("edu_daily_unit_cap", "4",
                                  "Units of 98960 allowed in one member day"))
EDU_ROLLING_UNIT_CAP = float(_param("edu_rolling_unit_cap", "60",
                                    "Units of 98960 allowed in any rolling window"))
EDU_ROLLING_WINDOW_DAYS = int(_param("edu_rolling_window_days", "30",
                                     "Length of the 98960 rolling window, in days"))
EDU_LIMIT_DATE = _param("edu_limit_date", "2023-04-01",
                        "Date the 98960 unit limits took effect")

# ── Per year limits whose YEAR BASIS AHCCCS HAS NOT STATED ───────────────────────────────────
#
# The Covered Behavioral Health Services Guide limits H0001 to 12 a year and H0002 to 24 a year
# for members aged 11 and over. It does not say WHICH year. The shared specification records a
# benefit year of 1 October to 30 September for respite hours, so that is the likely basis, but it
# is not stated for these codes and this notebook does not guess. Both the calendar year and the
# benefit year are counted and reported side by side, and a member who is above the limit on one
# basis and below it on the other is marked AMBIGUOUS rather than reported as a finding.
PER_YEAR_LIMITS = {"H0001": 12.0, "H0002": 24.0}
H0002_MIN_AGE = int(_param("h0002_min_age", "11",
                           "Youngest age the H0002 yearly limit applies to"))
BENEFIT_YEAR_START_MONTH = int(_param("benefit_year_start_month", "10",
                                      "Month the benefit year starts, used for the per year "
                                      "limits whose basis AHCCCS has not stated"))

# ── The funnel ───────────────────────────────────────────────────────────────────────────────
# How long after the last front door service the first treatment service appears before the two
# stop being read as connected at all.
FUNNEL_MAX_LAG_DAYS = int(_param("funnel_max_lag_days", "180",
                                 "Days after a front door service within which a first "
                                 "treatment service is still reported as following it"))

# ── H0018 place of service ───────────────────────────────────────────────────────────────────
POS_56_DATE = _param("pos_56_date", "2023-12-01",
                     "Date behavioural health residential facilities must bill place of service 56")
BHRF_POS_CODE = _param("bhrf_pos_code", "56", "Place of service a BHRF must bill")
BHRF_PROVIDER_TYPE = "B8"

# ── Bed hold and therapeutic leave, which need revenue codes ─────────────────────────────────
REV_HOME_PASS = _param("rev_home_pass", "0183", "Revenue code for therapeutic leave, home pass")
REV_BED_HOLD = _param("rev_bed_hold", "0189", "Revenue code for bed hold")
LEAVE_LIMIT_21_PLUS = int(_param("leave_limit_21_plus", "9",
                                 "Therapeutic leave days a contract year, members 21 and over"))
BED_HOLD_LIMIT_21_PLUS = int(_param("bed_hold_limit_21_plus", "12",
                                    "Bed hold days a contract year, members 21 and over"))
COMBINED_LIMIT_UNDER_21 = int(_param("combined_limit_under_21", "21",
                                     "Combined leave and bed hold days a contract year, under 21"))
CONTRACT_YEAR_START_MONTH = int(_param("contract_year_start_month", "10",
                                       "Month the AHCCCS contract year starts"))

# ── Episodes, length of stay and readmission ─────────────────────────────────────────────────
# The gap convention matches the member journey screen exactly: a new stay starts when the gap
# between one billed day and the next is MORE than this many days, so 1 means the days have to be
# consecutive.
STAY_GAP_DAYS = int(_param("stay_gap_days", "1",
                           "Gap in per diem days that still counts as one stay"))
READMIT_WINDOW_DAYS = int(_param("readmit_window_days", "30",
                                 "Days after a stay ends within which a return counts as a "
                                 "readmission"))
READMIT_MIN_RETURNS = int(_param("readmit_min_returns", "2",
                                 "Readmissions inside the window before a member is listed as "
                                 "returning repeatedly"))
# The long stay cut is DESCRIPTIVE. It is a percentile of this provider's own observed stay
# lengths, not a published limit, and the Criteria sheet says so in those words.
LONG_STAY_TAIL_PCT = float(_param("long_stay_tail_pct", "0.90",
                                  "Percentile of the provider's own stay lengths that marks the "
                                  "tail (descriptive, not a published limit)"))
MIN_STAYS_FOR_A_TAIL = int(_param("min_stays_for_a_tail", "10",
                                  "Stays needed before a tail percentile is reported at all"))

# ── Monthly dollars reference ────────────────────────────────────────────────────────────────
# NOT an AHCCCS control. It is the figure the state's forensic audit recommended as a review
# trigger for Integrated Clinics in recommendation 12, and it is reported as a comparison only.
MONTHLY_PAID_REFERENCE = float(_param("monthly_paid_reference", "500000",
                                      "Auditor recommended monthly paid review figure for "
                                      "Integrated Clinics (NOT an AHCCCS control)"))

# ── The By Report pricing era, used to date the pricing change in the client's own data ──────
BY_REPORT_RATIO = float(_param("by_report_ratio", "0.5866",
                               "Share of covered billed charges paid under By Report pricing"))
BY_REPORT_TOLERANCE = float(_param("by_report_tolerance", "0.005",
                                   "How close paid over billed has to be to count as clustering"))
BY_REPORT_CLUSTER_SHARE = float(_param("by_report_cluster_share", "0.5",
                                       "Share of lines inside the tolerance that counts as "
                                       "clustering in a month"))
S9480_FIXED_RATE = float(_param("s9480_fixed_rate", "370.41",
                                "Fixed S9480 rate that replaced By Report pricing"))
S9480_FIXED_RATE_DATE = _param("s9480_fixed_rate_date", "2022-10-01",
                               "Date the fixed S9480 rate took effect")

# ── The organisation scope, for a servicing provider that is an organisation ─────────────────
ORG_SCOPE_MAX_LINES = int(_param("org_scope_max_lines", "5000000",
                                 "Largest number of lines the organisation scope will load "
                                 "before it reports itself as not run"))

# The shape of the focus set decides which section leads. auto works it out from the dollars.
FOCUS_SHAPE_SETTING = _param("focus_shape", "auto",
                             "Shape of the focus set: auto / per_diem / timed").strip().lower()
if FOCUS_SHAPE_SETTING not in ("auto", "per_diem", "timed"):
    raise ValueError(f"focus_shape must be auto, per_diem or timed, not {FOCUS_SHAPE_SETTING!r}")

print(f"Clinician day hour thresholds reported: {', '.join(str(h) for h in HOUR_THRESHOLDS)}")
print(f"Documentation rule: above {DOC_8_THRESHOLD} units a day on "
      f"{', '.join(DOC_8_CODES)}, and above {DOC_4_THRESHOLD} on {', '.join(DOC_4_CODES)}, "
      f"for claims RECEIVED on or after {DOC_RECEIPT_CUTOFF}")
print(f"Span treatment: {SPAN_TREATMENT}. {SPAN_TREATMENT_TEXT}")
print(f"Peer floors: at least {PEER_MIN_MEMBERS} members and {PEER_MIN_DAYS} service days, "
      f"and at least {PEER_MIN_COUNT} peers before a percentile is reported")
print(f"Expanded documentation list, claims RECEIVED on or after {DOC_2023_07_17_CUTOFF}: "
      f"{', '.join(f'{c} above {t}' if t else f'{c} any units' for c, t in sorted(DOC_EXPANDED_THRESHOLDS.items()))}"
      f". Not testable on that list: {', '.join(sorted(DOC_EXPANDED_NOT_TESTABLE))} (unit basis "
      f"unverified)")
print(f"H2012 per hour tier covers up to {H2012_TIER_MAX_UNITS:g} units a day; above that the "
      f"service belongs on H2015 and then H2016")
print(f"S9480 prior authorization: fee for service from {S9480_PA_FFS_DATE}, Arizona Complete "
      f"Health from {S9480_PA_MCO_DATE}, IHS and 638 exempt")
print(f"Minimum intensity: S9480 at least {S9480_MIN_DAYS_PER_WEEK} days a week, which a claim "
      f"can count, so it is tested as a rule and reported as a QUESTION for the service record. "
      f"Weeks at the edge of a run of activity are "
      f"{'INCLUDED' if S9480_INCLUDE_EDGE_WEEKS else 'EXCLUDED'} by default.")
print(f"H0035 {H0035_MIN_HOURS_PER_WEEK:g} hours a week is published, but a per diem claim "
      f"carries NO HOURS, so it cannot be tested. There is no H0035 intensity rule and no H0035 "
      f"intensity finding count. The distribution of H0035 days per member per week is reported "
      f"instead, with no threshold on it.")
print(f"Stays: a gap of more than {STAY_GAP_DAYS} day(s) starts a new stay. A return within "
      f"{READMIT_WINDOW_DAYS} days is a readmission, and {READMIT_MIN_RETURNS} or more of them is "
      f"'repeatedly'. The long stay cut is the {LONG_STAY_TAIL_PCT:.0%} point of this provider's "
      f"OWN stay lengths and is descriptive, not a published limit.")
print(f"Monthly paid comparison figure: ${MONTHLY_PAID_REFERENCE:,.0f}. This is the state forensic "
      f"audit's RECOMMENDED review figure for Integrated Clinics, not a live AHCCCS control.")
print(f"98960 hard caps from {EDU_LIMIT_DATE}: {EDU_DAILY_UNIT_CAP:g} units a day and "
      f"{EDU_ROLLING_UNIT_CAP:g} units in any rolling {EDU_ROLLING_WINDOW_DAYS} days. Both are "
      f"unit counts keyed on the date of SERVICE, so neither needs a minutes a unit statement.")
print(f"Per year limits with an UNSTATED year basis: "
      f"{', '.join(f'{c} {int(v)} a year' for c, v in sorted(PER_YEAR_LIMITS.items()))}"
      f" (H0002 for members aged {H0002_MIN_AGE} and over). Counted BOTH ways: calendar year and "
      f"the benefit year starting 1 "
      f"{date(2000, BENEFIT_YEAR_START_MONTH, 1).strftime('%B')}. A member above the limit on one "
      f"basis and below it on the other is marked AMBIGUOUS, not reported as a finding.")

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
     "NOT RUN: paid over billed, on A_Provider_Profile and on the code sheets, AND the S9480 "
     "pricing era check, which dates the move from By Report pricing to the fixed rate in the "
     "client's own data by watching when paid over billed stops clustering at 58.66%. Billed "
     "dollars are reported as blank rather than zero, because a zero would read as nothing "
     "billed."),
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
     "NOT RUN: the S9480 prior authorization test. S9480 moved to prior authorization for dates of "
     "service on and after 2025-07-01 for fee for service and 2024-10-15 for Arizona Complete "
     "Health, and without this column there is no way to tell a line that carried an "
     "authorization from one that did not. A_Provider_Profile also cannot count lines carrying "
     "a prior authorization."),
    (DOB_COL, "DATE_OF_BIRTH", "member date of birth",
     "NOT RUN: member age at the date of service on A_Provider_Profile, and the age split in the "
     "bed hold and therapeutic leave test, where members 21 and over have separate limits from "
     "members under 21. Without it the leave day counts are reported with no limit applied."),
    (POS_CODE_COL or POS_DESC_COL, "POS_CODE or POS_DESC", "place of service",
     "NOT RUN: the H0018 place of service test. Behavioural health residential facilities must "
     "bill place of service 56 from 2023-12-01 and without this column no line can be checked "
     "against it. A_Provider_Profile also cannot list the places of service, so a residential "
     "setting cannot be told apart from an office one when a Part E row is reviewed."),
    (CLM_TYP_COL, "CLM_TYP_DESC", "claim type",
     "A_Provider_Profile cannot show the claim type mix."),
    (FORM_TYP_COL, "FORM_TYP_DESC", "claim form",
     "Professional lines cannot be told apart from institutional ones on A_Provider_Profile."),
    (REV_CODE_COL or REV_DESC_COL, "a revenue code column, detected by name",
     "revenue code on institutional lines",
     "NOT RUN: the bed hold and therapeutic leave test. Therapeutic leave is revenue code 0183 "
     "and bed hold is 0189, and neither can be seen without a revenue code column. Institutional "
     "lines also carry no revenue code on the sheets, and residential and subacute stays are "
     "priced by revenue code rather than by an H code, so without this an institutional line "
     "cannot be told apart from a professional one by its content."),
    (LOB_COL, "line_of_business", "line of business",
     "WEAKENS the S9480 prior authorization test: fee for service moved to prior authorization on "
     "2025-07-01 and Arizona Complete Health on 2024-10-15, and without this column the two "
     "cannot be told apart, so the LATER fee for service date is applied to every line, which "
     "reports fewer findings rather than more. A_Provider_Profile also cannot list the lines of "
     "business."),
    (LOB_ORG_COL, "lob_org_name", "plan or programme name", "No plan context on the sheets."),
    (IHS_638_COL, "IHS_638", "IHS or 638 indicator",
     "WEAKENS the S9480 prior authorization test: IHS and 638 facilities are EXEMPT from it, and "
     "without this column exempt lines cannot be removed, so an exempt line could appear as a "
     "finding. The sheet says so on every row when that happens."),
    (IHS_PROV_COL, "IHS_PROVIDER", "IHS provider indicator",
     "The IHS and 638 exemption in the S9480 prior authorization test falls back to the IHS_638 "
     "column alone."),
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
# MAGIC ## 6. Load every claim for the providers under review
# MAGIC
# MAGIC This cell loads the claims. Read the first sentence of the Criteria sheet before reading any
# MAGIC number that comes out of it.
# MAGIC
# MAGIC **Every claim for these providers is loaded, not only the focus codes.** The focus codes and
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
_level_map = F.create_map(*[F.lit(x) for kv in CODE_LEVEL_OF_CARE.items() for x in kv])


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


HAVE_SRV_END = have(SRV_END_COL)


def prepare(sdf):
    """Turn raw claim lines into the shape every part of this notebook works from."""
    out = (
        sdf
        .withColumn(PROC_COL, F.upper(F.trim(F.coalesce(F.col(PROC_COL).cast("string"), F.lit("")))))
        .withColumn("srv_date", F.to_date(F.col(SRV_BEG_COL).cast("string"), "yyyyMMdd"))
        # The service end date is optional. Referring to it unconditionally would stop the whole
        # notebook on an extract that does not carry it, which is exactly the silent-failure
        # shape this project guards against, so it is detected like every other optional column.
        .withColumn("_end_raw", F.to_date(F.col(SRV_END_COL).cast("string"), "yyyyMMdd")
                    if HAVE_SRV_END else F.lit(None).cast("date"))
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
        # Whether the minutes on this line rest on an AHCCCS statement or only on the national
        # HCPCS descriptor. The day sheets report the second kind separately, so an hour total
        # can never quietly rest on a duration AHCCCS has not published.
        .withColumn("duration_is_from_the_national_descriptor_only",
                    F.col(PROC_COL).isin(sorted(DURATION_FROM_NATIONAL_DESCRIPTOR_ONLY)))
        .withColumn("is_session", F.col("basis") == F.lit("PER_SESSION"))
        .withColumn("sessions", F.when(F.col("basis") == F.lit("PER_SESSION"), F.col("qty"))
                                 .otherwise(F.lit(0.0)))
        .withColumn("is_per_diem_any", F.col(PROC_COL).isin(sorted(PER_DIEM_ALL)))
        .withColumn("is_per_diem_testable", F.col(PROC_COL).isin(sorted(PER_DIEM_TESTABLE)))
        .withColumn("is_unverified_basis", F.col(PROC_COL).isin(sorted(UNVERIFIED_BASIS_CODES)))
        .withColumn("is_focus_code", F.col(PROC_COL).isin(FOCUS_CODES))
        .withColumn("proc_desc", _txt(PROC_DESC_COL))
        # The Monday of the service week. Used by the minimum intensity questions. Built with
        # trunc rather than a week number so a week that straddles a year end is still one week.
        .withColumn("week_start", F.trunc("srv_date", "week"))
        .withColumn("level_of_care", F.coalesce(_level_map[F.col(PROC_COL)],
                                                F.lit("not a level of care code")))
        # The only per diem lines a DAY may be counted from. H0019, H2018 and H2038 are per diems
        # by description but AHCCCS has never published what one of their units covers, so they
        # keep their dollars and contribute nothing to any census, stay or day count.
        .withColumn("counts_as_a_day_of_care",
                    F.col(PROC_COL).isin(sorted(PER_DIEM_TESTABLE)))
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
              .withColumn("is_focus_diag", F.col("diag1").isin(FOCUS_DIAGS) if HAVE_FOCUS_DIAGS
                          else F.lit(False)))
    # Any-position focus diagnosis, reported alongside the primary so the investigator can see
    # whether a focus diagnosis is the reason for the visit or a secondary note.
    if DIAGS and HAVE_FOCUS_DIAGS:
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
           # Place of service compared without leading zeros, so 056 and 56 are the same place
           # and a formatting difference cannot manufacture a finding against a facility that
           # billed correctly.
           .withColumn("pos_code_clean",
                       F.regexp_replace(F.upper(F.trim(_txt(POS_CODE_COL))), r"^0+", ""))
           .withColumn("pos_desc", _txt(POS_DESC_COL))
           .withColumn("clm_typ", _txt(CLM_TYP_COL))
           .withColumn("form_typ", _txt(FORM_TYP_COL))
           .withColumn("rev_desc", _txt(REV_DESC_COL))
           .withColumn("lob", _txt(LOB_COL))
           .withColumn("lob_org", _txt(LOB_ORG_COL))
           .withColumn("ihs_638", _txt(IHS_638_COL))
           .withColumn("ihs_provider", _txt(IHS_PROV_COL))
           # IHS and 638 facilities are EXEMPT from the S9480 prior authorization requirement, so
           # this flag can only ever remove a line from a finding, never add one. Anything that
           # is not plainly a no is treated as an IHS or 638 line, which is the conservative
           # direction: a wrongly exempted line is a missed lead, a wrongly flagged one is a
           # false accusation against a facility the rule never applied to.
           .withColumn("is_ihs_or_638",
                       F.upper(F.trim(_txt(IHS_638_COL))).isin("Y", "YES", "1", "T", "TRUE") |
                       F.upper(F.trim(_txt(IHS_PROV_COL))).isin("Y", "YES", "1", "T", "TRUE"))
           .withColumn("lob_upper", F.upper(F.trim(_txt(LOB_COL))))
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

print(f"\nClaim lines for the providers under review: {N_LINES:,}"
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
          # The part of the day's minutes that rests on a duration taken from the national HCPCS
          # descriptor rather than from an AHCCCS statement. It is a SUBSET of the minutes above,
          # not an addition to them, and it is reported so that an hour total can be read with
          # the right amount of confidence rather than with all of it.
          .withColumn("descriptor_only_minutes",
                      F.when(F.col("duration_is_from_the_national_descriptor_only"),
                             F.col("clin_minutes")).otherwise(F.lit(0.0)))
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
# MAGIC ## 8. Part A. Who these providers are
# MAGIC
# MAGIC One row for each provider, side by side, so they can be compared at a glance. It answers the
# MAGIC questions an investigator asks before looking at any billing: what is this provider, is it a
# MAGIC person or an organisation, is a payment suspension set, how long have they been billing, how
# MAGIC much have they been paid, how many members, who submits their claims and who refers to them.
# MAGIC
# MAGIC **Why the person or organisation question comes first.** A servicing provider ID on this table
# MAGIC may be an organisational NPI rather than one human being. If it is an organisation, then adding
# MAGIC up the hours billed in a day and comparing the total against 24 is meaningless, because several
# MAGIC members of staff bill under one number. This cell decides it from the provider type
# MAGIC description on the claims rather than assuming it from the name in the request, and whatever it
# MAGIC decides is carried through the rest of the workbook: if the answer is organisation, no day
# MAGIC total anywhere in this workbook is described as one person's day.
# MAGIC
# MAGIC An Integrated Clinic, a behavioural health outpatient clinic, a residential facility and a
# MAGIC community service agency are all organisations. A licensed professional counsellor, a
# MAGIC psychologist and a nurse practitioner are people. The lists this cell matches against are
# MAGIC printed in the code and the verdict always carries the reason it reached, because a bare
# MAGIC verdict could not be checked.
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
    "INTEGRATED CLINIC", "RESIDENTIAL", "OUTPATIENT TREATMENT",
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
        # A provider whose name was not supplied with the request has nothing to check. Comparing
        # against an empty string would report a match for every provider in the table, which
        # reads as confirmation that the right provider was pulled when nothing was confirmed.
        "name_matches_the_request":
            "no name was supplied with the request, so there was nothing to check"
            if not TARGET_NAMES_GIVEN.get(pid, "").strip() else
            ("yes" if (TARGET_NAMES_GIVEN.get(pid, "").strip().upper().replace(" ", "")
                       in (ident["name_on_claims"] or "").upper().replace(" ", "")) else
             "NO: check you have the right provider"),
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
IS_ORGANISATION = {p: PROFILES[p].get("person_or_organisation") == "organisation"
                   for p in TARGETS}
ANY_ORGANISATION = any(IS_ORGANISATION.values())
ALL_INDIVIDUALS = all(IS_PERSON.values()) and bool(TARGETS)
SUSP_VALUES = {p: PROFILES[p].get("payment_suspension_indicator_raw") for p in TARGETS}

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. Part A continued. Which instrument fits this focus set
# MAGIC
# MAGIC **This is the cell that decides what the rest of the workbook leads with, so read it before
# MAGIC anything else.**
# MAGIC
# MAGIC A procedure code is billed either by the clock or by the day, and the two need completely
# MAGIC different arithmetic. A timed code carries minutes, so the natural question is how many hours
# MAGIC were billed in a day. A per diem carries no minutes at all: one unit is one day of care,
# MAGIC whatever happened inside it. Ask a per diem how many hours it represents and the honest answer
# MAGIC is zero, and a reader who sees near zero hours against a provider billing millions of dollars
# MAGIC of residential care concludes wrongly that nothing is happening.
# MAGIC
# MAGIC So this cell works out which kind of set it has been given, by paid dollars rather than by
# MAGIC counting codes, because one high volume code decides the character of a set more than three
# MAGIC quiet ones do.
# MAGIC
# MAGIC - **Timed set.** The day section leads, and it behaves exactly as it always has.
# MAGIC - **Per diem set.** The daily census and the episode sections lead, and the day section is
# MAGIC   marked as **not the right instrument** with the reason printed on the sheet. The hours are
# MAGIC   still computed and still shown, because a per diem provider that also bills timed services
# MAGIC   has real hours worth seeing. They simply stop being the headline.
# MAGIC
# MAGIC **And a separate question with the same consequence.** If the servicing provider is an
# MAGIC organisation rather than a person, no day total is one person's day whatever the codes are.
# MAGIC Several members of staff bill under one number, so comparing a day total against twenty four
# MAGIC hours means nothing. That is stated here, on the profile sheet, on the day sheets and in the
# MAGIC email body.

# COMMAND ----------

# Dollars on the focus codes, split by what kind of unit each code carries. Dollars rather than
# a count of codes: a set of six where one per diem carries 90% of the money is a per diem set
# whatever the other five are.
_focus_money = (df_line.filter(F.col("is_focus_code"))
                .groupBy(PROC_COL)
                .agg(F.round(F.sum("paid"), 2).alias("paid"),
                     F.round(F.sum("qty"), 2).alias("units"),
                     F.count(F.lit(1)).alias("lines"))
                .toPandas())
FOCUS_SHAPE_ROWS = []
for _c in FOCUS_CODES:
    _m = _focus_money[_focus_money[PROC_COL] == _c]
    _paid = float(_m.iloc[0]["paid"]) if len(_m) else 0.0
    _basis = CODE_BASIS.get(_c, "UNCLASSIFIED")
    # A code is counted as a per diem here when it is documented as a per diem anywhere,
    # including the ones whose unit basis AHCCCS has never published. The question this split
    # answers is "can an hours total say anything about this code", and the answer for an
    # unverified per diem is no, exactly as it is for a published one.
    _kind = ("per diem" if _c in PER_DIEM_ALL else
             "timed" if _basis in ("TIMED_15", "TIMED_60") else
             "per session" if _basis == "PER_SESSION" else "other")
    FOCUS_SHAPE_ROWS.append({
        "procedure_code": _c, "unit_basis": _basis, "counts_as": _kind,
        "paid": _paid,
        "lines": int(_m.iloc[0]["lines"]) if len(_m) else 0,
        "units": float(_m.iloc[0]["units"]) if len(_m) else 0.0,
        "carries_minutes": "yes" if _basis in ("TIMED_15", "TIMED_60") else "no",
        "what_one_unit_means": {"TIMED_15": "15 minutes", "TIMED_60": "60 minutes",
                                "PER_DIEM": "one day", "PER_SESSION": "one session, any length",
                                "PER_DOSE": "one dose", "PER_MONTH": "one calendar month",
                                "UNVERIFIED": "NOT PUBLISHED BY AHCCCS",
                                "NOT_IN_AHCCCS_BH_SET": "not an AHCCCS behavioural health code",
                                }.get(_basis, "not classified"),
    })
focus_shape_df = pd.DataFrame(FOCUS_SHAPE_ROWS)

_per_diem_paid = float(focus_shape_df.loc[focus_shape_df.counts_as == "per diem", "paid"].sum())
_timed_paid = float(focus_shape_df.loc[focus_shape_df.counts_as == "timed", "paid"].sum())
_focus_paid_total = float(focus_shape_df["paid"].sum())
PER_DIEM_SHARE_OF_FOCUS_PAID = (_per_diem_paid / _focus_paid_total) if _focus_paid_total > 0 else None

if FOCUS_SHAPE_SETTING != "auto":
    FOCUS_SHAPE = "per diem" if FOCUS_SHAPE_SETTING == "per_diem" else "timed"
    FOCUS_SHAPE_BASIS = (f"set by hand with the focus_shape setting ({FOCUS_SHAPE_SETTING}), not "
                         f"worked out from the dollars")
elif _focus_paid_total > 0:
    FOCUS_SHAPE = "per diem" if _per_diem_paid > _timed_paid else "timed"
    FOCUS_SHAPE_BASIS = (
        f"worked out from paid dollars on the focus codes: ${_per_diem_paid:,.0f} on per diem "
        f"codes against ${_timed_paid:,.0f} on timed codes, out of ${_focus_paid_total:,.0f} on "
        f"the focus codes altogether")
else:
    # No money on the focus codes at all, so the dollars cannot decide. Fall back to counting
    # codes and say plainly that is what happened, because a shape decided on nothing would be
    # the quietest possible way for this notebook to mislead somebody.
    _n_pd = int((focus_shape_df.counts_as == "per diem").sum())
    _n_t = int((focus_shape_df.counts_as == "timed").sum())
    FOCUS_SHAPE = "per diem" if _n_pd > _n_t else "timed"
    FOCUS_SHAPE_BASIS = (
        f"NO paid dollars on any focus code for these providers, so the dollars could not decide "
        f"the shape. It was decided by counting codes instead ({_n_pd} per diem against {_n_t} "
        f"timed), and every number that follows should be read with that in mind: an empty focus "
        f"set usually means the provider IDs or the code list are wrong.")

SHAPE_IS_PER_DIEM = FOCUS_SHAPE == "per diem"

HOURS_INSTRUMENT_STATUS = (
    "NOT THE RIGHT INSTRUMENT FOR THIS FOCUS SET" if SHAPE_IS_PER_DIEM else
    "the right instrument for this focus set")
HOURS_INSTRUMENT_WHY = (
    ("Most of the money on the focus codes sits on PER DIEM codes. A per diem carries no minutes: "
     "one unit is one day of care whatever happened inside it, so it can never add an hour to a "
     "day total. Reading the hours below as a measure of this provider's activity would say that "
     "almost nothing happened, which would be wrong. The daily census and the episode sections "
     "are the instrument for a per diem set: how many people were on a per diem each day, and how "
     "long each of them stayed. The hours are still computed and still shown, because a provider "
     "billing per diems may also bill timed services and those hours are real, but they are not "
     "the headline and a near zero here is the arithmetic working correctly.")
    if SHAPE_IS_PER_DIEM else
    ("Most of the money on the focus codes sits on TIMED codes, which carry minutes, so the day "
     "section is the right place to start and behaves as it always has."))

ORG_DAY_WARNING = (
    ("At least one provider under review is an ORGANISATION, not a person. An organisation's day "
     "is not one person's day: several members of staff bill under one number, so no day total "
     "anywhere in this workbook may be compared against twenty four hours or read as one "
     "person's working day. Where a day total is shown for an organisation it is the whole "
     "organisation's billing for that day and nothing more.")
    if ANY_ORGANISATION else
    ("Every provider under review looks like an individual practitioner on the claims, so a day "
     "total can be read as one person's day, with the group billing caveat below."))

instrument_df = pd.DataFrame([
    ["Shape of the focus set", FOCUS_SHAPE],
    ["How that was decided", FOCUS_SHAPE_BASIS],
    ["Share of focus dollars on per diem codes",
     f"{PER_DIEM_SHARE_OF_FOCUS_PAID:.1%}" if PER_DIEM_SHARE_OF_FOCUS_PAID is not None
     else "no focus dollars at all"],
    ["What leads this workbook",
     "The daily census and the episodes: Parts H and I. Read those before the day section."
     if SHAPE_IS_PER_DIEM else
     "The day: Part D. Read that before the census and episode sections."],
    ["Status of the hours in Part D", HOURS_INSTRUMENT_STATUS],
    ["Why", HOURS_INSTRUMENT_WHY],
    ["Person or organisation",
     "; ".join(f"{p}: {PROFILES[p].get('person_or_organisation', 'unknown')}" for p in TARGETS)],
    ["What that means for every day total", ORG_DAY_WARNING],
], columns=["Item", "What it is"])

print(f"Shape of the focus set: {FOCUS_SHAPE.upper()}")
print(f"  {FOCUS_SHAPE_BASIS}")
print(f"\nHours in Part D: {HOURS_INSTRUMENT_STATUS}")
print(f"  {HOURS_INSTRUMENT_WHY}")
print(f"\n{ORG_DAY_WARNING}")
print("\nFocus codes by what one unit of each means:")
print(focus_shape_df[["procedure_code", "unit_basis", "counts_as", "what_one_unit_means",
                      "lines", "units", "paid"]].to_string(index=False))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 10. Part B. The focus lens
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
    out["focus_code_group"] = out[PROC_COL].map(lambda c: CODE_FOCUS_GROUP.get(c, "other"))
    out["level_of_care"] = out[PROC_COL].map(
        lambda c: CODE_LEVEL_OF_CARE.get(c, "not a level of care code"))
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
    # basis_note stays what it has always been: the note for a code that was asked for, and a
    # pointer to the dictionary for one that was not. The notebook now carries notes for codes
    # beyond the current focus set, and letting them leak into this column would change what the
    # All_Codes sheet says about a code nobody asked about.
    out["basis_note"] = [
        FOCUS_CODE_NOTE[c] if (c in FOCUS_CODES and c in FOCUS_CODE_NOTE)
        else "not a focus code; basis from the dictionary in section 2"
        for c in out[PROC_COL]]
    # The full note for every code this notebook knows about, in a column of its own, so the
    # richer description is available without changing the column above.
    out["code_note"] = out[PROC_COL].map(
        lambda c: FOCUS_CODE_NOTE.get(c, f"{CODE_BASIS.get(c, 'UNCLASSIFIED')}; see the Criteria "
                                         f"sheet and the dictionary in section 2"))
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
                               "focus_code_group": CODE_FOCUS_GROUP.get(c, "other"),
                               "level_of_care": CODE_LEVEL_OF_CARE.get(
                                   c, "not a level of care code"),
                               "basis_note": FOCUS_CODE_NOTE.get(c, ""),
                               "procedure_description": "this provider never billed this code"})
if _zero_rows:
    focus_codes_df = pd.concat([focus_codes_df, pd.DataFrame(_zero_rows)], ignore_index=True) \
        if len(focus_codes_df) else pd.DataFrame(_zero_rows)
FOCUS_CODE_SHEET_COLS = ["provider_id", "focus_code_group", PROC_COL, "procedure_description",
                         "unit_basis", "level_of_care",
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
# Grouped so the two kinds of service in a focus set are never read as one list. Treatment and
# level of care codes say where a member was being treated and how intensively. Assessment,
# screening and education codes are the front door that decides who is admitted. Comparing a
# number from one block against a number from the other answers nothing.
if len(focus_codes_df):
    focus_codes_df["_g"] = focus_codes_df["focus_code_group"].map(
        {g: i for i, g in enumerate(FOCUS_GROUP_ORDER)}).fillna(len(FOCUS_GROUP_ORDER))
    focus_codes_df["_c"] = focus_codes_df[PROC_COL].map(
        {c: i for i, c in enumerate(FOCUS_CODES)}).fillna(len(FOCUS_CODES))
    focus_codes_df = (focus_codes_df.sort_values(["provider_id", "_g", "_c"])
                      .drop(columns=["_g", "_c"]).reset_index(drop=True))

# One row per provider per block, with the whole book comparison for each block.
_grp_rows = []
for pid in TARGETS:
    _tbl = ALL_CODE_TABLES.get(pid, pd.DataFrame())
    _tot_paid = float(_tbl["paid"].sum()) if len(_tbl) else 0.0
    _tot_lines = int(_tbl["lines"].sum()) if len(_tbl) else 0
    for g in FOCUS_GROUP_ORDER:
        _codes = [c for c in FOCUS_CODES if CODE_FOCUS_GROUP.get(c, "other") == g]
        if not _codes:
            continue
        _sel = _tbl[_tbl[PROC_COL].isin(_codes)] if len(_tbl) else pd.DataFrame()
        _grp_rows.append({
            "provider_id": pid,
            "focus_code_group": g,
            "codes_in_this_block": ", ".join(_codes),
            "codes_in_this_block_the_provider_billed":
                ", ".join(sorted(set(_sel[PROC_COL]))) if len(_sel) else "none of them",
            "lines": int(_sel["lines"].sum()) if len(_sel) else 0,
            "units": round(float(_sel["units"].sum()), 2) if len(_sel) else 0.0,
            "paid": round(float(_sel["paid"].sum()), 2) if len(_sel) else 0.0,
            "members": int(_sel["members"].max()) if len(_sel) else 0,
            "share_of_the_whole_book_by_paid": _rate(
                float(_sel["paid"].sum()) if len(_sel) else 0.0, _tot_paid),
            "share_of_the_whole_book_by_lines": _rate(
                int(_sel["lines"].sum()) if len(_sel) else 0, _tot_lines),
            "what_this_block_is":
                "Where a member is being treated and how intensively. These are the levels of "
                "care: residential, partial hospitalisation, intensive outpatient, crisis and day "
                "treatment." if g == "treatment and level of care" else
                "The front door. Assessment, screening for eligibility for admission to a "
                "treatment programme, and education and training. These decide who is admitted."
                if g == "assessment, screening and education" else
                "Focus codes that are neither a level of care nor a front door service.",
            "members_note": "members is the largest member count on any single code in this "
                            "block, not a distinct count across the block, because the per code "
                            "table cannot be added up that way without double counting.",
        })
focus_group_summary_df = pd.DataFrame(_grp_rows)

print("Focus codes, by block:")
if len(focus_codes_df):
    print(focus_codes_df[_present(focus_codes_df,
                                  ["provider_id", "focus_code_group", PROC_COL, "unit_basis",
                                   "lines", "units",
                                   "paid", "members", "member_days",
                                   "mean_units_per_member_day", "max_units_in_one_member_day",
                                   "share_of_provider_paid"])].to_string(index=False))
if len(focus_group_summary_df):
    print("\nThe two halves of the focus set against the whole book:")
    print(focus_group_summary_df[["provider_id", "focus_code_group", "lines", "units", "paid",
                                  "share_of_the_whole_book_by_paid"]].to_string(index=False))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 11. Part B continued. The focus diagnoses, and the code by diagnosis cross tabulation
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
#
# When no focus diagnoses were supplied there is nothing to cross tabulate, and the sheet says
# that in words. It does NOT write a grid of zeros: a zero grid reads as "checked, nothing found"
# and the truth is that nobody asked the question.
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
for pid in (TARGETS if HAVE_FOCUS_DIAGS else []):
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
            "primary_diagnosis": "ALL FOCUS DIAGNOSES TOGETHER",
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
# MAGIC ## 12. Part C. The whole book, so the lens has a denominator
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
        print(f"   The {len(FOCUS_CODES)} focus codes are "
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
# MAGIC ## 13. Part D. The day
# MAGIC
# MAGIC **Check the instrument cell in Part A before reading a single number here.** If the focus set
# MAGIC is mostly per diem codes, this section is not the right instrument and the cell says so in
# MAGIC plain language. A per diem carries no minutes, so a provider billing nothing but per diems
# MAGIC will show close to zero hours here however much they are billing, and that is the arithmetic
# MAGIC working correctly rather than a quiet provider. The census in Part H and the episodes in Part
# MAGIC I are the instrument for that shape of question. The hours are still computed and still shown,
# MAGIC because a per diem provider may also bill timed services and those hours are real.
# MAGIC
# MAGIC **And if the provider is an organisation, this is not one person's day.** Several members of
# MAGIC staff bill under one number, so the total belongs to the organisation and comparing it against
# MAGIC twenty four hours means nothing. Part A says which providers that applies to and every row of
# MAGIC the day summary repeats it.
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
         F.round(F.sum("descriptor_only_minutes"), 1)
          .alias("minutes_from_a_duration_in_the_national_descriptor_only"),
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
           # Carried on every row so an investigator who opens this sheet on its own, without
           # having read Part A, still cannot mistake a near zero hour total for a quiet
           # provider when the focus codes are per diems.
           "is_this_the_right_instrument": HOURS_INSTRUMENT_STATUS,
           "why": HOURS_INSTRUMENT_WHY,
           "active_days": len(cd),
           "total_timed_hours": round(float(cd.timed_hours.sum()), 1) if len(cd) else 0.0,
           "group_minutes_set_aside": round(float(cd.group_minutes_set_aside.sum()), 1) if len(cd) else 0.0,
           # A subset of total_timed_hours, not an addition to it. Where this is more than a
           # sliver of the total, the hours rest partly on a duration AHCCCS has not published.
           "minutes_from_a_duration_in_the_national_descriptor_only":
               round(float(cd["minutes_from_a_duration_in_the_national_descriptor_only"].sum()), 1)
               if len(cd) else 0.0,
           "share_of_the_timed_minutes_resting_on_a_descriptor_only_duration":
               _rate(float(cd["minutes_from_a_duration_in_the_national_descriptor_only"].sum()),
                     float(cd.timed_minutes.sum())) if len(cd) else None,
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
# MAGIC ## 14. Part E. The published rules, tested
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
# MAGIC **One limitation stated plainly.** Tests 1, 4 and 5 can only see what these providers and
# MAGIC their billing entities billed. A member who received a per diem from a completely unrelated
# MAGIC provider on the same day is invisible here. The member journey screen is the notebook that does
# MAGIC that arithmetic.

# COMMAND ----------

# Every per diem line, from any provider and any billing entity, for the members these
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
# MAGIC ## 15. Part E continued. The published rules that govern a per diem programme
# MAGIC
# MAGIC Eleven more tests, every one of them anchored on a published AHCCCS rule or on the arithmetic
# MAGIC of the code itself. **None of them needs a peer group, which is why they rank above everything
# MAGIC statistical in this workbook.** Each gets its own sheet, one row per finding, and every row
# MAGIC names the member, the dates, the codes, the units, the dollars, the billing entity and the
# MAGIC servicing provider, so a row is an instruction to pull a record rather than a statistic.
# MAGIC
# MAGIC | # | Test | Where the rule comes from |
# MAGIC |---|---|---|
# MAGIC | 6 | One per diem per member per day | Effective 2024-02-06, codified in the Covered Behavioral Health Services Guide dated 2024-10-01 |
# MAGIC | 7 | Levels of care that cannot coexist | The per diem rule plus the verified same day prohibitions. The matrix is printed on the Criteria sheet |
# MAGIC | 8 | H2012 above 5 units in a day | FFS chapter 19: the day treatment family is H2012 per hour up to 5 hours, then H2015, then H2016 |
# MAGIC | 9 | The 2023-07-17 expanded documentation list | Keys on claim RECEIPT date, not service date |
# MAGIC | 10 | S9480 once a day | Covered Behavioral Health Services Guide |
# MAGIC | 11 | S9480 prior authorization | 2025-07-01 fee for service, 2024-10-15 Arizona Complete Health, IHS and 638 exempt |
# MAGIC | 12 | S9480 minimum intensity | 3 hours a day and at least 2 days a week |
# MAGIC | 13 | H0035 minimum intensity | 20 hours a week |
# MAGIC | 14 | H0018 place of service 56 | Required of behavioural health residential facilities from 2023-12-01 |
# MAGIC | 15 | Provider type against the codes billed | The AHCCCS code to provider type matrix |
# MAGIC | 16 | Bed hold and therapeutic leave | Revenue codes 0183 and 0189, with contract year limits |
# MAGIC
# MAGIC **Two things these tests will not do.**
# MAGIC
# MAGIC They will not use the service date when a rule says receipt date. Test 9 keys on the date the
# MAGIC claim arrived because the published wording is "claims received on and after", and if this
# MAGIC extract carries no receipt date the test reports NOT RUN with the reason rather than falling
# MAGIC back to the service date. The claim lag would mis-date it and a mis-dated documentation test
# MAGIC produces a false finding against a real provider.
# MAGIC
# MAGIC They will not call a minimum intensity shortfall a finding. A per diem claim cannot carry
# MAGIC hours, so nothing here can show that a programme ran for fewer than three hours. What a claim
# MAGIC can show is how many days in a week the programme was billed, so tests 12 and 13 are phrased
# MAGIC as questions about whether the minimum was met and are labelled that way on every row.

# COMMAND ----------

# Status for every new test, so Analysis_Status can say what ran and what could not. A test that
# cannot run says so here with the reason; it never writes a zero.
NEW_TEST_STATUS = {}
TARGET_ENTITY_SET = set(TARGET_ENTITIES)


def _mark(name, ok, reason=""):
    NEW_TEST_STATUS[name] = "ran" if ok else f"NOT RUN: {reason}"
    return ok


def _entity_list(col):
    return F.concat_ws(", ", F.array_sort(F.collect_set(col)))


# Every line at any level of care, for the members these providers treated, from any provider and
# any billing entity. The per diem source above cannot answer the level of care question on its
# own because H2012 is a timed code, not a per diem, and day treatment is a level of care.
LEVEL_ALL_CODES = sorted(RESIDENTIAL_CODES | DAY_OR_PARTIAL_CODES | INTENSIVE_OUTPATIENT_CODES)
if N_TARGET_MEMBERS:
    _lvl_raw = spark.table(SERVICE_TABLE).filter(
        F.upper(F.trim(F.coalesce(F.col(PROC_COL).cast("string"), F.lit("")))).isin(LEVEL_ALL_CODES))
    df_entity_level = (prepare(_lvl_raw)
                       .join(F.broadcast(_members_df), MEMBER_COL, "inner")
                       .cache())
    N_ENTITY_LEVEL = df_entity_level.count()
else:
    df_entity_level = df_line.filter(F.lit(False))
    N_ENTITY_LEVEL = 0
print(f"Level of care lines for those members, from any provider: {N_ENTITY_LEVEL:,}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Rule 6. One per diem per member per day
# MAGIC
# MAGIC **This is the central test when the focus codes are per diems, and five of six usually are.**
# MAGIC AHCCCS allows one per diem behavioural health service per member per day. One unit of a per diem
# MAGIC is one whole day of care, so two of them on one day cannot both be right.
# MAGIC
# MAGIC The sheet flags the member days where **one billing entity** billed two or more per diem codes.
# MAGIC Where the second per diem came from a **different** entity the row is counted on its own sheet and
# MAGIC never flagged: this provider did not bill somebody else's claim. Each row carries every per diem
# MAGIC code on that day and the dollars beyond the single largest one, because only one of them can
# MAGIC stand.

# COMMAND ----------

# ── Rule 6. One per diem behavioural health service per member per day ───────────────────────
#
# Five of the six codes in a residential or programme focus set are per diems, so this is the
# central test for that shape of request rather than an incidental one.
#
# WHAT GETS FLAGGED AND WHAT ONLY GETS COUNTED. The published rule is written at the member day
# level: one per diem per member per day, whoever billed it. This notebook flags the member days
# where ONE billing entity billed two or more per diem codes, because that is the version this
# provider can be held to and the version a per claim edit should have caught. Where the second
# per diem came from a DIFFERENT entity the row is counted in a column of its own and never
# flagged: this provider did not bill the other entity's per diem, and no per claim edit could
# have seen the pair. Those cross entity days are still worth somebody's attention, and the
# member journey screen is the notebook that does cross provider member day arithmetic properly.
#
# H0019, H2018 and H2038 keep their dollars and cannot drive a finding. A collision that rests on
# one of them is listed with a column saying the published rule cannot be applied to it, because
# nobody has published how many days one of their units covers.
_pd_src = entity_pd_src if N_ENTITY_PER_DIEM else df_line.filter(F.lit(False))
_pd_code = (_pd_src.groupBy(MEMBER_COL, "srv_date", BPROV_COL, PROC_COL)
            .agg(F.round(F.sum("qty"), 2).alias("code_units"),
                 F.round(F.sum("paid"), 2).alias("code_paid"),
                 F.count(F.lit(1)).alias("code_lines"),
                 _entity_list(SPROV_COL).alias("code_servicing_providers"),
                 _entity_list("claim_id").alias("code_claim_ids"),
                 F.max(F.col("counts_as_a_day_of_care").cast("int")).alias("code_counts_as_a_day")))

# What the whole member day looked like, across every entity.
_pd_day_all = (_pd_code.groupBy(MEMBER_COL, "srv_date")
               .agg(F.countDistinct(PROC_COL).alias("per_diem_codes_on_the_member_day"),
                    F.countDistinct(BPROV_COL).alias("billing_entities_on_the_member_day"),
                    _entity_list(PROC_COL).alias("every_per_diem_code_that_day"),
                    _entity_list(BPROV_COL).alias("every_billing_entity_that_day"),
                    F.round(F.sum("code_paid"), 2).alias("per_diem_paid_that_day"),
                    F.round(F.max("code_paid"), 2).alias("largest_single_per_diem_paid")))

# What ONE entity billed on that member day.
_pd_entity = (_pd_code.groupBy(MEMBER_COL, "srv_date", BPROV_COL)
              .agg(F.countDistinct(PROC_COL).alias("per_diem_codes_this_entity_billed"),
                   F.countDistinct(F.when(F.col("code_counts_as_a_day") == 1, F.col(PROC_COL)))
                    .alias("codes_with_a_published_unit"),
                   _entity_list(PROC_COL).alias("per_diem_codes"),
                   _entity_list(F.when(F.col("code_counts_as_a_day") == 0, F.col(PROC_COL)))
                    .alias("codes_with_no_published_unit"),
                   F.round(F.sum("code_units"), 2).alias("per_diem_units"),
                   F.round(F.sum("code_paid"), 2).alias("per_diem_paid"),
                   _entity_list("code_servicing_providers").alias("servicing_providers"),
                   _entity_list("code_claim_ids").alias("claim_ids")))

one_per_diem_df = (
    _pd_entity.filter(F.col("per_diem_codes_this_entity_billed") >= F.lit(2))
    .filter(F.col(BPROV_COL).isin(sorted(TARGET_ENTITY_SET)) if TARGET_ENTITY_SET else F.lit(False))
    .join(_pd_day_all, [MEMBER_COL, "srv_date"], "left")
    .withColumn("same_billing_entity_billed_them_all", F.lit(True))
    .withColumn("can_be_tested_against_the_published_rule",
                F.col("codes_with_a_published_unit") >= F.lit(2))
    .withColumn("dollars_beyond_the_largest_per_diem",
                F.round(F.col("per_diem_paid_that_day") - F.col("largest_single_per_diem_paid"), 2))
    .withColumn("rule", F.lit("only ONE per diem behavioural health service per member per day, "
                              "effective 2024-02-06 and codified in the Covered Behavioral Health "
                              "Services Guide dated 2024-10-01"))
    .withColumn("what_to_pull", F.lit("Pull every claim named in claim_ids for this member on this "
                                      "date and establish which single per diem stands."))
    .orderBy(F.desc("dollars_beyond_the_largest_per_diem"))
    .limit(DETAIL_ROWS).toPandas())

# MEASURED, NEVER FLAGGED: the member day carried per diems from more than one billing entity and
# no single entity billed two. A per claim edit could not have seen this and this provider did not
# bill the other entity's per diem.
_entity_max = (_pd_entity.groupBy(MEMBER_COL, "srv_date")
               .agg(F.max("per_diem_codes_this_entity_billed").alias("most_codes_from_one_entity")))
one_per_diem_cross_entity_df = (
    _pd_day_all.join(_entity_max, [MEMBER_COL, "srv_date"], "left")
    .filter((F.col("billing_entities_on_the_member_day") >= F.lit(2)) &
            (F.col("most_codes_from_one_entity") < F.lit(2)))
    .withColumn("why_this_is_not_flagged",
                F.lit("The per diems came from different billing entities. This provider did not "
                      "bill the other entity's per diem and no per claim edit could have seen the "
                      "pair, so this is counted and listed, never flagged. The member journey "
                      "screen is the notebook that does cross provider member day arithmetic."))
    .orderBy(F.desc("per_diem_paid_that_day"))
    .limit(DETAIL_ROWS).toPandas())

_r6_testable = int(one_per_diem_df["can_be_tested_against_the_published_rule"].sum()) \
    if len(one_per_diem_df) else 0
_mark("Part E rule 6, one per diem per member per day", True)
print(f"Rule 6. Member days where ONE billing entity billed two or more per diem codes: "
      f"{len(one_per_diem_df)}, of which {_r6_testable} rest only on codes with a published unit "
      f"basis and can be tested against the rule.")
print(f"   Measured, never flagged: member days where the per diems came from DIFFERENT billing "
      f"entities: {len(one_per_diem_cross_entity_df)}.")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Rule 7. Levels of care that cannot coexist
# MAGIC
# MAGIC A member cannot be in twenty four hour residential care and attending an intensive outpatient
# MAGIC programme on the same day. This cell builds the matrix of which levels of care can and cannot sit
# MAGIC together, prints it onto its own sheet so every finding can be checked against the rule that
# MAGIC produced it, and then tests the claims against it.
# MAGIC
# MAGIC **Read the cross entity column of the matrix.** Some same day combinations are perfectly proper
# MAGIC when a different entity bills the second service, and treating those as findings is the main false
# MAGIC positive risk in this whole workbook. Rows where a different entity billed are measured and listed,
# MAGIC never flagged. A row resting only on a code whose unit AHCCCS has never published cannot be flagged
# MAGIC either.

# COMMAND ----------

# ── Rule 7. Levels of care that cannot coexist ───────────────────────────────────────────────
#
# The matrix below is printed on the Criteria sheet so every finding can be checked against the
# rule that produced it. A member cannot be in twenty four hour residential care and attending an
# intensive outpatient programme on the same day.
CONFLICT_MATRIX = [
    {"level_a": "residential", "codes_a": ", ".join(sorted(RESIDENTIAL_CODES)),
     "level_b": "day or partial hospitalisation", "codes_b": ", ".join(sorted(DAY_OR_PARTIAL_CODES)),
     "same_entity": "CONFLICT. A member in twenty four hour residential care is not also "
                    "attending a day programme or partial hospitalisation.",
     "cross_entity": "Measured, never flagged here. It is still a conflict on its face, but this "
                     "provider did not bill the other entity's line.",
     "source": "AHCCCS levels of care plus the one per diem per member per day rule",
     "tested_on_its_own_sheet": "yes, E_Level_Of_Care_Conflicts"},
    {"level_a": "residential", "codes_a": ", ".join(sorted(RESIDENTIAL_CODES)),
     "level_b": "intensive outpatient", "codes_b": ", ".join(sorted(INTENSIVE_OUTPATIENT_CODES)),
     "same_entity": "CONFLICT. A member cannot be in twenty four hour residential care and "
                    "attending an intensive outpatient programme the same day.",
     "cross_entity": "Measured, never flagged here, for the same reason.",
     "source": "AHCCCS levels of care plus the one per diem per member per day rule",
     "tested_on_its_own_sheet": "yes, E_Level_Of_Care_Conflicts"},
    {"level_a": "day or partial hospitalisation", "codes_a": ", ".join(sorted(DAY_OR_PARTIAL_CODES)),
     "level_b": "intensive outpatient", "codes_b": ", ".join(sorted(INTENSIVE_OUTPATIENT_CODES)),
     "same_entity": "CONFLICT. Two programmes of care on one day.",
     "cross_entity": "Measured, never flagged here, for the same reason.",
     "source": "AHCCCS levels of care plus the one per diem per member per day rule",
     "tested_on_its_own_sheet": "yes, E_Level_Of_Care_Conflicts"},
    {"level_a": "residential", "codes_a": "H0018, H0019",
     "level_b": "residential", "codes_b": "H0018, H0019",
     "same_entity": "Covered by the one per diem per member per day rule.",
     "cross_entity": "Measured by the one per diem test, never flagged.",
     "source": "Covered Behavioral Health Services Guide 2024-10-01",
     "tested_on_its_own_sheet": "yes, E_One_Per_Diem_Per_Day"},
    {"level_a": "intensive outpatient", "codes_a": "S9480",
     "level_b": "intensive outpatient", "codes_b": "S9480",
     "same_entity": "S9480 is once a day.",
     "cross_entity": "Measured by the one per diem test, never flagged.",
     "source": "Covered Behavioral Health Services Guide",
     "tested_on_its_own_sheet": "yes, E_S9480_Once_A_Day"},
    {"level_a": "residential", "codes_a": "H0018",
     "level_b": "counselling and therapy", "codes_b": "H0004",
     "same_entity": "PROHIBITED. H0004 must not be billed the same day as a residential or "
                    "intensive outpatient per diem by the same entity, and counselling is bundled "
                    "into the H0018 services only per diem.",
     "cross_entity": "PERMITTED. AHCCCS explicitly allows therapy billed by a SEPARATE provider "
                     "on the same day as a residential per diem.",
     "source": "verified same day rules, shared specification section 4",
     "tested_on_its_own_sheet": "yes, E_H0004_With_Per_Diem"},
    {"level_a": "residential", "codes_a": "H0018",
     "level_b": "case management", "codes_b": "T1016",
     "same_entity": "PROHIBITED. Case management by the residential facility is bundled into the "
                    "per diem.",
     "cross_entity": "PERMITTED. Case management may be billed by a provider other than the "
                     "residential facility.",
     "source": "verified same day rules, shared specification section 4",
     "tested_on_its_own_sheet": "yes, E_T1016_With_H0018"},
    {"level_a": "residential", "codes_a": "H0018",
     "level_b": "CPT psychotherapy", "codes_b": ", ".join(sorted(CPT_PSYCHOTHERAPY_CODES)),
     "same_entity": "Bundled into the H0018 services only per diem and not separately billable by "
                    "the facility.",
     "cross_entity": "PERMITTED. CPT psychotherapy codes may be billed by a separate provider on "
                     "the same day as H0018, and specialised counselling by an independent "
                     "licensed behavioural health professional under the professional's own NPI "
                     "is separately billable.",
     "source": "verified same day rules, shared specification section 4",
     "tested_on_its_own_sheet": "NO. Listed here so a reviewer can check a row against it, but "
                                "this notebook does not test it."},
]
conflict_matrix_df = pd.DataFrame(CONFLICT_MATRIX)
CONFLICT_PAIRS = {("day or partial hospitalisation", "intensive outpatient"),
                  ("day or partial hospitalisation", "residential"),
                  ("intensive outpatient", "residential")}

_lvl = df_entity_level.filter(F.col("level_of_care") != F.lit("not a level of care code"))
_lvl_md = (_lvl.groupBy(MEMBER_COL, "srv_date", BPROV_COL, "level_of_care")
           .agg(_entity_list(PROC_COL).alias("codes"),
                F.round(F.sum("qty"), 2).alias("units"),
                F.round(F.sum("paid"), 2).alias("paid"),
                _entity_list(SPROV_COL).alias("servicing_providers"),
                _entity_list("claim_id").alias("claim_ids"),
                # A level that rests ONLY on codes with no published unit cannot carry a finding.
                F.min(F.col("counts_as_a_day_of_care").cast("int")).alias("_min_day"),
                F.max(F.col("counts_as_a_day_of_care").cast("int")).alias("_max_day"))
           .withColumn("rests_only_on_codes_with_no_published_unit", F.col("_max_day") == 0)
           .drop("_min_day", "_max_day"))

_a = _lvl_md.select(
    F.col(MEMBER_COL), F.col("srv_date"),
    F.col(BPROV_COL).alias("entity_a"), F.col("level_of_care").alias("level_a"),
    F.col("codes").alias("codes_a"), F.col("units").alias("units_a"),
    F.col("paid").alias("paid_a"), F.col("servicing_providers").alias("servicing_providers_a"),
    F.col("claim_ids").alias("claim_ids_a"),
    F.col("rests_only_on_codes_with_no_published_unit").alias("unverified_a"))
_b = _lvl_md.select(
    F.col(MEMBER_COL), F.col("srv_date"),
    F.col(BPROV_COL).alias("entity_b"), F.col("level_of_care").alias("level_b"),
    F.col("codes").alias("codes_b"), F.col("units").alias("units_b"),
    F.col("paid").alias("paid_b"), F.col("servicing_providers").alias("servicing_providers_b"),
    F.col("claim_ids").alias("claim_ids_b"),
    F.col("rests_only_on_codes_with_no_published_unit").alias("unverified_b"))

_pairs = (_a.join(_b, [MEMBER_COL, "srv_date"])
          .filter(F.col("level_a") < F.col("level_b"))
          .withColumn("same_billing_entity", F.col("entity_a") == F.col("entity_b"))
          .withColumn("levels_cannot_coexist",
                      F.concat_ws(" | ", F.col("level_a"), F.col("level_b")).isin(
                          sorted(f"{a} | {b}" for a, b in CONFLICT_PAIRS)))
          .filter(F.col("levels_cannot_coexist"))
          .withColumn("rests_on_a_code_with_no_published_unit",
                      F.col("unverified_a") | F.col("unverified_b"))
          .withColumn("paid_on_both_levels", F.round(F.col("paid_a") + F.col("paid_b"), 2))
          .withColumn("involves_a_target_billing_entity",
                      F.col("entity_a").isin(sorted(TARGET_ENTITY_SET)) |
                      F.col("entity_b").isin(sorted(TARGET_ENTITY_SET))
                      if TARGET_ENTITY_SET else F.lit(False)))

level_conflicts_df = (
    _pairs.filter(F.col("same_billing_entity") & F.col("involves_a_target_billing_entity") &
                  ~F.col("rests_on_a_code_with_no_published_unit"))
    .withColumn("rule", F.lit("levels of care that cannot coexist on one member day, from the "
                              "conflict matrix on the Criteria sheet"))
    .withColumn("what_to_pull", F.lit("Pull the claims on both sides for this member on this date "
                                      "and establish which level of care the member was actually "
                                      "in."))
    .drop("unverified_a", "unverified_b", "levels_cannot_coexist")
    .orderBy(F.desc("paid_on_both_levels"))
    .limit(DETAIL_ROWS).toPandas())

level_conflicts_cross_entity_df = (
    _pairs.filter(~F.col("same_billing_entity") & F.col("involves_a_target_billing_entity"))
    .withColumn("why_this_is_not_flagged",
                F.lit("The two levels of care were billed by DIFFERENT entities. Some same day "
                      "combinations are permitted when a different entity bills, and in any case "
                      "this provider did not bill the other entity's line, so this is measured "
                      "and listed rather than flagged. Check it against the conflict matrix on "
                      "the Criteria sheet."))
    .drop("unverified_a", "unverified_b", "levels_cannot_coexist")
    .orderBy(F.desc("paid_on_both_levels"))
    .limit(DETAIL_ROWS).toPandas())

level_conflicts_unverified_n = (
    _pairs.filter(F.col("same_billing_entity") & F.col("involves_a_target_billing_entity") &
                  F.col("rests_on_a_code_with_no_published_unit")).count())
_mark("Part E rule 7, levels of care that cannot coexist", True)
print(f"Rule 7. Member days where one billing entity billed two levels of care that cannot "
      f"coexist: {len(level_conflicts_df)}.")
print(f"   Measured, never flagged: different billing entities: "
      f"{len(level_conflicts_cross_entity_df)}. Resting on a code with no published unit, so the "
      f"rule cannot be applied: {level_conflicts_unverified_n}.")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Rule 8. H2012 above five units in a day
# MAGIC
# MAGIC H2012 is billed by the **hour**, up to five hours a day, even though it sits in the middle of a per
# MAGIC diem focus set. Above five hours the service belongs on H2015, and above ten hours on H2016.
# MAGIC
# MAGIC So more than five units of H2012 in one member day means the service went past the tier it was
# MAGIC billed under. Nothing about this needs judgement: the tiers are published and the arithmetic is a
# MAGIC sum.

# COMMAND ----------

# ── Rule 8. H2012 above 5 units in a day: the wrong tier was used ────────────────────────────
#
# FFS chapter 19 sets the day treatment family as H2012 per hour up to 5 hours, then H2015 per 15
# minutes above 5 hours to 10 hours, then H2016 per diem. More than 5 units of H2012 in one member
# day means the service exceeded the tier it was billed under. Published, clean and needing no
# judgement at all.
_h2012 = (df_line.filter(F.col(PROC_COL) == F.lit("H2012"))
          .groupBy(SPROV_COL, MEMBER_COL, "srv_date")
          .agg(F.round(F.sum("qty"), 2).alias("h2012_units_that_day"),
               F.count(F.lit(1)).alias("lines"),
               F.round(F.sum("paid"), 2).alias("paid"),
               _entity_list(BPROV_COL).alias("billing_entities"),
               _entity_list("claim_id").alias("claim_ids"),
               _entity_list(F.when(F.col("mod_set") != F.lit(""), F.col("mod_set"))).alias("modifier_sets"),
               F.max(F.col("is_group")).alias("any_group_line"),
               F.max("span_days").alias("longest_span_days")))
h2012_tier_df = (
    _h2012.filter(F.col("h2012_units_that_day") > F.lit(H2012_TIER_MAX_UNITS))
    .withColumn("hours_billed_on_H2012_that_day", F.round(F.col("h2012_units_that_day"), 2))
    .withColumn("units_above_the_tier",
                F.round(F.col("h2012_units_that_day") - F.lit(H2012_TIER_MAX_UNITS), 2))
    .withColumn("the_tier_that_fits",
                F.when(F.col("h2012_units_that_day") <= F.lit(10),
                       F.lit("H2015, per 15 minutes, above 5 hours to 10 hours"))
                 .otherwise(F.lit("H2016, per diem, above 10 hours")))
    .withColumn("rule", F.lit("FFS chapter 19 day treatment tiers: H2012 per hour up to 5 hours, "
                              "then H2015 per 15 minutes above 5 hours to 10 hours, then H2016 "
                              "per diem"))
    .withColumn("what_to_pull", F.lit("Pull the claim and the service record for this member day "
                                      "and establish how long the day treatment actually ran."))
    .orderBy(F.desc("h2012_units_that_day"))
    .limit(DETAIL_ROWS).toPandas())
_mark("Part E rule 8, H2012 above the per hour tier", True)
print(f"Rule 8. Member days with more than {H2012_TIER_MAX_UNITS:g} units of H2012: "
      f"{len(h2012_tier_df)}.")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Rule 9. The 2023-07-17 expanded documentation list
# MAGIC
# MAGIC A **second** documentation control, with its own code list and its own thresholds, separate from
# MAGIC the 2023-05-03 one tested earlier. H2012 sits on it in the group needing documentation above two
# MAGIC hourly units.
# MAGIC
# MAGIC Like the earlier one it applies to **claims received** on and after its date, so this test uses the
# MAGIC date the claim arrived and not the date of service. If this table carries no receipt date the test
# MAGIC reports NOT RUN. It does not fall back to the service date, because the claim lag would mis-date it
# MAGIC and a mis-dated documentation test produces a false finding against a real provider.
# MAGIC
# MAGIC A documentation requirement is not a cap. A day above the threshold is a day where documentation
# MAGIC should exist, not a day that was not allowed.

# COMMAND ----------

# ── Rule 9. The 2023-07-17 expanded documentation list, keyed on the CLAIM RECEIPT date ──────
#
# A SECOND documentation control with its own code list, its own thresholds and its own cutoff,
# separate from the 2023-05-03 one in rule 1. H2012 sits on it in the "above 2 hourly units"
# group. Like rule 1 it keys on the date the claim ARRIVED, and like rule 1 there is deliberately
# no fallback to the service date.
EXPANDED_DOC_STATUS = ("run, keyed on the claim receipt date" if RECEIPT_AVAILABLE else
                       "NOT RUN: this table carries no claim receipt date column and the "
                       "published rule applies to claims RECEIVED on or after " +
                       DOC_2023_07_17_CUTOFF + ". There is no fallback to the service date: the "
                       "claim lag would mis-date the test and could produce a false finding.")
_exp_thr_map = F.create_map(*[F.lit(x) for kv in DOC_EXPANDED_THRESHOLDS.items() for x in kv])
if RECEIPT_AVAILABLE:
    _exp = (df_line.filter(F.col(PROC_COL).isin(sorted(DOC_EXPANDED_THRESHOLDS)))
            .withColumn("doc_threshold_units", _exp_thr_map[F.col(PROC_COL)])
            .withColumn("received_on_or_after_cutoff",
                        F.col("recv_date").isNotNull() &
                        (F.col("recv_date") >= F.lit(DOC_2023_07_17_CUTOFF)))
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
                 _entity_list(BPROV_COL).alias("billing_entities"),
                 _entity_list("claim_id").alias("claim_ids")))
    doc_expanded_df = (
        _exp.withColumn("over_threshold_on_lines_received_after_cutoff",
                        F.col("units_on_lines_received_after_cutoff") > F.col("doc_threshold_units"))
            .withColumn("over_threshold_counting_every_line",
                        F.col("units_that_day") > F.col("doc_threshold_units"))
            .filter(F.col("over_threshold_on_lines_received_after_cutoff") |
                    F.col("over_threshold_counting_every_line"))
            .withColumn("what_the_threshold_means",
                        F.when(F.col("doc_threshold_units") == 0,
                               F.lit("documentation required for ANY units of this code"))
                         .when(F.col(PROC_COL).isin(sorted(TIMED_60_CODES)),
                               F.lit("documentation required above 2 HOURLY units"))
                         .otherwise(F.lit("documentation required above this many units a day")))
            .withColumn("is_focus_code", F.col(PROC_COL).isin(FOCUS_CODES))
            .withColumn("keyed_on", F.lit("claim receipt date, which is what the published rule "
                                          "says"))
            .withColumn("receipt_date_source", F.lit(RECEIPT_KEY_COL or RECEIPT_DT_COL))
            .withColumn("rule", F.lit("2023-07-17 expanded documentation thresholds. This is a "
                                      "documentation requirement, not a cap: it is a day where "
                                      "documentation should exist, not a day that was not "
                                      "allowed."))
            .orderBy(F.desc("units_that_day"))
            .limit(DETAIL_ROWS).toPandas())
    _exp_counted = int(doc_expanded_df["over_threshold_on_lines_received_after_cutoff"].sum()) \
        if len(doc_expanded_df) else 0
    _exp_excluded = len(doc_expanded_df) - _exp_counted
    _mark("Part E rule 9, the 2023-07-17 expanded documentation list", True)
    print(f"Rule 9. Member days above the expanded documentation thresholds: {_exp_counted} "
          f"counted (claims received on or after {DOC_2023_07_17_CUTOFF}), {_exp_excluded} listed "
          f"but NOT counted because the claim arrived before the cutoff.")
else:
    doc_expanded_df = pd.DataFrame()
    _exp_counted = _exp_excluded = None
    _mark("Part E rule 9, the 2023-07-17 expanded documentation list", False,
          "no claim receipt date column, and there is deliberately no fallback to the service "
          "date")
    print(f"Rule 9. {EXPANDED_DOC_STATUS}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Rule 10. S9480 once a day
# MAGIC
# MAGIC S9480 is payable once a day. One unit is one day of an intensive outpatient psychiatric programme,
# MAGIC so more than one unit for one member on one day does not fit the description of the code.

# COMMAND ----------

# ── Rule 10. S9480 once a day ────────────────────────────────────────────────────────────────
s9480_once_df = (
    df_line.filter(F.col(PROC_COL) == F.lit("S9480"))
    .groupBy(SPROV_COL, MEMBER_COL, "srv_date")
    .agg(F.round(F.sum("qty"), 2).alias("s9480_units_that_day"),
         F.count(F.lit(1)).alias("lines"),
         F.round(F.sum("paid"), 2).alias("paid"),
         _entity_list(BPROV_COL).alias("billing_entities"),
         _entity_list("claim_id").alias("claim_ids"),
         F.max("span_days").alias("longest_span_days"))
    .filter(F.col("s9480_units_that_day") > F.lit(S9480_UNITS_PER_DAY))
    .withColumn("units_above_the_limit",
                F.round(F.col("s9480_units_that_day") - F.lit(S9480_UNITS_PER_DAY), 2))
    .withColumn("rule", F.lit("S9480 is payable once a day, per the Covered Behavioral Health "
                              "Services Guide. One unit is one day of intensive outpatient "
                              "psychiatric programme."))
    .orderBy(F.desc("s9480_units_that_day"))
    .limit(DETAIL_ROWS).toPandas())
_mark("Part E rule 10, S9480 once a day", True)
print(f"Rule 10. Member days with more than {S9480_UNITS_PER_DAY:g} unit of S9480: "
      f"{len(s9480_once_df)}.")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Rule 11. S9480 prior authorization
# MAGIC
# MAGIC S9480 moved to prior authorization part way through the period, and it moved on **two different
# MAGIC dates**: 2025-07-01 for fee for service (AIHP, DDD-THP, TRBHA and ALTCS) and 2024-10-15 for Arizona
# MAGIC Complete Health managed care. IHS and 638 facilities are **exempt** and are removed before anything
# MAGIC is reported.
# MAGIC
# MAGIC All three things the test needs, the authorization number, the line of business and the IHS
# MAGIC indicators, are looked for by name. Without the authorization number the test does not run at all.
# MAGIC Without the line of business the later date is applied to every line, which reports fewer rows
# MAGIC rather than more. Without the IHS indicators the exempt lines cannot be removed, and every row then
# MAGIC says so.

# COMMAND ----------

# ── Rule 11. S9480 prior authorization ───────────────────────────────────────────────────────
#
# S9480 moved to prior authorization for dates of service on and after 2025-07-01 for fee for
# service (AIHP, DDD-THP, TRBHA and ALTCS), with IHS and 638 facilities EXEMPT. Arizona Complete
# Health required it for managed care from 2024-10-15. So the applicable date depends on the line
# of business, and the exemption depends on the IHS and 638 indicators. All three columns are
# detected rather than assumed.
PA_TEST_STATUS = ("NOT RUN: this table carries no prior authorization number column, so a line "
                  "that carried an authorization cannot be told from one that did not."
                  if not PA_COL else
                  "run" + ("" if LOB_COL else
                           ", BUT with no line_of_business column the later fee for service date "
                           "of " + S9480_PA_FFS_DATE + " was applied to every line, which reports "
                           "fewer findings rather than more")
                  + ("" if (IHS_638_COL or IHS_PROV_COL) else
                     ", AND with no IHS or 638 indicator the exempt lines could NOT be removed, so "
                     "an exempt facility could appear in these findings"))
def _any_token(col, tokens):
    """True when the column contains any of the tokens. Built by folding OR over the tokens
    rather than by taking the maximum of an array of booleans, because an array built from an
    empty token list has no type Spark can order and the whole test would fail at run time."""
    out = F.lit(False)
    for t in tokens:
        out = out | F.col(col).contains(F.lit(t))
    return out


if PA_COL:
    _lob_is_ffs = F.lit(False) if not LOB_COL else _any_token("lob_upper", FFS_LOB_TOKENS)
    _lob_is_mco = F.lit(False) if not LOB_COL else _any_token("lob_upper", MCO_PA_LOB_TOKENS)
    s9480_pa_df = (
        df_line.filter(F.col(PROC_COL) == F.lit("S9480"))
        .withColumn("line_of_business_group",
                    F.when(_lob_is_ffs, F.lit("fee for service"))
                     .when(_lob_is_mco, F.lit("Arizona Complete Health managed care"))
                     .otherwise(F.lit("not recognised, so the fee for service date was applied")))
        .withColumn("prior_authorization_applies_from",
                    F.when(_lob_is_mco & ~_lob_is_ffs, F.lit(S9480_PA_MCO_DATE))
                     .otherwise(F.lit(S9480_PA_FFS_DATE)))
        .withColumn("on_or_after_the_applicable_date",
                    F.col("srv_date") >= F.to_date(F.col("prior_authorization_applies_from")))
        .withColumn("carries_a_prior_authorization", F.col("pa_no").isNotNull())
        .withColumn("exempt_as_IHS_or_638", F.col("is_ihs_or_638"))
        .filter(F.col("on_or_after_the_applicable_date") &
                ~F.col("carries_a_prior_authorization") &
                ~F.col("exempt_as_IHS_or_638"))
        .groupBy(SPROV_COL, BPROV_COL, MEMBER_COL, "srv_date", "line_of_business_group",
                 "prior_authorization_applies_from")
        .agg(F.round(F.sum("qty"), 2).alias("units"),
             F.round(F.sum("paid"), 2).alias("paid"),
             F.count(F.lit(1)).alias("lines"),
             _entity_list("claim_id").alias("claim_ids"),
             _entity_list(F.when(F.col("lob") != F.lit(""), F.col("lob"))).alias("line_of_business"))
        .withColumn("rule", F.lit("S9480 requires prior authorization for dates of service on and "
                                  "after 2025-07-01 for fee for service (AIHP, DDD-THP, TRBHA, "
                                  "ALTCS) and 2024-10-15 for Arizona Complete Health managed "
                                  "care. IHS and 638 facilities are exempt."))
        .withColumn("the_exemption_was_applied",
                    F.lit("yes" if (IHS_638_COL or IHS_PROV_COL) else
                          "NO: this table carries no IHS or 638 indicator, so an exempt facility "
                          "could be on this sheet. Check before acting on a row."))
        .orderBy(F.desc("paid"))
        .limit(DETAIL_ROWS).toPandas())
    _pa_exempt_n = (df_line.filter((F.col(PROC_COL) == F.lit("S9480")) & F.col("is_ihs_or_638"))
                    .count())
    _mark("Part E rule 11, S9480 prior authorization", True)
    print(f"Rule 11. S9480 member days on or after the applicable date with no prior "
          f"authorization: {len(s9480_pa_df)}. IHS or 638 lines, which the rule does not apply "
          f"to: {_pa_exempt_n}.")
    print(f"   {PA_TEST_STATUS}")
else:
    s9480_pa_df = pd.DataFrame()
    _pa_exempt_n = None
    _mark("Part E rule 11, S9480 prior authorization", False,
          "no prior authorization number column on this table")
    print(f"Rule 11. {PA_TEST_STATUS}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Rule 12. The S9480 weekly minimum, and the calendar edges taken out of it
# MAGIC
# MAGIC S9480 has a published minimum of at least two days a week. **A claim can count days**, so this
# MAGIC one can be tested as published and it is a rule. Every row still says in words that it is a
# MAGIC question for the service record rather than a finding, because a claim cannot show the three
# MAGIC hours a day the rule also requires.
# MAGIC
# MAGIC **The calendar edge problem, and why labelling it was not enough.** A programme that starts on
# MAGIC a Friday shows one day in its first week. That is where the calendar falls, not a short
# MAGIC programme. An earlier version of this notebook put a column on the row saying so. That is not
# MAGIC enough: the column does not travel with the row when somebody pastes it into a referral.
# MAGIC
# MAGIC So those weeks are now **excluded by default**. A week is an edge week when either:
# MAGIC
# MAGIC - it is the first or last week of a run of S9480 activity that is **longer than one week**, so
# MAGIC   the programme carries on across the week boundary and part of that stretch of care falls in
# MAGIC   the neighbouring week; or
# MAGIC - it is the first or last week of the whole extract, so the member's record is cut off by the
# MAGIC   lookback rather than by anything the provider did.
# MAGIC
# MAGIC **A run of exactly one week is NOT an edge.** A member who had S9480 on one day and never
# MAGIC again has a complete picture in that week: the day count is not an artefact of where the
# MAGIC calendar falls, it is the whole programme. Excluding those would throw away the clearest
# MAGIC evidence the test exists to find.
# MAGIC
# MAGIC The excluded weeks are kept on a sheet of their own, the number removed is reported, and a
# MAGIC setting puts them back.
# MAGIC
# MAGIC ### H0035 has no rule here, on purpose
# MAGIC
# MAGIC H0035 has a published minimum of twenty **hours** a week. A per diem claim carries no hours at
# MAGIC all, so that minimum cannot be tested. An earlier version turned it into a number of days by
# MAGIC dividing by the longest day of programming it was willing to believe. **That divisor was ours,
# MAGIC not AHCCCS's**, so the threshold was ours too, and a row built on it would have sat on the rule
# MAGIC sheets beside nine published rules.
# MAGIC
# MAGIC It has been removed. There is no H0035 intensity rule, no H0035 finding count and no H0035 row
# MAGIC on the rule summary. What replaced it is the plain distribution of H0035 days per member per
# MAGIC week, in Part I, with the published twenty hours printed beside it as context. The records are
# MAGIC what answer the question; the distribution is the closest claims can get to asking it.

# COMMAND ----------

# ── Rule 12. The S9480 weekly minimum, with the calendar edges taken out ─────────────────────
#
# S9480 has a published minimum of at least 2 DAYS a week, and a claim can count days, so this is
# a rule. It also has a published minimum of 3 hours a day, which a per diem claim cannot show at
# all, so every row says it is a question for the service record rather than a finding.
#
# H0035 has NO rule here. Its published minimum is 20 HOURS a week and a per diem claim carries
# no hours, so turning it into a number of days would need a divisor nobody has published. The
# distribution of H0035 days per member per week is reported in Part I instead, with no threshold.
_FIRST_WEEK_OF_DATA = DATA_START - timedelta(days=DATA_START.weekday()) if DATA_START else None
_LAST_WEEK_OF_DATA = DATA_END - timedelta(days=DATA_END.weekday()) if DATA_END else None


def _weekly_days(code):
    """One row per member per calendar week for a code, with the run of weeks it belongs to and
    whether it sits at an edge of that run or at an edge of the extract itself."""
    w = (df_line.filter(F.col(PROC_COL) == F.lit(code))
         .groupBy(MEMBER_COL, "week_start")
         .agg(F.countDistinct("srv_date").alias("days_billed_that_week"),
              F.round(F.sum("qty"), 2).alias("units"),
              F.round(F.sum("paid"), 2).alias("paid"),
              F.min("srv_date").alias("first_day_that_week"),
              F.max("srv_date").alias("last_day_that_week"),
              _entity_list(BPROV_COL).alias("billing_entities"),
              _entity_list(SPROV_COL).alias("servicing_providers"),
              _entity_list("claim_id").alias("claim_ids")))
    # Runs of consecutive weeks. A gap of more than one week ends the run.
    _ww = Window.partitionBy(MEMBER_COL).orderBy("week_start")
    w = (w.withColumn("_prev_week", F.lag("week_start").over(_ww))
         .withColumn("_new_run", F.when(F.col("_prev_week").isNull() |
                                        (F.datediff("week_start", "_prev_week") > F.lit(7)),
                                        F.lit(1)).otherwise(F.lit(0)))
         .withColumn("_run", F.sum("_new_run").over(_ww.rowsBetween(Window.unboundedPreceding, 0))))
    _runs = (w.groupBy(MEMBER_COL, "_run")
             .agg(F.min("week_start").alias("run_starts"),
                  F.max("week_start").alias("run_ends"),
                  F.count(F.lit(1)).alias("weeks_in_this_run")))
    w = (w.join(_runs, [MEMBER_COL, "_run"], "left")
         .withColumn("the_member_had_this_code_in_an_adjacent_week",
                     F.col("weeks_in_this_run") > F.lit(1))
         # The edge of a run that CARRIES ON past this week. A run of exactly one week is not an
         # edge: that week holds the whole of this stretch of care, so its day count is complete
         # rather than an artefact of where the calendar falls.
         .withColumn("at_the_edge_of_a_run_that_carries_on",
                     (F.col("weeks_in_this_run") > F.lit(1)) &
                     ((F.col("week_start") == F.col("run_starts")) |
                      (F.col("week_start") == F.col("run_ends"))))
         # The edge of the EXTRACT. The record is cut off by the lookback here, not by anything
         # the provider did.
         .withColumn("at_the_edge_of_the_extract",
                     (F.col("week_start") <= F.lit(_FIRST_WEEK_OF_DATA)) |
                     (F.col("week_start") >= F.lit(_LAST_WEEK_OF_DATA))
                     if _FIRST_WEEK_OF_DATA else F.lit(False))
         .withColumn("this_week_is_a_calendar_edge",
                     F.col("at_the_edge_of_a_run_that_carries_on") |
                     F.col("at_the_edge_of_the_extract"))
         .withColumn("why_it_is_an_edge",
                     F.when(F.col("at_the_edge_of_a_run_that_carries_on") &
                            F.col("at_the_edge_of_the_extract"),
                            F.lit("the programme carries on past this week AND the extract ends "
                                  "here"))
                      .when(F.col("at_the_edge_of_a_run_that_carries_on"),
                            F.lit("the programme carries on into the week before or after, so "
                                  "part of this stretch of care falls in the neighbouring week"))
                      .when(F.col("at_the_edge_of_the_extract"),
                            F.lit("this is the first or last week of the extract, so the "
                                  "member's record is cut off by the lookback"))
                      .otherwise(F.lit("")))
         .drop("_prev_week", "_new_run", "_run"))
    return w


_s9480_weeks = _weekly_days("S9480")
_s9480_short = (
    _s9480_weeks
    .filter(F.col("days_billed_that_week") < F.lit(S9480_MIN_DAYS_PER_WEEK))
    .withColumn("published_minimum",
                F.lit(f"S9480 requires at least 3 hours a day and at least "
                      f"{S9480_MIN_DAYS_PER_WEEK} days a week"))
    .withColumn("this_row_is",
                F.lit("A QUESTION, not a finding. A per diem claim carries no hours, so this "
                      "cannot show that the programme ran for less than three hours. It shows "
                      "that the programme was billed on fewer days in this calendar week than the "
                      "published weekly minimum, which is a question for the service record."))
    .withColumn("what_to_pull", F.lit("Pull the programme attendance record for this member for "
                                      "this week."))
    .cache())

_keep_edges = F.lit(True) if S9480_INCLUDE_EDGE_WEEKS else ~F.col("this_week_is_a_calendar_edge")
s9480_intensity_df = (_s9480_short.filter(_keep_edges)
                      .withColumn("calendar_edge_weeks",
                                  F.lit("INCLUDED by setting" if S9480_INCLUDE_EDGE_WEEKS
                                        else "EXCLUDED: see E_Min_Intensity_Edges"))
                      .orderBy(F.desc("paid")).limit(DETAIL_ROWS).toPandas())
s9480_intensity_edge_df = (
    _s9480_short.filter(F.col("this_week_is_a_calendar_edge"))
    .withColumn("why_this_is_not_on_the_rule_sheet",
                F.lit("This week has fewer S9480 days than the published weekly minimum, but the "
                      "shortfall is where the calendar falls rather than a short programme. It is "
                      "kept here so nothing is hidden and it is NOT counted as a finding. Set "
                      "s9480_include_edge_weeks to true to put these rows back on the rule "
                      "sheet."))
    .orderBy(F.desc("paid")).limit(DETAIL_ROWS).toPandas())
S9480_EDGE_WEEKS_REMOVED = 0 if S9480_INCLUDE_EDGE_WEEKS else len(s9480_intensity_edge_df)

# ── The H0035 days per week DISTRIBUTION, which is not a rule and has no threshold ───────────
_h0035_weeks = _weekly_days("H0035")
h0035_days_per_week_df = (
    _h0035_weeks.groupBy("days_billed_that_week")
    .agg(F.count(F.lit(1)).alias("member_weeks"),
         F.countDistinct(MEMBER_COL).alias("members"),
         F.round(F.sum("units"), 2).alias("units"),
         F.round(F.sum("paid"), 2).alias("paid"),
         F.sum(F.col("this_week_is_a_calendar_edge").cast("int"))
          .alias("member_weeks_sitting_at_a_calendar_edge"))
    .orderBy("days_billed_that_week").toPandas())
if len(h0035_days_per_week_df):
    _tw = int(h0035_days_per_week_df["member_weeks"].sum())
    h0035_days_per_week_df["share_of_member_weeks"] = [
        _rate(v, _tw) for v in h0035_days_per_week_df.member_weeks]
    h0035_days_per_week_df["how_to_read_this"] = (
        f"H0035 has a published minimum of {H0035_MIN_HOURS_PER_WEEK:g} HOURS a week. A per diem "
        f"claim carries no hours, so this notebook cannot test that minimum and does not try. "
        f"There is no threshold on this sheet, no flag and no finding count. What the claims can "
        f"show is how many days in a calendar week the programme was billed, which is this "
        f"distribution. The service records are what answer whether the twenty hours were met. "
        f"Note the calendar edge column: a week at the start or end of a run of activity shows "
        f"fewer days because of where the calendar falls, not because the programme was short.")

_mark("Part E rule 12, S9480 weekly minimum", True)
_mark("Part I, H0035 days per week distribution", True)
print(f"Rule 12. Member weeks with S9480 on fewer than {S9480_MIN_DAYS_PER_WEEK} days: "
      f"{len(s9480_intensity_df)} (questions, not findings). Calendar edge weeks "
      f"{'INCLUDED' if S9480_INCLUDE_EDGE_WEEKS else 'EXCLUDED'}: "
      f"{S9480_EDGE_WEEKS_REMOVED} rows were removed and are on their own sheet.")
print(f"H0035 days per week: a DISTRIBUTION, not a rule. "
      f"{len(h0035_days_per_week_df)} distinct day counts across "
      f"{int(h0035_days_per_week_df['member_weeks'].sum()) if len(h0035_days_per_week_df) else 0} "
      f"member weeks. No threshold is applied and nothing on it is a finding.")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Rule 14. H0018 place of service
# MAGIC
# MAGIC Behavioural health residential facilities must bill place of service 56 from 2023-12-01. This lists
# MAGIC the H0018 lines on or after that date carrying something else.
# MAGIC
# MAGIC Place of service values are compared with leading zeros removed, so 056 and 56 are the same place
# MAGIC and a formatting difference in the extract cannot manufacture a finding against a facility that
# MAGIC billed correctly.

# COMMAND ----------

# ── Rule 14. H0018 place of service ──────────────────────────────────────────────────────────
POS_TEST_STATUS = ("NOT RUN: this table carries no place of service column, so no H0018 line can "
                   "be checked against the requirement." if not POS_CODE_COL else "run")
if POS_CODE_COL:
    h0018_pos_df = (
        df_line.filter((F.col(PROC_COL) == F.lit("H0018")) &
                       (F.col("srv_date") >= F.to_date(F.lit(POS_56_DATE))) &
                       (F.col("pos_code_clean") != F.lit(str(int(BHRF_POS_CODE)))))
        .groupBy(SPROV_COL, BPROV_COL, MEMBER_COL, "srv_date", "pos_code", "pos_desc")
        .agg(F.round(F.sum("qty"), 2).alias("units"),
             F.round(F.sum("paid"), 2).alias("paid"),
             F.count(F.lit(1)).alias("lines"),
             _entity_list("claim_id").alias("claim_ids"))
        .withColumn("place_of_service_required", F.lit(BHRF_POS_CODE))
        .withColumn("rule", F.lit(f"Behavioural health residential facilities must bill place of "
                                  f"service {BHRF_POS_CODE} from {POS_56_DATE}."))
        .withColumn("what_to_pull", F.lit("Pull the claim and confirm the setting the service was "
                                          "delivered in."))
        .orderBy(F.desc("paid"))
        .limit(DETAIL_ROWS).toPandas())
    _mark("Part E rule 14, H0018 place of service 56", True)
    print(f"Rule 14. H0018 member days on or after {POS_56_DATE} with a place of service other "
          f"than {BHRF_POS_CODE}: {len(h0018_pos_df)}.")
else:
    h0018_pos_df = pd.DataFrame()
    _mark("Part E rule 14, H0018 place of service 56", False, "no place of service column")
    print(f"Rule 14. {POS_TEST_STATUS}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Rule 15. Provider type against the codes billed
# MAGIC
# MAGIC H0018 is the behavioural health residential facility per diem, and provider type B8 is the provider
# MAGIC type that bills it. If the provider under review is a different type, that is the first thing to
# MAGIC check and it belongs at the top of a report rather than in a column halfway down a sheet.
# MAGIC
# MAGIC **This notebook does not assert that another provider type may not bill the code.** The AHCCCS code
# MAGIC to provider type matrix settles that. What this cell does is produce a single plain statement of
# MAGIC what this provider type billed, which goes onto its own sheet, onto the Criteria sheet, into the
# MAGIC email and into the printout below.

# COMMAND ----------

# ── Rule 15. Provider type against the codes this provider actually billed ───────────────────
#
# This is the shape of finding that was the headline of the previous run, and it belongs near the
# top of any report rather than buried in a column. H0018 is the behavioural health residential
# facility per diem and provider type B8 is the type that bills it. This notebook does NOT assert
# that another type is not allowed to: the AHCCCS code to provider type matrix settles that, and
# checking it is the first thing to do with this row.
CODE_TO_PROVIDER_TYPE = {
    "H0018": "the behavioural health residential facility per diem; provider type B8 "
             "(Behavioral Health Residential Facility) is the type that bills it, and type B8 "
             "bills essentially nothing else",
    "H0019": "long term residential. Its unit basis is not published by AHCCCS and it is absent "
             "from the billing manual, so which provider types may bill it is a question for the "
             "code to provider type matrix as well",
}
_ptype_rows = []
for pid in TARGETS:
    _pt_code = PROFILES[pid].get("provider_type_code") or ""
    _pt_desc = PROFILES[pid].get("provider_type_description") or ""
    _tbl = ALL_CODE_TABLES.get(pid, pd.DataFrame())
    for c in FOCUS_CODES:
        _m = _tbl[_tbl[PROC_COL] == c] if len(_tbl) else pd.DataFrame()
        _paid = float(_m.iloc[0]["paid"]) if len(_m) else 0.0
        _lines = int(_m.iloc[0]["lines"]) if len(_m) else 0
        _mem = int(_m.iloc[0]["members"]) if len(_m) else 0
        _ptype_rows.append({
            "provider_id": pid,
            "provider_type_code": _pt_code,
            "provider_type_description": _pt_desc,
            "person_or_organisation": PROFILES[pid].get("person_or_organisation", "unknown"),
            "procedure_code": c,
            "unit_basis": CODE_BASIS.get(c, "UNCLASSIFIED"),
            "lines": _lines, "members": _mem, "paid": round(_paid, 2),
            "billed_by_this_provider": "yes" if _lines else "no",
            "check_this_one_first": "YES" if (_lines and c in CODE_TO_PROVIDER_TYPE) else "",
            "what_the_code_is": CODE_TO_PROVIDER_TYPE.get(
                c, FOCUS_CODE_NOTE.get(c, "see the Criteria sheet for the unit basis")),
            "how_to_settle_it": ("The AHCCCS code to provider type matrix settles whether this "
                                 "provider type may bill this code. This notebook does not assert "
                                 "that it may not. Check the matrix before anything else."
                                 if _lines and c in CODE_TO_PROVIDER_TYPE else ""),
        })
provider_type_check_df = pd.DataFrame(_ptype_rows)
_flagged_codes = sorted({r["procedure_code"] for r in _ptype_rows if r["check_this_one_first"]})
_billed_codes = sorted({r["procedure_code"] for r in _ptype_rows
                        if r["billed_by_this_provider"] == "yes"})
PROVIDER_TYPE_STATEMENT = (
    "; ".join(
        f"Provider {p} is {PROFILES[p].get('provider_type_description') or 'of an unknown type'}"
        + (f" (type code {PROFILES[p].get('provider_type_code')})"
           if PROFILES[p].get('provider_type_code') else "")
        + ", "
        + (("and billed these focus codes: "
            + ", ".join(sorted({r['procedure_code'] for r in _ptype_rows
                                if r['provider_id'] == p and r['billed_by_this_provider'] == 'yes'})))
           if any(r['provider_id'] == p and r['billed_by_this_provider'] == 'yes'
                  for r in _ptype_rows)
           else "and billed none of the focus codes")
        for p in TARGETS)
    + (". CHECK FIRST: "
       + ", ".join(_flagged_codes)
       + " "
       + ("is" if len(_flagged_codes) == 1 else "are")
       + " a residential per diem. Provider type B8 (Behavioral Health Residential Facility) is "
         "the provider type that bills H0018. This notebook does NOT assert that this provider "
         "type may not bill it: the AHCCCS code to provider type matrix settles that, and it is "
         "the first thing to check."
       if _flagged_codes else
       ". No residential per diem code was billed by these providers, so there is nothing for the "
       "code to provider type matrix to settle here."))
_mark("Part E rule 15, provider type against the codes billed", True)
print("\n" + "=" * 78)
print("PROVIDER TYPE AGAINST THE CODES BILLED")
print("=" * 78)
print(PROVIDER_TYPE_STATEMENT)
print("=" * 78 + "\n")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Rule 16. Bed hold and therapeutic leave
# MAGIC
# MAGIC Revenue code 0183 is home pass, which is therapeutic leave, and 0189 is bed hold. The limits run
# MAGIC per contract year: for members 21 and over, nine days of therapeutic leave and twelve days of bed
# MAGIC hold; for members under 21, twenty one days of the two together.
# MAGIC
# MAGIC Both need a revenue code column and the age split needs a date of birth. Each is looked for by
# MAGIC name. Without the revenue code there is no test at all. Without the date of birth the days are
# MAGIC counted and reported with no limit applied and nothing is flagged, because the two limits are
# MAGIC different and guessing which one applies to a member is how a false finding gets made.
# MAGIC
# MAGIC A member who turned 21 inside a contract year sits between the two limits. Those rows are flagged
# MAGIC only if they exceed both, and they say so.

# COMMAND ----------

# ── Rule 16. Bed hold and therapeutic leave ──────────────────────────────────────────────────
#
# Revenue code 0183 is home pass, which is therapeutic leave, and 0189 is bed hold. The published
# limits are per CONTRACT YEAR: for members 21 and over, 9 days of therapeutic leave and 12 days
# of bed hold; for members under 21, 21 days of the two combined. Both need a revenue code column,
# and the age split needs a date of birth. Each is detected, and when one is missing the test says
# so rather than producing a number that looks right.
LEAVE_TEST_STATUS = ("NOT RUN: this table carries no revenue code column. Therapeutic leave is "
                     "revenue code " + REV_HOME_PASS + " and bed hold is " + REV_BED_HOLD + ", and "
                     "neither can be seen without one." if not REV_CODE_COL else
                     "run" + ("" if DOB_COL else
                              ", BUT with no date of birth the 21 and over limits cannot be told "
                              "from the under 21 limit, so the day counts are reported with NO "
                              "limit applied and nothing is flagged"))
if REV_CODE_COL:
    _rev = (df_line
            .withColumn("_rev4", F.lpad(F.coalesce(F.col("rev_code"), F.lit("")), 4, "0"))
            .filter(F.col("_rev4").isin([REV_HOME_PASS, REV_BED_HOLD]))
            .withColumn("leave_kind", F.when(F.col("_rev4") == F.lit(REV_HOME_PASS),
                                             F.lit("therapeutic leave (home pass)"))
                                       .otherwise(F.lit("bed hold")))
            .withColumn("contract_year_start",
                        F.when(F.month("srv_date") >= F.lit(CONTRACT_YEAR_START_MONTH),
                               F.make_date(F.year("srv_date"), F.lit(CONTRACT_YEAR_START_MONTH),
                                           F.lit(1)))
                         .otherwise(F.make_date(F.year("srv_date") - F.lit(1),
                                                F.lit(CONTRACT_YEAR_START_MONTH), F.lit(1)))))
    bed_hold_df = (
        _rev.groupBy(SPROV_COL, BPROV_COL, MEMBER_COL, "contract_year_start")
        .agg(F.countDistinct(F.when(F.col("leave_kind").startswith("therapeutic"),
                                    F.col("srv_date"))).alias("therapeutic_leave_days"),
             F.countDistinct(F.when(F.col("leave_kind") == F.lit("bed hold"),
                                    F.col("srv_date"))).alias("bed_hold_days"),
             F.countDistinct("srv_date").alias("leave_or_bed_hold_days_combined"),
             F.round(F.sum("paid"), 2).alias("paid"),
             F.count(F.lit(1)).alias("lines"),
             F.min("member_age").alias("youngest_age_in_the_contract_year"),
             F.max("member_age").alias("oldest_age_in_the_contract_year"),
             _entity_list("claim_id").alias("claim_ids"),
             _entity_list("rev_code").alias("revenue_codes"))
        .withColumn("contract_year",
                    F.concat_ws(" to ", F.col("contract_year_start"),
                                F.date_add(F.add_months("contract_year_start", 12), -1)))
        .withColumn("age_group",
                    F.when(F.col("oldest_age_in_the_contract_year").isNull(),
                           F.lit("unknown: no date of birth on this table"))
                     .when(F.col("oldest_age_in_the_contract_year") < F.lit(21),
                           F.lit("under 21"))
                     .when(F.col("youngest_age_in_the_contract_year") >= F.lit(21),
                           F.lit("21 and over"))
                     .otherwise(F.lit("turned 21 during the contract year")))
        .withColumn("limit_that_applies",
                    F.when(F.col("age_group") == F.lit("under 21"),
                           F.lit(f"{COMBINED_LIMIT_UNDER_21} days of leave and bed hold combined"))
                     .when(F.col("age_group") == F.lit("21 and over"),
                           F.lit(f"{LEAVE_LIMIT_21_PLUS} days therapeutic leave and "
                                 f"{BED_HOLD_LIMIT_21_PLUS} days bed hold"))
                     .when(F.col("age_group") == F.lit("turned 21 during the contract year"),
                           F.lit("the member turned 21 inside this contract year, so both limits "
                                 "touch it. This row is flagged only if it exceeds BOTH, and it "
                                 "must be checked by hand either way."))
                     .otherwise(F.lit("NONE APPLIED: no date of birth on this table")))
        .withColumn("_over_21_plus", (F.col("therapeutic_leave_days") > F.lit(LEAVE_LIMIT_21_PLUS)) |
                                     (F.col("bed_hold_days") > F.lit(BED_HOLD_LIMIT_21_PLUS)))
        .withColumn("_over_under_21",
                    F.col("leave_or_bed_hold_days_combined") > F.lit(COMBINED_LIMIT_UNDER_21))
        .withColumn("over_the_limit",
                    F.when(F.col("age_group") == F.lit("under 21"), F.col("_over_under_21"))
                     .when(F.col("age_group") == F.lit("21 and over"), F.col("_over_21_plus"))
                     .when(F.col("age_group") == F.lit("turned 21 during the contract year"),
                           F.col("_over_21_plus") & F.col("_over_under_21"))
                     .otherwise(F.lit(False)))
        .withColumn("rule", F.lit(f"Members 21 and over: {LEAVE_LIMIT_21_PLUS} days therapeutic "
                                  f"leave and {BED_HOLD_LIMIT_21_PLUS} days bed hold per contract "
                                  f"year. Under 21: {COMBINED_LIMIT_UNDER_21} days combined. The "
                                  f"contract year used here starts on 1 "
                                  f"{date(2000, CONTRACT_YEAR_START_MONTH, 1).strftime('%B')}."))
        .drop("_over_21_plus", "_over_under_21")
        .orderBy(F.desc("leave_or_bed_hold_days_combined"))
        .limit(DETAIL_ROWS).toPandas())
    bed_hold_over_n = int(bed_hold_df["over_the_limit"].sum()) if len(bed_hold_df) else 0
    _mark("Part E rule 16, bed hold and therapeutic leave", True)
    print(f"Rule 16. Member contract years with therapeutic leave or bed hold days: "
          f"{len(bed_hold_df)}, of which {bed_hold_over_n} are above the limit that applies.")
    if not DOB_COL:
        print(f"   {LEAVE_TEST_STATUS}")
else:
    bed_hold_df = pd.DataFrame()
    bed_hold_over_n = None
    _mark("Part E rule 16, bed hold and therapeutic leave", False,
          "no revenue code column, so therapeutic leave and bed hold lines cannot be identified")
    print(f"Rule 16. {LEAVE_TEST_STATUS}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 16. Part E continued. The front door codes and their published limits
# MAGIC
# MAGIC Four more tests, on the assessment, screening and education codes. These are a different kind
# MAGIC of service from the levels of care above and they carry a different kind of rule.
# MAGIC
# MAGIC | # | Test | Where the rule comes from | Keyed on |
# MAGIC |---|---|---|---|
# MAGIC | 17 | 98960 above 4 units in a day | A hard unit limit effective 2023-04-01 | date of service |
# MAGIC | 18 | 98960 above 60 units in any rolling 30 days | The same control | date of service |
# MAGIC | 19 | H0001 above 12 a year | Covered Behavioral Health Services Guide, effective 2024-10-01 | date of service |
# MAGIC | 20 | H0002 above 24 a year, members aged 11 and over | The same guide | date of service, plus date of birth |
# MAGIC
# MAGIC **Tests 17 and 18 are exact and need no view about what one unit means.** Both published
# MAGIC limits are counts of units. The 30 minutes attached to 98960 comes from the national HCPCS
# MAGIC descriptor rather than from an AHCCCS statement, and it is used only to put the code into the
# MAGIC day hours total. Neither of these tests touches it.
# MAGIC
# MAGIC **Test 18 is a rolling window, and the machinery behind it is general.** A member can stay
# MAGIC under four units every single day and still be far above sixty units in a month, so a daily
# MAGIC cap and a rolling cap catch different things and the rolling one has to be computed its own
# MAGIC way. Several other AHCCCS limits take exactly this shape, including the 600 hours a benefit
# MAGIC year on respite and the 160 units a week attached to the U3 modifier, so this is written as a
# MAGIC reusable function rather than as a one off for 98960.
# MAGIC
# MAGIC **Tests 19 and 20 report two answers each, on purpose.** The guide says "per year" and does
# MAGIC not say which year. The shared specification records a benefit year of 1 October to 30
# MAGIC September for respite hours, so that is the likely basis, but it is not stated for these codes
# MAGIC and this notebook will not guess. Both the calendar year and the benefit year are counted and
# MAGIC shown side by side. A member who is above the limit on one basis and below it on the other is
# MAGIC marked **AMBIGUOUS**, because that is what the member is: thirteen assessments in a calendar
# MAGIC year and eleven in a benefit year is a question about which year AHCCCS meant, not a finding.

# COMMAND ----------

# ── A general rolling window unit limit ──────────────────────────────────────────────────────
#
# Written generally because this is a shape, not a one off: 98960 at 60 units in 30 days today,
# and respite at 600 hours a benefit year or the U3 modifier at 160 units a week tomorrow.
#
# The window is computed with a RANGE frame over the day number rather than a row frame, because
# a row frame would count the last N billed days rather than the last N calendar days and would
# quietly let a member billed on 40 scattered days inside a month slip under a 30 day window.
def rolling_unit_limit(src, codes, window_days, limit, effective_date, rule_text):
    """Units of a family of codes inside any rolling window of calendar days, per member.
    Returns one row per member who crossed the limit, carrying the peak window so the row is an
    instruction to pull a record rather than one row per overlapping window."""
    d = (src.filter(F.col(PROC_COL).isin(sorted(codes)))
         .filter(F.col("srv_date") >= F.to_date(F.lit(effective_date)))
         .groupBy(MEMBER_COL, "srv_date")
         .agg(F.round(F.sum("qty"), 2).alias("units_that_day"),
              F.round(F.sum("paid"), 2).alias("paid_that_day"),
              F.count(F.lit(1)).alias("lines_that_day"),
              _entity_list(PROC_COL).alias("codes_that_day"),
              _entity_list(BPROV_COL).alias("billing_entities_that_day"),
              _entity_list(SPROV_COL).alias("servicing_providers_that_day"),
              _entity_list("claim_id").alias("claim_ids_that_day"))
         .withColumn("_day_num", F.datediff("srv_date", F.lit("1970-01-01")).cast("long")))
    _w = (Window.partitionBy(MEMBER_COL).orderBy(F.col("_day_num"))
          .rangeBetween(-(window_days - 1), 0))
    d = (d.withColumn("units_in_the_window", F.round(F.sum("units_that_day").over(_w), 2))
         .withColumn("paid_in_the_window", F.round(F.sum("paid_that_day").over(_w), 2))
         .withColumn("billed_days_in_the_window", F.count(F.lit(1)).over(_w))
         .withColumn("window_ends", F.col("srv_date"))
         .withColumn("window_starts", F.date_sub(F.col("srv_date"), window_days - 1))
         .withColumn("over_the_limit", F.col("units_in_the_window") > F.lit(limit)))
    # The peak window for each member, so one member is one row.
    _rank = Window.partitionBy(MEMBER_COL).orderBy(F.desc("units_in_the_window"),
                                                   F.desc("window_ends"))
    peak = (d.withColumn("_rk", F.row_number().over(_rank)).filter(F.col("_rk") == 1)
            .select(MEMBER_COL,
                    F.col("units_in_the_window").alias("units_in_the_peak_window"),
                    F.col("paid_in_the_window").alias("paid_in_the_peak_window"),
                    F.col("billed_days_in_the_window").alias("billed_days_in_the_peak_window"),
                    F.col("window_starts").alias("peak_window_starts"),
                    F.col("window_ends").alias("peak_window_ends")))
    totals = (d.groupBy(MEMBER_COL)
              .agg(F.sum(F.col("over_the_limit").cast("int")).alias("windows_over_the_limit"),
                   F.min(F.when(F.col("over_the_limit"), F.col("window_ends")))
                    .alias("first_window_over_the_limit_ends"),
                   F.max(F.when(F.col("over_the_limit"), F.col("window_ends")))
                    .alias("last_window_over_the_limit_ends"),
                   F.round(F.sum("units_that_day"), 2).alias("units_altogether"),
                   F.round(F.sum("paid_that_day"), 2).alias("paid_altogether"),
                   F.countDistinct("srv_date").alias("days_billed_altogether"),
                   F.max("units_that_day").alias("most_units_on_any_single_day"),
                   _entity_list("billing_entities_that_day").alias("billing_entities"),
                   _entity_list("servicing_providers_that_day").alias("servicing_providers"),
                   _entity_list("claim_ids_that_day").alias("claim_ids")))
    return (totals.filter(F.col("windows_over_the_limit") > F.lit(0))
            .join(peak, MEMBER_COL, "left")
            .withColumn("the_limit", F.lit(limit))
            .withColumn("the_window_in_days", F.lit(window_days))
            .withColumn("rule", F.lit(rule_text))
            .withColumn("what_to_pull",
                        F.lit("Pull every claim named in claim_ids for this member between the "
                              "peak window start and end dates."))
            .orderBy(F.desc("units_in_the_peak_window")))


# ── Rule 17. 98960 above the daily unit cap ──────────────────────────────────────────────────
#
# A HARD CAP, not a documentation threshold, so a day above it is a day that exceeded a published
# limit rather than a day where paperwork should exist. It keys on the date of SERVICE.
_edu_day = (df_line.filter(F.col(PROC_COL).isin(sorted(EDUCATION_CODES)))
            .filter(F.col("srv_date") >= F.to_date(F.lit(EDU_LIMIT_DATE)))
            .groupBy(SPROV_COL, MEMBER_COL, "srv_date", PROC_COL)
            .agg(F.round(F.sum("qty"), 2).alias("units_that_day"),
                 F.count(F.lit(1)).alias("lines"),
                 F.round(F.sum("paid"), 2).alias("paid"),
                 _entity_list(BPROV_COL).alias("billing_entities"),
                 _entity_list("claim_id").alias("claim_ids"),
                 F.max("span_days").alias("longest_span_days")))
# Context only: what the whole 98960 family came to that day. The cap is applied per code,
# which is the reading that produces fewer findings, and this column shows the alternative.
_edu_family_day = (df_line.filter(F.col(PROC_COL).isin(sorted(EDUCATION_CODES)))
                   .groupBy(MEMBER_COL, "srv_date")
                   .agg(F.round(F.sum("qty"), 2).alias("units_across_the_whole_family_that_day"),
                        _entity_list(PROC_COL).alias("family_codes_that_day")))
edu_daily_cap_df = (
    _edu_day.filter(F.col("units_that_day") > F.lit(EDU_DAILY_UNIT_CAP))
    .join(_edu_family_day, [MEMBER_COL, "srv_date"], "left")
    .withColumn("the_daily_cap", F.lit(EDU_DAILY_UNIT_CAP))
    .withColumn("units_above_the_cap",
                F.round(F.col("units_that_day") - F.lit(EDU_DAILY_UNIT_CAP), 2))
    .withColumn("rule", F.lit(f"{', '.join(sorted(EDUCATION_CODES))} are limited to "
                              f"{EDU_DAILY_UNIT_CAP:g} units a day and "
                              f"{EDU_ROLLING_UNIT_CAP:g} units in any "
                              f"{EDU_ROLLING_WINDOW_DAYS} days, effective {EDU_LIMIT_DATE}. This "
                              f"is a hard limit, not a documentation requirement, and it keys on "
                              f"the date of SERVICE."))
    .withColumn("the_cap_was_applied_per_code",
                F.lit("yes. The published control names the code range rather than saying whether "
                      "the cap is per code or across the family, so it is applied per code, which "
                      "reports fewer rows. The family total for the same day is shown beside it."))
    .orderBy(F.desc("units_that_day"))
    .limit(DETAIL_ROWS).toPandas())
_mark("Part E rule 17, 98960 above the daily unit cap", True)
print(f"Rule 17. Member days above {EDU_DAILY_UNIT_CAP:g} units of an education and training "
      f"code: {len(edu_daily_cap_df)}.")

# ── Rule 18. 98960 above the rolling window unit cap ─────────────────────────────────────────
edu_rolling_df = (
    rolling_unit_limit(
        df_line, EDUCATION_CODES, EDU_ROLLING_WINDOW_DAYS, EDU_ROLLING_UNIT_CAP, EDU_LIMIT_DATE,
        f"{', '.join(sorted(EDUCATION_CODES))} are limited to {EDU_ROLLING_UNIT_CAP:g} units in "
        f"any {EDU_ROLLING_WINDOW_DAYS} days, effective {EDU_LIMIT_DATE}. A member can stay under "
        f"the daily cap every single day and still cross this one, so the two tests catch "
        f"different things.")
    .limit(DETAIL_ROWS).toPandas())
_mark("Part E rule 18, 98960 above the rolling window unit cap", True)
print(f"Rule 18. Members above {EDU_ROLLING_UNIT_CAP:g} units of an education and training code "
      f"in any rolling {EDU_ROLLING_WINDOW_DAYS} days: {len(edu_rolling_df)}.")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Rules 19 and 20. Two limits, and two possible years
# MAGIC
# MAGIC H0001 is limited to twelve a year, and H0002 to twenty four a year for members aged eleven and
# MAGIC over. **AHCCCS does not say which year.** The shared specification records a benefit year running
# MAGIC 1 October to 30 September for respite hours, so that is the likely basis, but nobody has said it
# MAGIC for these codes.
# MAGIC
# MAGIC So this notebook does not pick one. It counts **both** the calendar year and the benefit year and
# MAGIC puts them side by side. A member above the limit on one basis and below it on the other is marked
# MAGIC AMBIGUOUS, because that is exactly what the member is: thirteen assessments in a calendar year and
# MAGIC eleven in a benefit year is a question about which year AHCCCS meant, not a finding about the
# MAGIC provider.
# MAGIC
# MAGIC The H0002 limit applies only to members aged eleven and over, so only the units billed while the
# MAGIC member was that age count towards it. That needs a date of birth, and without one the test reports
# MAGIC NOT RUN.

# COMMAND ----------

# ── Rules 19 and 20. Per year limits whose YEAR BASIS AHCCCS HAS NOT STATED ──────────────────
#
# Counted BOTH ways and reported side by side. A member above the limit on one basis and below it
# on the other is marked AMBIGUOUS, because that is exactly what the member is.
def _per_year_limit(code, limit, min_age=None):
    d = df_line.filter(F.col(PROC_COL) == F.lit(code))
    if min_age is not None:
        # The limit is written for members aged 11 and over, so only the units billed while the
        # member was that age can count towards it. A member who was younger for every line has
        # nothing to compare against the limit at all.
        d = d.withColumn("_counts_towards_the_limit",
                         F.col("member_age").isNotNull() & (F.col("member_age") >= F.lit(min_age)))
    else:
        d = d.withColumn("_counts_towards_the_limit", F.lit(True))
    d = (d.withColumn("_q", F.when(F.col("_counts_towards_the_limit"), F.col("qty"))
                       .otherwise(F.lit(0.0)))
         .withColumn("calendar_year", F.year("srv_date"))
         .withColumn("benefit_year_start",
                     F.when(F.month("srv_date") >= F.lit(BENEFIT_YEAR_START_MONTH),
                            F.make_date(F.year("srv_date"), F.lit(BENEFIT_YEAR_START_MONTH),
                                        F.lit(1)))
                      .otherwise(F.make_date(F.year("srv_date") - F.lit(1),
                                             F.lit(BENEFIT_YEAR_START_MONTH), F.lit(1)))))
    cal = (d.groupBy(MEMBER_COL, "calendar_year")
           .agg(F.round(F.sum("_q"), 2).alias("_u"))
           .groupBy(MEMBER_COL)
           .agg(F.round(F.max("_u"), 2).alias("most_in_any_calendar_year"),
                F.max(F.struct(F.col("_u"), F.col("calendar_year")))
                 .alias("_cal_struct")))
    cal = cal.withColumn("the_calendar_year_that_was", F.col("_cal_struct.calendar_year")) \
             .drop("_cal_struct")
    ben = (d.groupBy(MEMBER_COL, "benefit_year_start")
           .agg(F.round(F.sum("_q"), 2).alias("_u"))
           .groupBy(MEMBER_COL)
           .agg(F.round(F.max("_u"), 2).alias("most_in_any_benefit_year"),
                F.max(F.struct(F.col("_u"), F.col("benefit_year_start")))
                 .alias("_ben_struct")))
    ben = ben.withColumn("the_benefit_year_that_was", F.col("_ben_struct.benefit_year_start")) \
             .drop("_ben_struct")
    tot = (d.groupBy(MEMBER_COL)
           .agg(F.round(F.sum("qty"), 2).alias("units_altogether_including_any_that_cannot_count"),
                F.round(F.sum("_q"), 2).alias("units_that_count_towards_the_limit"),
                F.round(F.sum("paid"), 2).alias("paid"),
                F.countDistinct("srv_date").alias("service_days"),
                F.min("srv_date").alias("first_service_date"),
                F.max("srv_date").alias("last_service_date"),
                F.min("member_age").alias("youngest_age_seen"),
                F.max("member_age").alias("oldest_age_seen"),
                _entity_list(BPROV_COL).alias("billing_entities"),
                _entity_list(SPROV_COL).alias("servicing_providers"),
                _entity_list("claim_id").alias("claim_ids")))
    out = (tot.join(cal, MEMBER_COL, "left").join(ben, MEMBER_COL, "left")
           .withColumn("the_limit", F.lit(limit))
           .withColumn("over_on_the_calendar_year_basis",
                       F.col("most_in_any_calendar_year") > F.lit(limit))
           .withColumn("over_on_the_benefit_year_basis",
                       F.col("most_in_any_benefit_year") > F.lit(limit))
           .filter(F.col("over_on_the_calendar_year_basis") |
                   F.col("over_on_the_benefit_year_basis"))
           .withColumn("the_two_bases",
                       F.when(F.col("over_on_the_calendar_year_basis") &
                              F.col("over_on_the_benefit_year_basis"),
                              F.lit("BOTH bases put this member above the limit"))
                        .otherwise(F.lit("AMBIGUOUS: above the limit on one year basis and below "
                                         "it on the other. AHCCCS says 'per year' without saying "
                                         "which year, so this is a question about the basis, not "
                                         "a finding.")))
           .withColumn("rule", F.lit(f"{code} is limited to {limit:g} a year by the Covered "
                                     f"Behavioral Health Services Guide effective 2024-10-01"
                                     + (f", for members aged {min_age} and over" if min_age
                                        else "")
                                     + ". The guide does not state which year, so both the "
                                       "calendar year and the 1 "
                                     + date(2000, BENEFIT_YEAR_START_MONTH, 1).strftime('%B')
                                     + " benefit year are counted here."))
           .withColumn("the_age_rule",
                       F.lit("not age limited" if min_age is None else
                             f"only units billed while the member was {min_age} or over count "
                             f"towards this limit")))
    return out.orderBy(F.desc("most_in_any_calendar_year"))


h0001_year_df = (_per_year_limit("H0001", PER_YEAR_LIMITS["H0001"])
                 .limit(DETAIL_ROWS).toPandas())
_h0001_ambiguous = int((h0001_year_df["the_two_bases"].str.startswith("AMBIGUOUS")).sum()) \
    if len(h0001_year_df) else 0
_mark("Part E rule 19, H0001 above 12 a year", True)
print(f"Rule 19. Members above {PER_YEAR_LIMITS['H0001']:g} H0001 on at least one year basis: "
      f"{len(h0001_year_df)}, of which {_h0001_ambiguous} are AMBIGUOUS (above on one basis, "
      f"below on the other).")

H0002_TEST_STATUS = ("NOT RUN: the limit applies to members aged "
                     f"{H0002_MIN_AGE} and over and this table carries no date of birth, so a "
                     "member the limit does not apply to cannot be told from one it does."
                     if not DOB_COL else "run")
if DOB_COL:
    h0002_year_df = (_per_year_limit("H0002", PER_YEAR_LIMITS["H0002"], min_age=H0002_MIN_AGE)
                     .limit(DETAIL_ROWS).toPandas())
    _h0002_ambiguous = int((h0002_year_df["the_two_bases"].str.startswith("AMBIGUOUS")).sum()) \
        if len(h0002_year_df) else 0
    _mark("Part E rule 20, H0002 above 24 a year for members 11 and over", True)
    print(f"Rule 20. Members above {PER_YEAR_LIMITS['H0002']:g} H0002 on at least one year basis: "
          f"{len(h0002_year_df)}, of which {_h0002_ambiguous} are AMBIGUOUS.")
else:
    h0002_year_df = pd.DataFrame()
    _h0002_ambiguous = None
    _mark("Part E rule 20, H0002 above 24 a year for members 11 and over", False,
          f"no date of birth column, and the limit applies only to members aged {H0002_MIN_AGE} "
          f"and over")
    print(f"Rule 20. {H0002_TEST_STATUS}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Every rule test, counted in one place
# MAGIC
# MAGIC This adds the new tests to the summary that the E_Rule_Summary sheet and the email are built from,
# MAGIC with a written note for each one saying what kind of row it is: a finding, a decoy, a permitted
# MAGIC pattern, a measured column that can never flag, or a question.
# MAGIC
# MAGIC The notes are written out rather than guessed from the label, because a question printed as though
# MAGIC it were a finding is exactly the kind of quiet error this project has been burned by.

# COMMAND ----------

RULE_COUNTS.update({
    "Rule 6: one per diem per member per day, same billing entity": len(one_per_diem_df),
    "Rule 6 testable against the published unit basis": _r6_testable,
    "Rule 6 measured: the per diems came from different billing entities":
        len(one_per_diem_cross_entity_df),
    "Rule 7: levels of care that cannot coexist, same billing entity": len(level_conflicts_df),
    "Rule 7 measured: the conflicting level came from a different billing entity":
        len(level_conflicts_cross_entity_df),
    "Rule 8: H2012 above the per hour tier in one member day": len(h2012_tier_df),
    "Rule 9: above the 2023-07-17 expanded documentation thresholds":
        _exp_counted if _exp_counted is not None else "NOT RUN",
    "Rule 9 decoy: above the threshold but the claim arrived BEFORE the cutoff":
        _exp_excluded if _exp_excluded is not None else "NOT RUN",
    "Rule 10: S9480 more than once in one member day": len(s9480_once_df),
    "Rule 11: S9480 with no prior authorization on or after the applicable date":
        len(s9480_pa_df) if PA_COL else "NOT RUN",
    "Rule 11 exempt: IHS or 638 lines, which the rule does not apply to":
        _pa_exempt_n if _pa_exempt_n is not None else "NOT RUN",
    "Rule 12: S9480 on too few days in a calendar week, a QUESTION": len(s9480_intensity_df),
    "Rule 12 excluded: short weeks that are a calendar edge, not a short programme":
        S9480_EDGE_WEEKS_REMOVED,
    "Rule 14: H0018 with a place of service other than 56":
        len(h0018_pos_df) if POS_CODE_COL else "NOT RUN",
    "Rule 15: focus codes billed that the code to provider type matrix should settle":
        len(_flagged_codes),
    "Rule 16: bed hold or therapeutic leave above the contract year limit":
        bed_hold_over_n if bed_hold_over_n is not None else "NOT RUN",
    "Rule 17: 98960 above the daily unit cap": len(edu_daily_cap_df),
    "Rule 18: 98960 above the rolling window unit cap": len(edu_rolling_df),
    "Rule 19: H0001 above 12 a year on at least one year basis": len(h0001_year_df),
    "Rule 19 ambiguous: above on one year basis and below on the other": _h0001_ambiguous,
    "Rule 20: H0002 above 24 a year on at least one year basis":
        len(h0002_year_df) if DOB_COL else "NOT RUN",
    "Rule 20 ambiguous: above on one year basis and below on the other":
        _h0002_ambiguous if _h0002_ambiguous is not None else "NOT RUN",
})
_assert_unique(RULE_COUNTS, "rule count labels")

# Notes for the new tests, written out rather than inferred from the label, so a question is
# never printed as though it were a finding.
RULE_NOTE_OVERRIDES = {
    "Rule 6 testable against the published unit basis":
        "how many of the rule 6 rows rest only on codes with a published unit basis",
    "Rule 6 measured: the per diems came from different billing entities":
        "measured, never flagged: no per claim edit could have seen this and this provider did "
        "not bill the other entity's per diem",
    "Rule 7 measured: the conflicting level came from a different billing entity":
        "measured, never flagged: some same day combinations are permitted when a different "
        "entity bills",
    "Rule 9 decoy: above the threshold but the claim arrived BEFORE the cutoff":
        "a decoy, counted so you can see how much of it there is. It cannot flag anybody",
    "Rule 11 exempt: IHS or 638 lines, which the rule does not apply to":
        "exempt lines, counted so you can see how many were removed from the test",
    "Rule 12: S9480 on too few days in a calendar week, a QUESTION":
        "A QUESTION, NOT A FINDING. A per diem claim carries no hours, so this cannot show that "
        "the programme ran short. It shows how many days it was billed in the week. Weeks that "
        "are a calendar edge have been excluded, not merely labelled",
    "Rule 12 excluded: short weeks that are a calendar edge, not a short programme":
        "excluded rows, counted so you can see how many were taken out. They are on "
        "E_Min_Intensity_Edges and they are NOT findings. There is deliberately no H0035 "
        "equivalent of this test: its published minimum is in HOURS and a per diem claim carries "
        "none, so the distribution of H0035 days per week is reported instead with no threshold",
    "Rule 15: focus codes billed that the code to provider type matrix should settle":
        "read the single statement on E_Provider_Type_Check, not this count",
    "Rule 19 ambiguous: above on one year basis and below on the other":
        "AMBIGUOUS, not a finding. AHCCCS says 'per year' without saying which year, and these "
        "members are above the limit on one basis and below it on the other",
    "Rule 20 ambiguous: above on one year basis and below on the other":
        "AMBIGUOUS, not a finding, for the same reason",
}

# COMMAND ----------

# MAGIC %md
# MAGIC ## 17. Part H. The daily census
# MAGIC
# MAGIC **This is the organisation level equivalent of the clinician day test, and for a per diem
# MAGIC focus set it is the headline rather than the hours.**
# MAGIC
# MAGIC For every calendar day in the period, how many distinct members this provider had on a per
# MAGIC diem: overall, and for each per diem code separately. That is a count of people in care on a
# MAGIC day, which is a thing an organisation can be compared against, whereas a count of hours is
# MAGIC not.
# MAGIC
# MAGIC **Two things about the arithmetic.**
# MAGIC
# MAGIC Every calendar day between the first and last service date gets a row, including the days
# MAGIC with nobody on a per diem at all. A census curve with the empty days missing looks completely
# MAGIC different from the real one, so the mean over every calendar day and the mean over the days
# MAGIC with somebody in care are both reported and both labelled.
# MAGIC
# MAGIC Codes whose unit basis AHCCCS has never published contribute **nothing** to the census. H0019
# MAGIC is the one that matters in this focus set: it keeps every dollar it was paid, it is reported
# MAGIC in a column of its own, and it adds nobody to any day's count, because a census built partly
# MAGIC on an unpublished unit is a guess dressed up as arithmetic.
# MAGIC
# MAGIC **What the census is for, and what this notebook will not do with it.** The number to compare
# MAGIC a census against is the facility's licensed capacity. The state's own forensic audit did
# MAGIC exactly that in its finding F12, comparing billing against a licensed capacity of eight.
# MAGIC **This notebook does not hold licensed capacity and does not guess at it.** Capacity is the
# MAGIC state's record, held in provider enrolment and licensing, and checking the peak census here
# MAGIC against that record is a job for the investigator with the licence in front of them. What
# MAGIC this workbook can give them is the curve, the peak, and the dates the peak happened on.

# COMMAND ----------

# Only per diem codes with a published unit basis may put a member into a census.
_census_src = df_day.filter(F.col("counts_as_a_day_of_care"))
_census_codes = sorted({r[0] for r in _census_src.select(PROC_COL).distinct().collect()})
_unverified_pd_codes = sorted({r[0] for r in
                               df_day.filter(F.col("is_per_diem_any") &
                                             ~F.col("counts_as_a_day_of_care"))
                               .select(PROC_COL).distinct().collect()})

if N_LINES and DATA_START and DATA_END:
    _n_days = (DATA_END - DATA_START).days + 1
    date_spine = (spark.range(0, _n_days)
                  .select(F.date_add(F.lit(DATA_START), F.col("id").cast("int")).alias("srv_date")))
else:
    _n_days = 0
    date_spine = spark.createDataFrame([], "srv_date date")

if _census_codes:
    _overall = (_census_src.groupBy("srv_date")
                .agg(F.countDistinct(MEMBER_COL).alias("members_on_a_per_diem"),
                     F.round(F.sum("day_paid"), 2).alias("per_diem_paid_that_day"),
                     F.round(F.sum("day_qty"), 2).alias("per_diem_units_that_day"),
                     F.count(F.lit(1)).alias("per_diem_day_rows"),
                     _entity_list(PROC_COL).alias("per_diem_codes_that_day"),
                     F.countDistinct(BPROV_COL).alias("billing_entities_that_day")))
    _by_code = (_census_src.groupBy("srv_date").pivot(PROC_COL, _census_codes)
                .agg(F.countDistinct(MEMBER_COL)))
    _by_code = _by_code.select("srv_date", *[F.col(f"`{c}`").alias(f"members_on_{c}")
                                             for c in _census_codes])
    _assert_unique(_by_code.columns, "daily census columns")
    census = _overall.join(_by_code, "srv_date", "outer")
else:
    census = spark.createDataFrame([], "srv_date date").withColumn(
        "members_on_a_per_diem", F.lit(0))

# The measured, never counted column: members on a per diem code AHCCCS has never published a
# unit for. These people are not in any census figure above and the column exists so that is
# visible rather than invisible.
if _unverified_pd_codes:
    _unv_census = (df_day.filter(F.col("is_per_diem_any") & ~F.col("counts_as_a_day_of_care"))
                   .groupBy("srv_date")
                   .agg(F.countDistinct(MEMBER_COL)
                        .alias("members_on_a_code_with_no_published_unit_NOT_counted"),
                        F.round(F.sum("day_paid"), 2)
                        .alias("paid_on_codes_with_no_published_unit")))
    census = census.join(_unv_census, "srv_date", "outer")

# A day with nobody in care is a zero, not a blank, and a day with nobody in care has no codes
# on it rather than a null. Only the numeric columns are filled with zero: filling a text column
# with a zero would print the digit 0 where the honest answer is that nothing was billed.
_num_types = ("int", "bigint", "double", "float", "long", "smallint", "tinyint", "decimal")
_joined = date_spine.join(census, "srv_date", "left")
_fill = {c: 0 for c, t in _joined.dtypes
         if c != "srv_date" and any(t.startswith(n) for n in _num_types)}
_text = [c for c, t in _joined.dtypes if t == "string"]
daily_census_df = (_joined.fillna(_fill)
                   .fillna({c: "nothing was billed on this day" for c in _text})
                   .withColumn("day_of_week", F.date_format("srv_date", "EEEE"))
                   .orderBy("srv_date")
                   .limit(DETAIL_ROWS).toPandas())

CENSUS_AVAILABLE = bool(len(daily_census_df)) and bool(_census_codes)
if CENSUS_AVAILABLE:
    _c = daily_census_df["members_on_a_per_diem"].astype(float)
    _active = _c[_c > 0]
    CENSUS_PEAK = float(_c.max())
    CENSUS_PEAK_DATES = daily_census_df.loc[_c == CENSUS_PEAK, "srv_date"].tolist()
    census_summary_df = pd.DataFrame([
        ["Calendar days in the period", _n_days],
        ["Days with at least one member on a per diem", int((_c > 0).sum())],
        ["Days with nobody on a per diem", int((_c == 0).sum())],
        ["Mean members on a per diem, over EVERY calendar day", round(float(_c.mean()), 2)],
        ["Mean members on a per diem, over days with somebody in care",
         round(float(_active.mean()), 2) if len(_active) else 0.0],
        ["Median members on a per diem, over EVERY calendar day", round(float(_c.median()), 2)],
        ["Median members on a per diem, over days with somebody in care",
         round(float(_active.median()), 2) if len(_active) else 0.0],
        ["75th percentile, over days with somebody in care",
         round(float(_active.quantile(0.75)), 2) if len(_active) else 0.0],
        ["90th percentile, over days with somebody in care",
         round(float(_active.quantile(0.90)), 2) if len(_active) else 0.0],
        ["PEAK members on a per diem in one day", int(CENSUS_PEAK)],
        ["The date or dates the peak happened on",
         ", ".join(str(d) for d in CENSUS_PEAK_DATES[:40])
         + (f" and {len(CENSUS_PEAK_DATES) - 40} more" if len(CENSUS_PEAK_DATES) > 40 else "")],
        ["Codes counted into the census", ", ".join(_census_codes)],
        ["Per diem codes NOT counted into the census",
         ", ".join(_unverified_pd_codes) + ". AHCCCS has never published what one of their units "
         "covers, so they keep their dollars and add nobody to any day."
         if _unverified_pd_codes else "none: every per diem code billed has a published unit"],
        ["What to compare the peak against",
         "The facility's LICENSED CAPACITY. That is the state's own record, held in provider "
         "enrolment and licensing. This notebook does not hold it and does not guess at it. The "
         "state's forensic audit compared billing against a licensed capacity of eight in its "
         "finding F12, and that comparison is the job this sheet is meant to make possible, not "
         "one it performs."],
    ], columns=["Item", "What it is"])
    census_distribution_df = (daily_census_df.groupby("members_on_a_per_diem")
                              .size().reset_index(name="days_at_this_census")
                              .rename(columns={"members_on_a_per_diem": "members_on_a_per_diem"})
                              .sort_values("members_on_a_per_diem").reset_index(drop=True))
    census_distribution_df["share_of_calendar_days"] = [
        _rate(v, len(daily_census_df)) for v in census_distribution_df.days_at_this_census]
    print(f"Daily census: peak {int(CENSUS_PEAK)} members on "
          f"{', '.join(str(d) for d in CENSUS_PEAK_DATES[:5])}"
          f"{' and others' if len(CENSUS_PEAK_DATES) > 5 else ''}; "
          f"mean {float(_c.mean()):.2f} over every calendar day, "
          f"{(float(_active.mean()) if len(_active) else 0):.2f} over days with somebody in care.")
    print("Compare the peak against the facility's LICENSED CAPACITY, which is the state's record "
          "and is not in this notebook.")
else:
    CENSUS_PEAK, CENSUS_PEAK_DATES = None, []
    census_summary_df = pd.DataFrame()
    census_distribution_df = pd.DataFrame()
    print("Daily census: NOT BUILT. These providers billed no per diem code with a published unit "
          "basis, so there is no day of care to count.")
_mark("Part H, the daily census", CENSUS_AVAILABLE,
      "these providers billed no per diem code with a published unit basis, so there is no day "
      "of care to count")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 18. Part I. Episodes, length of stay and returns
# MAGIC
# MAGIC A census says how many people were in care on a day. This says how long each of them stayed.
# MAGIC
# MAGIC A stay is built from a member's consecutive billed days on one per diem code. The gap that
# MAGIC still counts as one stay is a setting, and it matches the member journey screen exactly: a gap
# MAGIC of more than one day starts a new stay, so the days have to be consecutive. The setting is on
# MAGIC the Criteria sheet because it changes every number on these sheets.
# MAGIC
# MAGIC **Three questions, and the third is the one to read first.**
# MAGIC
# MAGIC 1. **How long are the stays?** The whole distribution is reported, not a headline average,
# MAGIC    because a mean length of stay hides exactly the thing worth seeing.
# MAGIC 2. **Who came back, and how often?** A member whose stay ends and who returns inside a short
# MAGIC    window, repeatedly, is a different question from a member with one long stay.
# MAGIC 3. **How long are the H0018 runs?** H0018 is **short term** residential. That is what the code
# MAGIC    means. So a long unbroken run of it is a question on the face of the code, before anybody
# MAGIC    looks at anything else.
# MAGIC
# MAGIC **About the long stay cut, and this matters.** AHCCCS publishes no maximum length of stay for
# MAGIC H0018. So this notebook does not invent one. What it does is report the whole distribution of
# MAGIC this provider's own stays and mark the tail of it, at a percentile given in the settings. That
# MAGIC cut is **descriptive**: it says a stay is long compared with this provider's other stays. It
# MAGIC does not say the stay was too long, and it is not a published limit. If the provider has too
# MAGIC few stays for a percentile to mean anything, the tail is not marked at all and the sheet says
# MAGIC so.

# COMMAND ----------

_stay_src = (df_day.filter(F.col("counts_as_a_day_of_care"))
             .groupBy(SPROV_COL, MEMBER_COL, PROC_COL, "srv_date")
             .agg(F.round(F.sum("day_paid"), 2).alias("d_paid"),
                  F.round(F.sum("day_qty"), 2).alias("d_units"),
                  _entity_list(BPROV_COL).alias("d_entities"),
                  _entity_list("claim_id").alias("d_claims")))
_sw = Window.partitionBy(SPROV_COL, MEMBER_COL, PROC_COL).orderBy("srv_date")
_stay_src = (_stay_src
             .withColumn("_prev", F.lag("srv_date").over(_sw))
             .withColumn("_new_stay",
                         F.when(F.col("_prev").isNull() |
                                (F.datediff("srv_date", "_prev") > F.lit(STAY_GAP_DAYS)),
                                F.lit(1)).otherwise(F.lit(0)))
             .withColumn("stay_id", F.sum("_new_stay").over(
                 _sw.rowsBetween(Window.unboundedPreceding, 0))))
stays = (_stay_src.groupBy(SPROV_COL, MEMBER_COL, PROC_COL, "stay_id")
         .agg(F.min("srv_date").alias("stay_starts"),
              F.max("srv_date").alias("stay_ends"),
              F.countDistinct("srv_date").alias("days_billed_in_the_stay"),
              F.round(F.sum("d_paid"), 2).alias("paid"),
              F.round(F.sum("d_units"), 2).alias("units"),
              _entity_list("d_entities").alias("billing_entities"),
              _entity_list("d_claims").alias("claim_ids"))
         .withColumn("calendar_days_the_stay_covers",
                     F.datediff("stay_ends", "stay_starts") + F.lit(1))
         .cache())

# Returns: the next stay on the same code for the same member, and how long after this one ended.
_rw = Window.partitionBy(SPROV_COL, MEMBER_COL, PROC_COL).orderBy("stay_starts", "stay_ends")
stays = (stays
         .withColumn("_next_start", F.lead("stay_starts").over(_rw))
         .withColumn("days_until_the_next_stay",
                     F.when(F.col("_next_start").isNotNull(),
                            F.datediff("_next_start", "stay_ends")))
         .withColumn("the_member_returned_within_the_window",
                     F.col("days_until_the_next_stay").isNotNull() &
                     (F.col("days_until_the_next_stay") <= F.lit(READMIT_WINDOW_DAYS)))
         .drop("_next_start"))

N_STAYS = stays.count()
stays_df = (stays.orderBy(F.desc("days_billed_in_the_stay"), F.desc("paid"))
            .limit(DETAIL_ROWS).toPandas())

# The length of stay distribution, per code, reported whole rather than as an average.
los_distribution_df = (
    stays.groupBy(PROC_COL)
    .agg(F.count(F.lit(1)).alias("stays"),
         F.countDistinct(MEMBER_COL).alias("members"),
         F.round(F.avg("days_billed_in_the_stay"), 2).alias("mean_days"),
         F.expr("percentile_approx(days_billed_in_the_stay, 0.25)").alias("p25_days"),
         F.expr("percentile_approx(days_billed_in_the_stay, 0.5)").alias("median_days"),
         F.expr("percentile_approx(days_billed_in_the_stay, 0.75)").alias("p75_days"),
         F.expr("percentile_approx(days_billed_in_the_stay, 0.9)").alias("p90_days"),
         F.expr("percentile_approx(days_billed_in_the_stay, 0.95)").alias("p95_days"),
         F.max("days_billed_in_the_stay").alias("longest_stay_days"),
         F.round(F.sum("paid"), 2).alias("paid"),
         F.round(F.sum("days_billed_in_the_stay"), 0).alias("days_of_care_altogether"))
    .orderBy(F.desc("paid")).toPandas())
if len(los_distribution_df):
    los_distribution_df["what_the_code_is"] = los_distribution_df[PROC_COL].map(
        lambda c: FOCUS_CODE_NOTE.get(c, CODE_BASIS.get(c, "")))

# The tail cut: a percentile of THIS PROVIDER'S OWN stays, which is descriptive and not a limit.
_tail_rows, LONG_STAY_CUTS = [], {}
for _code in (sorted({r[0] for r in stays.select(PROC_COL).distinct().collect()})
              if N_STAYS else []):
    _sl = stays_df[stays_df[PROC_COL] == _code]["days_billed_in_the_stay"] \
        if len(stays_df) else pd.Series(dtype=float)
    if len(_sl) >= MIN_STAYS_FOR_A_TAIL:
        _cut = float(pd.to_numeric(_sl).quantile(LONG_STAY_TAIL_PCT))
        LONG_STAY_CUTS[_code] = _cut
        _tail_rows.append({
            "procedure_code": _code, "stays": len(_sl),
            "the_tail_cut_in_days": round(_cut, 1),
            "how_the_cut_was_set": f"the {LONG_STAY_TAIL_PCT:.0%} point of THIS PROVIDER'S OWN "
                                   f"stay lengths on this code",
            "what_the_cut_means": "DESCRIPTIVE ONLY. It says a stay is long compared with this "
                                  "provider's other stays. AHCCCS publishes no maximum length of "
                                  "stay for these codes, so this is not a published limit and a "
                                  "stay in the tail is not by itself a finding.",
            "stays_in_the_tail": int((pd.to_numeric(_sl) >= _cut).sum()),
        })
    else:
        _tail_rows.append({
            "procedure_code": _code, "stays": len(_sl),
            "the_tail_cut_in_days": None,
            "how_the_cut_was_set": f"NOT SET: fewer than {MIN_STAYS_FOR_A_TAIL} stays on this "
                                   f"code, so a percentile of them would be noise",
            "what_the_cut_means": "The distribution is still reported on the length of stay sheet. "
                                  "Read that instead.",
            "stays_in_the_tail": None,
        })
long_stay_cut_df = pd.DataFrame(_tail_rows)

if len(stays_df):
    stays_df["in_the_long_tail_of_this_providers_own_distribution"] = [
        (None if r[PROC_COL] not in LONG_STAY_CUTS
         else bool(r["days_billed_in_the_stay"] >= LONG_STAY_CUTS[r[PROC_COL]]))
        for _, r in stays_df.iterrows()]
    stays_df["this_code_is_short_term_residential"] = stays_df[PROC_COL] == "H0018"

# H0018 specifically, because the code says SHORT TERM and a long unbroken run of it is a
# question on the face of the code.
h0018_long_stays_df = (stays_df[stays_df[PROC_COL] == "H0018"]
                       .sort_values("days_billed_in_the_stay", ascending=False)
                       .head(DETAIL_ROWS).reset_index(drop=True)
                       if len(stays_df) else pd.DataFrame())
if len(h0018_long_stays_df):
    h0018_long_stays_df["why_this_sheet_exists"] = (
        "H0018 is SHORT TERM residential. A long unbroken run of it is a question on the face of "
        "the code, before anybody looks at anything else. The tail cut on this sheet is this "
        "provider's own distribution and is not a published limit: AHCCCS publishes no maximum "
        "length of stay for H0018.")

longest_stays_df = (stays_df.sort_values("days_billed_in_the_stay", ascending=False)
                    .head(100).reset_index(drop=True) if len(stays_df) else pd.DataFrame())

# Returns, per member per code.
readmissions_df = (
    stays.groupBy(SPROV_COL, MEMBER_COL, PROC_COL)
    .agg(F.count(F.lit(1)).alias("stays"),
         F.sum(F.col("the_member_returned_within_the_window").cast("int"))
          .alias("returns_within_the_window"),
         F.round(F.avg(F.when(F.col("the_member_returned_within_the_window"),
                              F.col("days_until_the_next_stay"))), 1)
          .alias("mean_days_between_a_stay_ending_and_the_next_beginning"),
         F.min("days_until_the_next_stay").alias("shortest_gap_between_stays_in_days"),
         F.round(F.sum("days_billed_in_the_stay"), 0).alias("days_of_care_altogether"),
         F.round(F.sum("paid"), 2).alias("paid"),
         F.min("stay_starts").alias("first_stay_starts"),
         F.max("stay_ends").alias("last_stay_ends"))
    .filter(F.col("returns_within_the_window") >= F.lit(READMIT_MIN_RETURNS))
    .withColumn("the_window_in_days", F.lit(READMIT_WINDOW_DAYS))
    .withColumn("what_this_row_is",
                F.lit(f"This member's stay ended and they returned inside "
                      f"{READMIT_WINDOW_DAYS} days, {READMIT_MIN_RETURNS} times or more. That is "
                      f"a different question from one long stay and it is a description of the "
                      f"billing pattern, not a conclusion about the care."))
    .orderBy(F.desc("returns_within_the_window"), F.desc("paid"))
    .limit(DETAIL_ROWS).toPandas())

_mark("Part I, episodes and length of stay", N_STAYS > 0,
      "no per diem code with a published unit basis was billed, so no stay could be built")
print(f"Stays built: {N_STAYS:,}. Members returning within {READMIT_WINDOW_DAYS} days "
      f"{READMIT_MIN_RETURNS} times or more: {len(readmissions_df)}.")
if len(los_distribution_df):
    print(los_distribution_df[[PROC_COL, "stays", "members", "median_days", "p90_days",
                               "longest_stay_days", "days_of_care_altogether"]]
          .to_string(index=False))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 19. Part J. The servicing providers under this organisation
# MAGIC
# MAGIC An organisation has clinicians. This section looks at every line billed by the billing
# MAGIC entities that appear on the target's claims and asks who the servicing provider was.
# MAGIC
# MAGIC **The question this is really asking.** When an organisation puts its own identifier in the
# MAGIC servicing provider field on its own claims, the claim says the organisation performed the
# MAGIC service rather than naming the person who did. That is the pattern the state's forensic audit
# MAGIC raised about organisational and individual provider numbers. It is also what makes a day total
# MAGIC for that identifier meaningless as one person's day, which is the connection to Part D.
# MAGIC
# MAGIC This is a description of how the claims were filled in. It is not a finding on its own: there
# MAGIC are legitimate reasons for an organisation to appear as the servicing provider, and the
# MAGIC enrolment record settles which applies here.

# COMMAND ----------

ORG_SCOPE_STATUS = "run"
org_servicing_df = pd.DataFrame()
org_self_servicing_df = pd.DataFrame()
ORG_SELF_SERVICING_STATEMENT = ""
if not TARGET_ENTITY_SET:
    ORG_SCOPE_STATUS = ("NOT RUN: no billing entity appears on these providers' claims, so there "
                        "is no organisation scope to look at.")
else:
    _org_raw = (spark.table(SERVICE_TABLE)
                .withColumn("_bp", F.trim(F.coalesce(F.col(BPROV_COL).cast("string"), F.lit(""))))
                .filter(F.col("_bp").isin(sorted(TARGET_ENTITY_SET))))
    _org_n = _org_raw.count()
    if _org_n > ORG_SCOPE_MAX_LINES:
        ORG_SCOPE_STATUS = (f"NOT RUN: the billing entities on these providers' claims carry "
                            f"{_org_n:,} lines altogether, which is above the "
                            f"{ORG_SCOPE_MAX_LINES:,} line ceiling in the settings. One of them is "
                            f"probably a clearing house rather than the organisation itself. "
                            f"Narrow the targets or raise org_scope_max_lines.")
        print(ORG_SCOPE_STATUS)
    else:
        df_org = prepare(_org_raw.withColumn(BPROV_COL, F.col("_bp"))).cache()
        _org_tot = df_org.agg(F.round(F.sum("paid"), 2).alias("p"),
                              F.count(F.lit(1)).alias("l")).first()
        org_servicing_df = (
            df_org.groupBy(BPROV_COL, SPROV_COL)
            .agg(F.concat_ws(" | ", F.array_sort(F.collect_set(
                     F.when(F.col("sprov_name") != F.lit(""), F.col("sprov_name")))))
                  .alias("servicing_provider_name"),
                 F.concat_ws(" | ", F.array_sort(F.collect_set(
                     F.when(F.col("sprov_type_desc") != F.lit(""), F.col("sprov_type_desc")))))
                  .alias("servicing_provider_type"),
                 F.count(F.lit(1)).alias("lines"),
                 F.round(F.sum("qty"), 2).alias("units"),
                 F.round(F.sum("paid"), 2).alias("paid"),
                 F.countDistinct(MEMBER_COL).alias("members"),
                 F.countDistinct("srv_date").alias("service_days"),
                 F.countDistinct(PROC_COL).alias("codes"),
                 F.min("srv_date").alias("first_service_date"),
                 F.max("srv_date").alias("last_service_date"),
                 F.round(F.sum(F.when(F.col("is_focus_code"), F.col("paid")).otherwise(0.0)), 2)
                  .alias("paid_on_focus_codes"),
                 _entity_list(F.when(F.col("is_focus_code"), F.col(PROC_COL)))
                  .alias("focus_codes_billed"))
            .withColumn("the_servicing_provider_IS_the_billing_entity",
                        F.col(SPROV_COL) == F.col(BPROV_COL))
            .withColumn("is_one_of_the_targets",
                        F.col(SPROV_COL).isin(TARGETS))
            .orderBy(F.desc("paid"))
            .limit(DETAIL_ROWS).toPandas())
        if len(org_servicing_df):
            org_servicing_df["share_of_the_entity_paid"] = [
                _rate(a, _org_tot["p"]) for a in org_servicing_df.paid]
        _self = (df_org.agg(
            F.round(F.sum(F.when(F.col(SPROV_COL) == F.col(BPROV_COL), F.col("paid"))
                          .otherwise(0.0)), 2).alias("self_paid"),
            F.sum(F.when(F.col(SPROV_COL) == F.col(BPROV_COL), F.lit(1)).otherwise(F.lit(0)))
             .alias("self_lines"),
            F.countDistinct(SPROV_COL).alias("distinct_servicing_providers")).first())
        org_self_servicing_df = pd.DataFrame([
            ["Billing entities in scope", ", ".join(sorted(TARGET_ENTITY_SET))],
            ["Lines billed by those entities", f"{_org_n:,}"],
            ["Distinct servicing providers appearing on them",
             int(_self["distinct_servicing_providers"] or 0)],
            ["Lines where the servicing provider IS the billing entity",
             f"{int(_self['self_lines'] or 0):,} of {int(_org_tot['l'] or 0):,} "
             f"({_rate(_self['self_lines'], _org_tot['l']):.1%})"
             if _org_tot["l"] else "0"],
            ["Dollars on those lines",
             f"${float(_self['self_paid'] or 0):,.2f} of ${float(_org_tot['p'] or 0):,.2f}"],
            ["What this means",
             "When an organisation's own identifier is the servicing provider on its own claims, "
             "the claim says the organisation performed the service rather than naming the person "
             "who did. That is the pattern the state's forensic audit raised about organisational "
             "and individual provider numbers, and it is why a day total for that identifier "
             "cannot be read as one person's day. It is a description of how the claims were "
             "filled in and not a finding on its own: the enrolment record settles whether it is "
             "correct here."],
        ], columns=["Item", "What it is"])
        ORG_SELF_SERVICING_STATEMENT = (
            f"Of {int(_org_tot['l'] or 0):,} lines billed by these entities, "
            f"{int(_self['self_lines'] or 0):,} carry the billing entity's own identifier as the "
            f"servicing provider"
            + (f" ({_rate(_self['self_lines'], _org_tot['l']):.1%})" if _org_tot["l"] else "")
            + f", and {int(_self['distinct_servicing_providers'] or 0)} distinct servicing "
              f"providers appear on them altogether.")
        print(ORG_SELF_SERVICING_STATEMENT)
_mark("Part J, servicing providers under the organisation",
      ORG_SCOPE_STATUS == "run", ORG_SCOPE_STATUS.replace("NOT RUN: ", ""))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 20. Part K. Paid dollars by month
# MAGIC
# MAGIC Every dollar this provider was paid, by calendar month, beside a reference figure of
# MAGIC $500,000 a month.
# MAGIC
# MAGIC **Read this sentence before reading the column.** That figure is **not an AHCCCS control**. It
# MAGIC is the figure the state's forensic audit proposed in its recommendation 12 as a review trigger
# MAGIC for Integrated Clinics. Nothing was denied for crossing it, nothing is owed for crossing it,
# MAGIC and a month above it is not a finding. It is a comparison, reported here for the same reason
# MAGIC the cluster screen reports it: so that a reviewer who has read the audit can see where this
# MAGIC provider sits against the number the auditors had in mind.

# COMMAND ----------

monthly_dollars_df = (
    df_line.groupBy(SPROV_COL, "month")
    .agg(F.round(F.sum("paid"), 2).alias("paid"),
         F.round(F.sum("billed"), 2).alias("billed"),
         F.count(F.lit(1)).alias("lines"),
         F.round(F.sum("qty"), 2).alias("units"),
         F.countDistinct(MEMBER_COL).alias("members"),
         F.countDistinct("srv_date").alias("service_days"),
         F.countDistinct(PROC_COL).alias("codes"),
         F.round(F.sum(F.when(F.col("is_focus_code"), F.col("paid")).otherwise(0.0)), 2)
          .alias("paid_on_focus_codes"),
         F.round(F.sum(F.when(F.col("counts_as_a_day_of_care"), F.col("qty")).otherwise(0.0)), 2)
          .alias("days_of_care_billed"))
    .withColumn("the_reference_figure", F.lit(MONTHLY_PAID_REFERENCE))
    .withColumn("above_the_reference_figure", F.col("paid") > F.lit(MONTHLY_PAID_REFERENCE))
    .withColumn("dollars_above_the_reference_figure",
                F.round(F.greatest(F.col("paid") - F.lit(MONTHLY_PAID_REFERENCE), F.lit(0.0)), 2))
    .withColumn("what_the_reference_figure_is",
                F.lit("The state forensic audit's recommendation 12 proposed roughly "
                      "$500,000 a month as a review trigger for Integrated Clinics. It is an "
                      "AUDITOR'S RECOMMENDED FIGURE, not a live AHCCCS control. A month above it "
                      "is not a finding and nothing was denied for crossing it."))
    .orderBy(SPROV_COL, "month")
    .limit(DETAIL_ROWS).toPandas())
MONTHS_OVER_REFERENCE = int(monthly_dollars_df["above_the_reference_figure"].sum()) \
    if len(monthly_dollars_df) else 0
_mark("Part K, paid dollars by month against the auditor reference figure", True)
print(f"Months paid above the ${MONTHLY_PAID_REFERENCE:,.0f} auditor reference figure: "
      f"{MONTHS_OVER_REFERENCE} of {len(monthly_dollars_df)}. That figure is a recommendation from "
      f"the state's forensic audit, NOT an AHCCCS control, and a month above it is not a finding.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 21. Part L. The S9480 pricing era check
# MAGIC
# MAGIC AHCCCS paid S9480 at 58.66% of covered billed charges under By Report pricing, and replaced
# MAGIC that with a fixed $370.41 effective 2022-10-01. So dividing paid by billed on each S9480 line
# MAGIC and watching when the ratio stops clustering at 0.5866 dates the pricing change **in the
# MAGIC client's own data**.
# MAGIC
# MAGIC That is worth doing for two reasons. It confirms the extract behaves the way the published
# MAGIC notices say it should, which is a check on the extract rather than on the provider. And where
# MAGIC a rate date is genuinely in conflict between AHCCCS sources, this is how the conflict gets
# MAGIC resolved empirically instead of by picking a source.
# MAGIC
# MAGIC It needs a billed amount column. Without one the check reports NOT RUN.

# COMMAND ----------

PRICING_ERA_STATUS = ("NOT RUN: this table carries no billed amount column, so paid over billed "
                      "cannot be computed and the pricing era cannot be dated from the data."
                      if not BILL_COL else "run")
if BILL_COL:
    _s9480 = (df_line.filter((F.col(PROC_COL) == F.lit("S9480")) & (F.col("billed") > 0))
              .withColumn("paid_over_billed", F.col("paid") / F.col("billed"))
              .withColumn("within_the_by_report_ratio",
                          F.abs(F.col("paid_over_billed") - F.lit(BY_REPORT_RATIO)) <=
                          F.lit(BY_REPORT_TOLERANCE))
              .withColumn("paid_per_unit",
                          F.when(F.col("qty") > 0, F.col("paid") / F.col("qty")))
              .withColumn("at_the_fixed_rate",
                          F.abs(F.coalesce(F.col("paid_per_unit"), F.lit(0.0)) -
                                F.lit(S9480_FIXED_RATE)) <= F.lit(0.01)))
    s9480_pricing_df = (
        _s9480.groupBy("month")
        .agg(F.count(F.lit(1)).alias("s9480_lines_with_a_billed_amount"),
             F.round(F.avg("paid_over_billed"), 4).alias("mean_paid_over_billed"),
             F.round(F.expr("percentile_approx(paid_over_billed, 0.5)"), 4)
              .alias("median_paid_over_billed"),
             F.sum(F.col("within_the_by_report_ratio").cast("int"))
              .alias("lines_at_the_by_report_ratio"),
             F.sum(F.col("at_the_fixed_rate").cast("int")).alias("lines_at_the_fixed_rate"),
             F.round(F.avg("paid_per_unit"), 2).alias("mean_paid_per_unit"),
             F.round(F.sum("paid"), 2).alias("paid"))
        .withColumn("share_at_the_by_report_ratio",
                    F.round(F.col("lines_at_the_by_report_ratio") /
                            F.col("s9480_lines_with_a_billed_amount"), 4))
        .withColumn("share_at_the_fixed_rate",
                    F.round(F.col("lines_at_the_fixed_rate") /
                            F.col("s9480_lines_with_a_billed_amount"), 4))
        .withColumn("clustering_at_the_by_report_ratio",
                    F.col("share_at_the_by_report_ratio") >= F.lit(BY_REPORT_CLUSTER_SHARE))
        .orderBy("month").toPandas())
    PRICING_CHANGE_MONTH = None
    if len(s9480_pricing_df):
        _clust = s9480_pricing_df["clustering_at_the_by_report_ratio"].tolist()
        _months = s9480_pricing_df["month"].tolist()
        for _i in range(1, len(_clust)):
            if _clust[_i - 1] and not _clust[_i]:
                PRICING_CHANGE_MONTH = _months[_i]
                break
        s9480_pricing_df["this_is_the_month_the_ratio_stops_clustering"] = [
            m == PRICING_CHANGE_MONTH for m in _months]
        s9480_pricing_df["how_to_read_this"] = (
            f"Under By Report pricing AHCCCS paid S9480 at {BY_REPORT_RATIO:.4f} of covered "
            f"billed charges. A fixed ${S9480_FIXED_RATE:,.2f} replaced it effective "
            f"{S9480_FIXED_RATE_DATE}. The month the ratio stops clustering dates that change in "
            f"this extract. This is a check on the EXTRACT, not a finding against the provider.")
    _mark("Part L, the S9480 pricing era check", True)
    if PRICING_CHANGE_MONTH is not None:
        print(f"S9480 paid over billed stops clustering at {BY_REPORT_RATIO} in "
              f"{PRICING_CHANGE_MONTH}. The published fixed rate took effect "
              f"{S9480_FIXED_RATE_DATE}.")
    elif len(s9480_pricing_df):
        print(f"S9480 paid over billed never clusters at {BY_REPORT_RATIO} and then stops in this "
              f"extract, so the pricing change could not be dated from the data. The monthly "
              f"ratios are on the sheet.")
    else:
        print("No S9480 line carries a billed amount, so the pricing era could not be dated.")
else:
    s9480_pricing_df = pd.DataFrame()
    PRICING_CHANGE_MONTH = None
    _mark("Part L, the S9480 pricing era check", False,
          "no billed amount column, so paid over billed cannot be computed")
    print(f"Part L. {PRICING_ERA_STATUS}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 22. Part M. The funnel: the front door and the treatment behind it
# MAGIC
# MAGIC The focus set has two halves. Assessment, screening for eligibility for admission and
# MAGIC education are the **front door**: they decide who gets admitted. The per diems and day
# MAGIC treatment are the **treatment** people are admitted to. A provider billing both halves is
# MAGIC doing both jobs, and this section describes the order in which it billed them.
# MAGIC
# MAGIC **This is a description of billing order. It is not an allegation of anything.** Integrated
# MAGIC care providers assess and then treat their own patients every day, entirely properly, and that
# MAGIC is the normal case rather than the odd one. What the sheets give an investigator is the
# MAGIC ability to see how often it happens here, and how tightly the two halves are spaced.
# MAGIC
# MAGIC **Four questions, and all four are worth the same attention.**
# MAGIC
# MAGIC 1. **For each member, how long after the front door did treatment start?** The whole lag
# MAGIC    distribution is reported. Ordinary clinical intake produces lags all over the place,
# MAGIC    because people are assessed and then start when there is a place for them. A tight lag
# MAGIC    distribution means something is running to a schedule, and whether that schedule is a
# MAGIC    well run programme or something else is not a question claims data can answer.
# MAGIC 2. **Who went through the front door and never started treatment at all?** With the dollars.
# MAGIC    That is the opposite question and it deserves the same column.
# MAGIC 3. **Who started treatment here with no front door service from anybody?** That asks who
# MAGIC    decided they were eligible.
# MAGIC 4. **How often did the same entity bill the H0002 screening and the treatment it screened
# MAGIC    for?** H0002 specifically determines eligibility for admission to a treatment programme.

# COMMAND ----------

# Front door services from ANY provider, for the members these providers treated, so question 3
# can tell "nobody assessed this member" from "somebody else assessed them".
if N_TARGET_MEMBERS:
    _fd_raw = spark.table(SERVICE_TABLE).filter(
        F.upper(F.trim(F.coalesce(F.col(PROC_COL).cast("string"), F.lit(""))))
         .isin(sorted(FRONT_DOOR_CODES)))
    df_entity_front_door = (prepare(_fd_raw)
                            .join(F.broadcast(_members_df), MEMBER_COL, "inner")
                            .cache())
    N_ENTITY_FRONT_DOOR = df_entity_front_door.count()
else:
    df_entity_front_door = df_line.filter(F.lit(False))
    N_ENTITY_FRONT_DOOR = 0

# The target's own front door services and its own treatment services.
_own_fd = (df_line.filter(F.col(PROC_COL).isin(sorted(FRONT_DOOR_CODES)))
           .groupBy(MEMBER_COL)
           .agg(F.min("srv_date").alias("first_front_door_date"),
                F.round(F.sum("paid"), 2).alias("front_door_paid"),
                F.round(F.sum("qty"), 2).alias("front_door_units"),
                F.countDistinct("srv_date").alias("front_door_days"),
                _entity_list(PROC_COL).alias("front_door_codes"),
                _entity_list(BPROV_COL).alias("front_door_billing_entities")))
_first_fd_code = (df_line.filter(F.col(PROC_COL).isin(sorted(FRONT_DOOR_CODES)))
                  .groupBy(MEMBER_COL, "srv_date")
                  .agg(_entity_list(PROC_COL).alias("_codes_that_day")))
_own_fd = (_own_fd.join(
    _first_fd_code.withColumnRenamed("srv_date", "first_front_door_date"),
    [MEMBER_COL, "first_front_door_date"], "left")
    .withColumnRenamed("_codes_that_day", "first_front_door_code"))

_own_tx = (df_line.filter(F.col(PROC_COL).isin(sorted(TREATMENT_LEVEL_CODES)))
           .groupBy(MEMBER_COL)
           .agg(F.min("srv_date").alias("first_treatment_date"),
                F.round(F.sum("paid"), 2).alias("treatment_paid"),
                F.countDistinct("srv_date").alias("treatment_days"),
                _entity_list(PROC_COL).alias("treatment_codes"),
                _entity_list(BPROV_COL).alias("treatment_billing_entities")))
_first_tx_code = (df_line.filter(F.col(PROC_COL).isin(sorted(TREATMENT_LEVEL_CODES)))
                  .groupBy(MEMBER_COL, "srv_date")
                  .agg(_entity_list(PROC_COL).alias("_codes_that_day")))
_own_tx = (_own_tx.join(
    _first_tx_code.withColumnRenamed("srv_date", "first_treatment_date"),
    [MEMBER_COL, "first_treatment_date"], "left")
    .withColumnRenamed("_codes_that_day", "first_treatment_code"))

# Front door services from anybody at all, which answers question 3.
_any_fd = (df_entity_front_door.groupBy(MEMBER_COL)
           .agg(F.min("srv_date").alias("first_front_door_date_from_anybody"),
                _entity_list(SPROV_COL).alias("front_door_servicing_providers_anywhere"),
                _entity_list(BPROV_COL).alias("front_door_billing_entities_anywhere"),
                F.countDistinct(SPROV_COL).alias("distinct_front_door_providers_anywhere")))

funnel = (_own_fd.join(_own_tx, MEMBER_COL, "outer").join(_any_fd, MEMBER_COL, "outer")
          .withColumn("lag_days_from_the_front_door_to_the_first_treatment",
                      F.when(F.col("first_front_door_date").isNotNull() &
                             F.col("first_treatment_date").isNotNull() &
                             (F.col("first_treatment_date") >= F.col("first_front_door_date")),
                             F.datediff("first_treatment_date", "first_front_door_date")))
          .withColumn("treatment_followed_the_front_door",
                      F.col("lag_days_from_the_front_door_to_the_first_treatment").isNotNull() &
                      (F.col("lag_days_from_the_front_door_to_the_first_treatment") <=
                       F.lit(FUNNEL_MAX_LAG_DAYS)))
          .withColumn("the_same_entity_billed_both_halves",
                      F.col("front_door_billing_entities").isNotNull() &
                      F.col("treatment_billing_entities").isNotNull() &
                      (F.size(F.array_intersect(
                          F.split(F.col("front_door_billing_entities"), ", "),
                          F.split(F.col("treatment_billing_entities"), ", "))) > 0))
          .withColumn("what_this_member_is",
                      F.when(F.col("treatment_followed_the_front_door"),
                             F.lit("front door then treatment, both from this provider"))
                       .when(F.col("first_front_door_date").isNotNull() &
                             F.col("first_treatment_date").isNull(),
                             F.lit("front door from this provider and NO treatment from it at all"))
                       .when(F.col("first_front_door_date").isNull() &
                             F.col("first_treatment_date").isNotNull() &
                             F.col("first_front_door_date_from_anybody").isNull(),
                             F.lit("treatment from this provider with NO front door service from "
                                   "ANYBODY in this table"))
                       .when(F.col("first_front_door_date").isNull() &
                             F.col("first_treatment_date").isNotNull(),
                             F.lit("treatment from this provider, front door service from a "
                                   "different provider"))
                       .otherwise(F.lit("front door and treatment, but the treatment came first")))
          .cache())

funnel_df = (funnel.orderBy("lag_days_from_the_front_door_to_the_first_treatment")
             .limit(DETAIL_ROWS).toPandas())

funnel_summary_df = (
    funnel.groupBy("what_this_member_is")
    .agg(F.countDistinct(MEMBER_COL).alias("members"),
         F.round(F.sum(F.coalesce(F.col("front_door_paid"), F.lit(0.0))), 2)
          .alias("front_door_paid"),
         F.round(F.sum(F.coalesce(F.col("treatment_paid"), F.lit(0.0))), 2)
          .alias("treatment_paid"),
         F.round(F.avg("lag_days_from_the_front_door_to_the_first_treatment"), 1)
          .alias("mean_lag_days"),
         F.expr("percentile_approx(lag_days_from_the_front_door_to_the_first_treatment, 0.5)")
          .alias("median_lag_days"))
    .orderBy(F.desc("members")).toPandas())

# The lag distribution, reported whole. A tight distribution means something is running to a
# schedule. Claims data cannot say whether that schedule is a well run programme or anything else.
_lags = (funnel.filter(F.col("treatment_followed_the_front_door"))
         .select("lag_days_from_the_front_door_to_the_first_treatment").toPandas())
if len(_lags):
    _lv = pd.to_numeric(_lags["lag_days_from_the_front_door_to_the_first_treatment"]).dropna()
    _mode = _lv.mode()
    _modal = float(_mode.iloc[0]) if len(_mode) else None
    funnel_lag_distribution_df = (_lv.value_counts().rename_axis("lag_days")
                                  .reset_index(name="members").sort_values("lag_days")
                                  .reset_index(drop=True))
    funnel_lag_distribution_df["share_of_members_who_went_on_to_treatment"] = [
        _rate(v, len(_lv)) for v in funnel_lag_distribution_df.members]
    funnel_lag_stats_df = pd.DataFrame([
        ["Members who went from the front door to treatment with this provider", int(len(_lv))],
        ["Mean lag in days", round(float(_lv.mean()), 2)],
        ["Median lag in days", round(float(_lv.median()), 2)],
        ["Standard deviation of the lag in days", round(float(_lv.std(ddof=0)), 2)],
        ["Shortest lag in days", int(_lv.min())],
        ["Longest lag in days", int(_lv.max())],
        ["Most common lag in days", _modal],
        ["Share of members sitting on that one lag value",
         _rate(int((_lv == _modal).sum()), len(_lv)) if _modal is not None else None],
        ["How to read this",
         "Ordinary clinical intake produces lags spread out, because people are assessed and then "
         "start when there is a place for them. A tight distribution, with a large share of "
         "members on a single lag value, means something is running to a schedule. Whether that "
         "schedule is a well run programme or something else is not a question claims data can "
         "answer, and this sheet does not try to."],
    ], columns=["Item", "What it is"])
else:
    funnel_lag_distribution_df = pd.DataFrame()
    funnel_lag_stats_df = pd.DataFrame()
    _modal = None

# H0002 specifically: the same entity billed the screening that determines eligibility for
# admission AND the treatment that was admitted to.
_h0002_members = (df_line.filter(F.col(PROC_COL) == F.lit("H0002"))
                  .groupBy(MEMBER_COL)
                  .agg(F.min("srv_date").alias("first_H0002_date"),
                       F.round(F.sum("paid"), 2).alias("H0002_paid"),
                       F.round(F.sum("qty"), 2).alias("H0002_units"),
                       _entity_list(BPROV_COL).alias("H0002_billing_entities")))
h0002_screening_df = (
    _h0002_members.join(_own_tx, MEMBER_COL, "left")
    .withColumn("the_same_entity_billed_the_screening_and_the_treatment",
                F.col("treatment_billing_entities").isNotNull() &
                (F.size(F.array_intersect(F.split(F.col("H0002_billing_entities"), ", "),
                                          F.split(F.col("treatment_billing_entities"), ", "))) > 0))
    .withColumn("lag_days_from_the_screening_to_the_first_treatment",
                F.when(F.col("first_treatment_date").isNotNull() &
                       (F.col("first_treatment_date") >= F.col("first_H0002_date")),
                       F.datediff("first_treatment_date", "first_H0002_date")))
    .withColumn("what_this_is",
                F.lit("H0002 is the screening that determines eligibility for admission to a "
                      "treatment programme. Where the same entity bills the screening and the "
                      "treatment it screened for, that entity decided who was admitted to its own "
                      "programme. This is COMMON AND LEGITIMATE in integrated care and is "
                      "reported so an investigator can see how often it happens here, not as an "
                      "impropriety."))
    .orderBy(F.desc("H0002_paid"))
    .limit(DETAIL_ROWS).toPandas())
H0002_SAME_ENTITY_N = int(
    h0002_screening_df["the_same_entity_billed_the_screening_and_the_treatment"].sum()) \
    if len(h0002_screening_df) else 0

FUNNEL_AVAILABLE = bool(len(funnel_df))
_mark("Part M, the funnel from the front door to treatment", FUNNEL_AVAILABLE,
      "these providers billed neither a front door code nor a treatment code")
print(f"Funnel: {len(funnel_df)} members with a front door service or a treatment service from "
      f"these providers.")
if len(funnel_summary_df):
    print(funnel_summary_df.to_string(index=False))
if _modal is not None:
    print(f"Most common lag from the front door to the first treatment: {_modal:g} days, carrying "
          f"{_rate(int((_lv == _modal).sum()), len(_lv)):.1%} of the members who went on to "
          f"treatment.")
print(f"Members whose H0002 screening and treatment were billed by the same entity: "
      f"{H0002_SAME_ENTITY_N} of {len(h0002_screening_df)}. That is common and legitimate in "
      f"integrated care and is reported so it can be seen, not as an impropriety.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 23. Part F. Who they work with, and whether the two are connected
# MAGIC
# MAGIC When more than one provider is named, this section answers the obvious question directly. When
# MAGIC only one is named, the overlap sheets say so instead of being empty.
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
print("\nOnly one provider was named, so there is no member overlap to report."
      if len(TARGETS) < 2 else
      f"\nMembers seen by every one of the {len(TARGETS)} providers: {len(SHARED_MEMBERS)}")

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

if len(TARGETS) < 2:
    print("Only one provider was named, so there is no shared billing entity or shared referring "
          "provider to report.")
else:
    print(f"Billing entities every provider bills under: {len(SHARED_ENTITIES)}"
          + (f" ({', '.join(SHARED_ENTITIES[:10])})" if SHARED_ENTITIES else ""))
    print(f"Referring providers who send to all of them: {len(SHARED_REFERRERS)}"
          + (f" ({', '.join(SHARED_REFERRERS[:10])})" if SHARED_REFERRERS else ""))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 24. Part G. Peer comparison, and why to distrust it
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
]
# The focus diagnosis measure exists only when a diagnosis list was supplied. Leaving it in with
# an empty list would report every provider at zero and a percentile over a column of zeros is
# noise that reads like a result.
if HAVE_FOCUS_DIAGS:
    PEER_MEASURES.append(
        ("focus_diagnosis_share_of_members",
         "Share of members carrying a focus primary diagnosis"))
else:
    print("No focus diagnoses were supplied, so the focus diagnosis peer measure is dropped. A "
          "percentile over a column of zeros is noise that reads like a result.")
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
# MAGIC ## 25. What is new since the last run
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

# The new rule sheets. Each one is keyed on whatever identifies the finding on that sheet, and
# none of the keys is ever shortened: a truncated key collides and a collision silently marks a
# real finding as one that was already seen.
one_per_diem_df, _k = mark_new(one_per_diem_df, "one_per_diem", BPROV_COL, MEMBER_COL, "srv_date")
_all_keys.append(_k)
level_conflicts_df, _k = mark_new(level_conflicts_df, "level_conflict", "entity_a", MEMBER_COL,
                                  "srv_date", "level_b")
_all_keys.append(_k)
h2012_tier_df, _k = mark_new(h2012_tier_df, "h2012_tier", SPROV_COL, MEMBER_COL, "srv_date")
_all_keys.append(_k)
doc_expanded_df, _k = mark_new(doc_expanded_df, "doc_expanded", SPROV_COL, MEMBER_COL, "srv_date",
                               PROC_COL)
_all_keys.append(_k)
s9480_once_df, _k = mark_new(s9480_once_df, "s9480_once", SPROV_COL, MEMBER_COL, "srv_date")
_all_keys.append(_k)
s9480_pa_df, _k = mark_new(s9480_pa_df, "s9480_pa", SPROV_COL, MEMBER_COL, "srv_date")
_all_keys.append(_k)
s9480_intensity_df, _k = mark_new(s9480_intensity_df, "s9480_intensity", "billing_entities",
                                  MEMBER_COL, "week_start")
_all_keys.append(_k)
h0018_pos_df, _k = mark_new(h0018_pos_df, "h0018_pos", SPROV_COL, MEMBER_COL, "srv_date")
_all_keys.append(_k)
bed_hold_df, _k = mark_new(bed_hold_df, "bed_hold", SPROV_COL, MEMBER_COL, "contract_year_start")
_all_keys.append(_k)
edu_daily_cap_df, _k = mark_new(edu_daily_cap_df, "edu_daily_cap", SPROV_COL, MEMBER_COL,
                                "srv_date", PROC_COL)
_all_keys.append(_k)
edu_rolling_df, _k = mark_new(edu_rolling_df, "edu_rolling", "billing_entities", MEMBER_COL,
                              "peak_window_ends")
_all_keys.append(_k)
h0001_year_df, _k = mark_new(h0001_year_df, "h0001_year", "billing_entities", MEMBER_COL,
                             "the_calendar_year_that_was")
_all_keys.append(_k)
h0002_year_df, _k = mark_new(h0002_year_df, "h0002_year", "billing_entities", MEMBER_COL,
                             "the_calendar_year_that_was")
_all_keys.append(_k)

KEYS_THIS_RUN = pd.concat([k for k in _all_keys if len(k)], ignore_index=True) \
    if any(len(k) for k in _all_keys) else pd.Series(dtype=str)
NEW_FINDINGS = 0
for _d in (doc_over_df, h0004_perdiem_df, t1016_h0018_df, multi_per_diem_df, per_diem_units_df,
           one_per_diem_df, level_conflicts_df, h2012_tier_df, doc_expanded_df, s9480_once_df,
           s9480_pa_df, h0018_pos_df, edu_daily_cap_df, edu_rolling_df, h0001_year_df,
           h0002_year_df):
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
# MAGIC ## 26. The Criteria sheet, the Data_Availability sheet and the workbook
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
     "Every claim line for these providers is loaded, whatever the code and whatever the "
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
    # One row per focus code, built from whatever codes were asked for rather than from a fixed
    # list, so the Criteria sheet cannot silently describe the previous request's codes.
    *[[f"Unit basis, {c}",
       f"{CODE_BASIS.get(c, 'UNCLASSIFIED')}. "
       + FOCUS_CODE_NOTE.get(c, "No note is recorded for this code in this notebook. Its unit "
                                "basis comes from the dictionary in section 2 and should be "
                                "confirmed against the current fee schedule before any number "
                                "about it is quoted.")]
      for c in FOCUS_CODES],
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
    ["Minutes that rest on a national descriptor rather than on an AHCCCS statement",
     "Most of the durations in this notebook come from AHCCCS: 15 minutes a unit, 60 minutes a "
     "unit, one day, one session. For "
     + ", ".join(sorted(DURATION_FROM_NATIONAL_DESCRIPTOR_ONLY)) +
     " the 30 minutes comes from the national HCPCS descriptor of the code and AHCCCS has "
     "published no minutes a unit statement for it. Those minutes ARE included in the day hour "
     "totals, and the day sheets carry a column showing how many of each day's minutes came from "
     "them, together with the share of the whole. Where that share is more than a sliver, read "
     "the hour total with that much less confidence. The two published limits on those codes, "
     "4 units a day and 60 units in any 30 days, are UNIT counts and do not depend on the "
     "duration at all, so the tests built on them are exact."],
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
     "properly the per diem side loads every per diem line for the members these providers "
     "treated, from any provider and any billing entity, and then decides on the billing entity. "
     "Scoping the per diem side to the targets' own entities instead would make the permitted "
     "cross entity pattern invisible rather than counted."],
    ["One per diem per member per day, Part E test 4",
     "Published rule, effective 2024-02-06 and restated in the Covered Behavioral Health Services "
     "Guide dated 2024-10-01. Collisions involving a code with an unverified unit basis are listed "
     "but carry no finding, because nobody has published how many days one of those units covers."],
    ["What Part E cannot see",
     "It only sees what these providers and their billing entities billed. A member who "
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

# ── Criteria rows for everything added for a per diem, organisation shaped request ───────────
criteria = pd.concat([criteria, pd.DataFrame([
    ["WHICH INSTRUMENT THIS RUN USED", f"{FOCUS_SHAPE} set. {FOCUS_SHAPE_BASIS}."],
    ["Why the shape matters",
     "A timed code carries minutes and a per diem does not. One unit of a per diem is one day of "
     "care whatever happened inside it, so asking a per diem how many hours it represents gives "
     "zero. A workbook that led with hours against a per diem provider would report near zero "
     "hours against somebody billing millions of dollars of residential care, and a reader would "
     "conclude wrongly that nothing was happening. So the shape of the focus set decides which "
     "section leads: the day for a timed set, the census and the episodes for a per diem set."],
    ["Status of the hours in Part D", HOURS_INSTRUMENT_STATUS + ". " + HOURS_INSTRUMENT_WHY],
    ["Person or organisation, and what it costs", ORG_DAY_WARNING],
    ["The focus lens is reported in two blocks",
     "Treatment and level of care codes say where a member is being treated and how intensively. "
     "Assessment, screening and education codes are the front door that decides who is admitted. "
     "They answer different questions, so they are reported separately and a number from one "
     "block should not be compared against a number from the other. Part M describes the order in "
     "which this provider billed them, which is a description of billing order and nothing more."],
    ["Focus diagnoses in this run",
     ", ".join(FOCUS_DIAGS) if HAVE_FOCUS_DIAGS else
     "NONE WERE SUPPLIED. Every diagnosis sheet says so rather than showing an empty table, the "
     "code by diagnosis cross tabulation is not produced at all rather than written as a grid of "
     "zeros, and the focus diagnosis measure is dropped from the peer comparison. A grid of zeros "
     "reads as 'checked, nothing found' when the truth is that nobody asked the question."],
    ["Rule 4 and rule 6 overlap, so do NOT add them together",
     "Both test the one per diem per member per day rule and they will often name the same "
     "member days. Rule 4 is the older test and groups by the SERVICING provider; rule 6 groups "
     "by the BILLING ENTITY, which is the right key for an organisation whose clinicians bill "
     "under several servicing identifiers, and it reports the dollars beyond the largest per "
     "diem and the cross entity version separately. For a per diem focus set read rule 6 and "
     "treat rule 4 as the earlier view of the same thing. Adding the two counts together would "
     "double count every member day they share."],
    ["ONE PER DIEM PER MEMBER PER DAY, the central test for a per diem set",
     "Published rule, effective 2024-02-06 and codified in the Covered Behavioral Health Services "
     "Guide dated 2024-10-01. The rule is written at the member day level: one per diem per "
     "member per day, whoever billed it. This notebook FLAGS the member days where ONE billing "
     "entity billed two or more per diem codes, because that is the version this provider can be "
     "held to and the version a per claim edit should have caught. Where the second per diem came "
     "from a DIFFERENT entity the row is counted in a column of its own and never flagged: this "
     "provider did not bill the other entity's per diem. Those cross entity days still matter, "
     "and the member journey screen is the notebook that does cross provider member day "
     "arithmetic. The exposure reported is everything paid beyond the single largest per diem on "
     "that member day, because only one per diem can be right."],
    ["THE LEVEL OF CARE CONFLICT MATRIX",
     "; ".join(f"{r['level_a']} against {r['level_b']}: same entity, {r['same_entity']} "
               f"Different entity, {r['cross_entity']}" for r in CONFLICT_MATRIX)
     + " The matrix is also written out as a table on the E_Conflict_Matrix sheet so every "
       "finding can be checked against the rule that produced it."],
    ["H2012 is NOT a per diem",
     f"One unit of H2012 is sixty minutes, up to five hours a day. FFS chapter 19 sets the day "
     f"treatment family as H2012 per hour up to 5 hours, then H2015 per 15 minutes above 5 hours "
     f"to 10 hours, then H2016 per diem. More than {H2012_TIER_MAX_UNITS:g} units of H2012 in one "
     f"member day means the service exceeded the tier it was billed under. Treating H2012 as a "
     f"per diem because the codes around it are per diems would turn five ordinary hours of day "
     f"treatment into five days of care."],
    ["The 2023-07-17 expanded documentation list",
     f"A SECOND documentation control, separate from the 2023-05-03 one, with its own code list "
     f"and its own thresholds: "
     + "; ".join(f"{c} {'any units' if t == 0 else f'above {t} units'}"
                 for c, t in sorted(DOC_EXPANDED_THRESHOLDS.items()))
     + f". Not testable on that list: {', '.join(sorted(DOC_EXPANDED_NOT_TESTABLE))}, whose unit "
       f"basis is unverified. It keys on the claim RECEIPT date, like the 2023-05-03 one, and "
       f"there is no fallback to the service date. Status in this run: {EXPANDED_DOC_STATUS}"],
    ["S9480 prior authorization",
     f"Fee for service (AIHP, DDD-THP, TRBHA, ALTCS) from {S9480_PA_FFS_DATE}; Arizona Complete "
     f"Health managed care from {S9480_PA_MCO_DATE}. IHS and 638 facilities are EXEMPT. The line "
     f"of business is matched on these tokens: {', '.join(FFS_LOB_TOKENS)} for fee for service "
     f"and {', '.join(MCO_PA_LOB_TOKENS)} for managed care. A line of business this notebook does "
     f"not recognise is given the LATER fee for service date, which reports fewer findings rather "
     f"than more. Status in this run: {PA_TEST_STATUS}"],
    ["Minimum intensity: one rule for S9480, and NO rule for H0035",
     f"S9480 has a published minimum of at least {S9480_MIN_DAYS_PER_WEEK} DAYS a week. A claim "
     f"can count days, so that is tested as published and it is a rule. S9480 also requires 3 "
     f"hours a day, which a per diem claim cannot show, so every row on that sheet says it is a "
     f"QUESTION for the service record rather than a finding. H0035 has a published minimum of "
     f"{H0035_MIN_HOURS_PER_WEEK:g} HOURS a week, and a per diem claim carries no hours at all, "
     f"so THAT MINIMUM CANNOT BE TESTED AND THIS NOTEBOOK DOES NOT TRY. An earlier version turned "
     f"it into a number of days by dividing by the longest day of programming it was willing to "
     f"believe. That divisor was ours and not AHCCCS's, so the threshold was ours too, and a row "
     f"built on it would have sat on the rule sheets beside nine published rules where a reader "
     f"skimming them would take it for one. It has been removed: there is no H0035 intensity "
     f"rule, no H0035 finding count and no H0035 row on the rule summary. What replaced it is the "
     f"plain DISTRIBUTION of H0035 days per member per week, on its own sheet, with no threshold "
     f"on it, and with the published {H0035_MIN_HOURS_PER_WEEK:g} hours printed beside it as "
     f"context for reading it. The service records are what answer whether the twenty hours were "
     f"met; the distribution is the closest claims can get to asking."],
    ["Calendar edge weeks are EXCLUDED from the S9480 rule, not merely labelled",
     f"A programme that starts on a Friday shows one day in its first week. That is where the "
     f"calendar falls, not a short programme. An earlier version put a column on the row saying "
     f"so, which is not enough: a column does not travel with a row that has been pasted into a "
     f"referral. A week is now treated as a calendar edge when it is the first or last week of a "
     f"run of S9480 activity that is LONGER THAN ONE WEEK, so the programme carries on across the "
     f"boundary, or when it is the first or last week of the extract, so the record is cut off by "
     f"the lookback. A run of exactly ONE week is NOT an edge: that week holds the whole of that "
     f"stretch of care, so its day count is complete rather than an artefact, and excluding those "
     f"would throw away the clearest evidence the test exists to find. Edge weeks are "
     f"{'INCLUDED in this run by the s9480_include_edge_weeks setting' if S9480_INCLUDE_EDGE_WEEKS else 'EXCLUDED by default'}"
     f", kept on the E_Min_Intensity_Edges sheet so nothing is hidden, and the number removed is "
     f"reported on the rule summary. In this run that number is {S9480_EDGE_WEEKS_REMOVED}."],
    ["H0018 place of service",
     f"Behavioural health residential facilities must bill place of service {BHRF_POS_CODE} from "
     f"{POS_56_DATE}. Place of service values are compared with leading zeros removed, so 056 and "
     f"56 are the same place and a formatting difference cannot manufacture a finding. Status in "
     f"this run: {POS_TEST_STATUS}"],
    ["Provider type against the codes billed", PROVIDER_TYPE_STATEMENT],
    ["Bed hold and therapeutic leave",
     f"Revenue code {REV_HOME_PASS} is home pass, which is therapeutic leave, and {REV_BED_HOLD} "
     f"is bed hold. Members 21 and over: {LEAVE_LIMIT_21_PLUS} days of therapeutic leave and "
     f"{BED_HOLD_LIMIT_21_PLUS} days of bed hold per contract year. Under 21: "
     f"{COMBINED_LIMIT_UNDER_21} days combined. The contract year used here starts on 1 "
     f"{date(2000, CONTRACT_YEAR_START_MONTH, 1).strftime('%B')}. A member who turned 21 inside a "
     f"contract year is flagged only if the counts exceed BOTH limits, and must be checked by "
     f"hand either way. Status in this run: {LEAVE_TEST_STATUS}"],
    ["98960 and the education and training family",
     f"Hard limits of {EDU_DAILY_UNIT_CAP:g} units a day and {EDU_ROLLING_UNIT_CAP:g} units in "
     f"any rolling {EDU_ROLLING_WINDOW_DAYS} days, effective {EDU_LIMIT_DATE} and keyed on the "
     f"date of SERVICE. These are hard limits, not documentation requirements. BOTH ARE UNIT "
     f"COUNTS, so neither test needs a view about what one unit means and both are exact. The 30 "
     f"minutes attached to this code comes from the national HCPCS descriptor rather than from an "
     f"AHCCCS statement; it is used only to put the code into the day hours total and the day "
     f"sheets report it separately. The daily cap is applied PER CODE, which reports fewer rows "
     f"than applying it across the family, and the family total is shown beside each row."],
    ["The rolling window is general machinery",
     f"A member can stay under a daily cap every single day and still be far above a monthly one, "
     f"so a rolling window has to be computed its own way. It is computed over CALENDAR days, not "
     f"over billed days: a row based window would count the last N days somebody was billed for "
     f"and would let a member billed on 40 scattered days inside a month slip under a 30 day "
     f"window. The function is written generally because several other AHCCCS limits take this "
     f"shape, including 600 hours a benefit year on respite and 160 units a week on the U3 "
     f"modifier."],
    ["PER YEAR LIMITS WHOSE YEAR BASIS AHCCCS HAS NOT STATED",
     f"The Covered Behavioral Health Services Guide limits H0001 to "
     f"{PER_YEAR_LIMITS['H0001']:g} a year and H0002 to {PER_YEAR_LIMITS['H0002']:g} a year for "
     f"members aged {H0002_MIN_AGE} and over. IT DOES NOT SAY WHICH YEAR. The shared "
     f"specification records a benefit year of 1 October to 30 September for respite hours, so "
     f"that is the likely basis, but it is not stated for these codes and this notebook does not "
     f"guess. BOTH the calendar year and the 1 "
     f"{date(2000, BENEFIT_YEAR_START_MONTH, 1).strftime('%B')} benefit year are counted and "
     f"reported side by side. A member above the limit on one basis and below it on the other is "
     f"marked AMBIGUOUS and is NOT a finding: a member at 13 assessments on one basis and 11 on "
     f"the other is a question about which year AHCCCS meant. A finding on either of these sheets "
     f"depends on which year basis turns out to be right, and nothing here settles that."],
    ["THE DAILY CENSUS, and what it must be compared against",
     "For every calendar day between the first and last service date, how many distinct members "
     "this provider had on a per diem. Every day gets a row, including the days with nobody in "
     "care, because a census curve with the empty days missing looks completely different from "
     "the real one; the mean over every calendar day and the mean over days with somebody in care "
     "are both reported and both labelled. Codes whose unit basis AHCCCS has never published "
     "contribute NOTHING to any census figure and are reported in a column of their own. THE "
     "NUMBER TO COMPARE A CENSUS AGAINST IS THE FACILITY'S LICENSED CAPACITY, which is the "
     "state's own record in provider enrolment and licensing. This notebook does not hold it and "
     "does not guess at it. The state's forensic audit compared billing against a licensed "
     "capacity of eight in its finding F12, and that comparison is the job this sheet is meant to "
     "make possible rather than one it performs."],
    ["Stays, length of stay and returns",
     f"A stay is a member's consecutive billed days on one per diem code. A gap of more than "
     f"{STAY_GAP_DAYS} day(s) starts a new stay, which matches the member journey screen exactly. "
     f"A return inside {READMIT_WINDOW_DAYS} days is counted as a readmission and "
     f"{READMIT_MIN_RETURNS} or more of them is reported as returning repeatedly. Codes with no "
     f"published unit basis build no stays at all."],
    ["THE LONG STAY CUT IS DESCRIPTIVE, NOT A PUBLISHED LIMIT",
     f"AHCCCS publishes no maximum length of stay for H0018 or for any of these per diems, so "
     f"this notebook does not invent one. What it does is report the whole distribution of this "
     f"provider's own stays and mark the tail at the {LONG_STAY_TAIL_PCT:.0%} point OF THIS "
     f"PROVIDER'S OWN DISTRIBUTION. That says a stay is long compared with this provider's other "
     f"stays. It does NOT say the stay was too long. Where a code has fewer than "
     f"{MIN_STAYS_FOR_A_TAIL} stays the tail is not marked at all, because a percentile of a "
     f"handful of stays is noise. H0018 gets its own sheet because the code means SHORT TERM "
     f"residential, so a long unbroken run of it is a question on the face of the code."],
    ["Servicing providers under the organisation",
     "Every line billed by the billing entities that appear on the target's claims, grouped by "
     "the servicing provider on it. Where an organisation's own identifier is the servicing "
     "provider on its own claims, the claim says the organisation performed the service rather "
     "than naming the person who did. That is the pattern the state's forensic audit raised about "
     "organisational and individual provider numbers, and it is why a day total for that "
     "identifier cannot be read as one person's day. It is a description of how the claims were "
     "filled in, not a finding: the enrolment record settles whether it is correct. Status in "
     "this run: " + ORG_SCOPE_STATUS],
    ["THE MONTHLY DOLLAR FIGURE IS AN AUDITOR'S RECOMMENDATION, NOT AN AHCCCS CONTROL",
     f"Paid dollars by month are reported beside ${MONTHLY_PAID_REFERENCE:,.0f}. That figure is "
     f"what the state's forensic audit proposed in its recommendation 12 as a review trigger for "
     f"Integrated Clinics. It is NOT a live AHCCCS control. Nothing was denied for crossing it, "
     f"nothing is owed for crossing it, and a month above it is not a finding. It is reported for "
     f"the same reason the cluster screen reports it: so a reviewer who has read the audit can "
     f"see where this provider sits against the number the auditors had in mind."],
    ["The S9480 pricing era check",
     f"AHCCCS paid S9480 at {BY_REPORT_RATIO:.4f} of covered billed charges under By Report "
     f"pricing, replaced by a fixed ${S9480_FIXED_RATE:,.2f} effective {S9480_FIXED_RATE_DATE}. "
     f"Dividing paid by billed on each S9480 line and watching when the ratio stops clustering "
     f"dates that change in this extract. A month counts as clustering when at least "
     f"{BY_REPORT_CLUSTER_SHARE:.0%} of its lines sit within {BY_REPORT_TOLERANCE} of the ratio. "
     f"THIS IS A CHECK ON THE EXTRACT, NOT A FINDING AGAINST THE PROVIDER. Status in this run: "
     + (f"the ratio stops clustering in {PRICING_CHANGE_MONTH}" if PRICING_CHANGE_MONTH is not None
        else PRICING_ERA_STATUS if not BILL_COL
        else "the ratio never clusters and then stops in this extract, so the change could not be "
             "dated from the data")],
    ["THE FUNNEL IS A DESCRIPTION OF BILLING ORDER",
     f"Part M reports, for each member, the first front door service this provider billed "
     f"({', '.join(sorted(FRONT_DOOR_CODES))}: assessment, screening for eligibility for "
     f"admission, and mental health assessment) and the first treatment service it billed "
     f"afterwards, with the lag in days. It also reports members who went through the front door "
     f"and never started treatment, members who started treatment here with no front door service "
     f"from anybody, and how often the same entity billed the H0002 screening and the treatment it "
     f"screened for. A first treatment service more than {FUNNEL_MAX_LAG_DAYS} days after the "
     f"front door is not reported as following it. THIS IS A DESCRIPTION OF BILLING ORDER AND "
     f"NOTHING MORE. Integrated care providers assess and then treat their own patients every "
     f"day, entirely properly, and that is the normal case rather than the odd one. A tight lag "
     f"distribution means something is running to a schedule, and whether that schedule is a well "
     f"run programme or something else is not a question claims data can answer."],
], columns=["Item", "What it is"])], ignore_index=True)

_susp_uniform = {p: SUSP_VALUES.get(p) for p in TARGETS}
data_availability_notes = pd.DataFrame([
    ["Claim lines loaded for the providers under review", f"{N_LINES:,}"
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

# Every new test and every new section records its own status as it runs, so nothing can be
# quietly absent from this sheet.
for _section, _st in NEW_TEST_STATUS.items():
    rule_status_rows.append({
        "section_of_the_workbook": _section,
        "status": "ran" if _st == "ran" else "NOT RUN",
        "reason_if_not_fully_available": "" if _st == "ran" else _st.replace("NOT RUN: ", "")})
rule_status_rows.append({
    "section_of_the_workbook": "Which instrument fits this focus set",
    "status": "ran",
    "reason_if_not_fully_available": f"{FOCUS_SHAPE} set. {HOURS_INSTRUMENT_STATUS}"})
rule_status_rows.append({
    "section_of_the_workbook": "Part B, the focus diagnosis half",
    "status": "ran" if HAVE_FOCUS_DIAGS else "NOT RUN",
    "reason_if_not_fully_available": "" if HAVE_FOCUS_DIAGS else
    "no focus diagnoses were supplied with the request, so there was nothing to look for. The "
    "diagnosis sheets say so rather than showing an empty table, and the diagnosis measure was "
    "dropped from the peer comparison."})
_assert_unique([r["section_of_the_workbook"] for r in rule_status_rows], "analysis status rows")
_bad = [r for r in rule_status_rows if r["status"] != "ran"]
print("\nAnalysis status by section:")
if _bad:
    for r in _bad:
        print(f"   {r['status']:<9} {r['section_of_the_workbook']}: {r['reason_if_not_fully_available']}")
else:
    print("   every section of the workbook ran with every column it wanted")

# ── The rule summary, and why its counts must NOT be added together ──────────────────────────
#
# Several of these tests name the same member day. Rules 4 and 6 both test the one per diem per
# member per day rule, one keyed on the servicing provider and one on the billing entity. Rule 7
# fires on a member day that often breaks the per diem rule as well. Rule 5 and rule 10 overlap
# on S9480. Adding the counts together produces a finding total larger than the number of member
# days involved, and a double counted finding total in a referral discredits everything beside it.
#
# So each row says which other rules name the same findings, how many of its rows nothing else
# names, and the sheet carries a deduplicated total of distinct member days across every test.
# The overlaps are COMPUTED from this run's own findings rather than asserted, so they are true
# for the data in front of the reader rather than true in general.
def _finding_keys(df, kind, *cols):
    """The identifying key of every row on a finding sheet, as a set. Nothing is ever shortened:
    a truncated key collides and a collision would understate the deduplicated total."""
    if df is None or not len(df) or any(c not in df.columns for c in cols):
        return set()
    return {(kind,) + tuple(str(r[c]) for c in cols) for _, r in df.iterrows()}


_MD = "member day"
RULE_KEY_SETS = {
    "Rule 1: above the documentation threshold, claims received on or after the cutoff":
        (_MD, _finding_keys(doc_over_df[doc_over_df.get(
            "over_threshold_on_lines_received_after_cutoff", False) == True]  # noqa: E712
            if len(doc_over_df) else doc_over_df, _MD, MEMBER_COL, "srv_date")),
    "Rule 2: H0004 on a residential or IOP per diem day, same billing entity":
        (_MD, _finding_keys(h0004_perdiem_df, _MD, MEMBER_COL, "srv_date")),
    "Rule 3: T1016 on an H0018 day, same billing entity":
        (_MD, _finding_keys(t1016_h0018_df, _MD, MEMBER_COL, "srv_date")),
    "Rule 4: two or more per diem codes for one member in one day":
        (_MD, _finding_keys(multi_per_diem_df, _MD, MEMBER_COL, "srv_date")),
    "Rule 5: a per diem code above one unit in one member day":
        (_MD, _finding_keys(per_diem_units_df, _MD, MEMBER_COL, "srv_date")),
    "Rule 6: one per diem per member per day, same billing entity":
        (_MD, _finding_keys(one_per_diem_df, _MD, MEMBER_COL, "srv_date")),
    "Rule 7: levels of care that cannot coexist, same billing entity":
        (_MD, _finding_keys(level_conflicts_df, _MD, MEMBER_COL, "srv_date")),
    "Rule 8: H2012 above the per hour tier in one member day":
        (_MD, _finding_keys(h2012_tier_df, _MD, MEMBER_COL, "srv_date")),
    "Rule 9: above the 2023-07-17 expanded documentation thresholds":
        (_MD, _finding_keys(doc_expanded_df[doc_expanded_df.get(
            "over_threshold_on_lines_received_after_cutoff", False) == True]  # noqa: E712
            if len(doc_expanded_df) else doc_expanded_df, _MD, MEMBER_COL, "srv_date")),
    "Rule 10: S9480 more than once in one member day":
        (_MD, _finding_keys(s9480_once_df, _MD, MEMBER_COL, "srv_date")),
    "Rule 11: S9480 with no prior authorization on or after the applicable date":
        (_MD, _finding_keys(s9480_pa_df, _MD, MEMBER_COL, "srv_date")),
    "Rule 14: H0018 with a place of service other than 56":
        (_MD, _finding_keys(h0018_pos_df, _MD, MEMBER_COL, "srv_date")),
    "Rule 17: 98960 above the daily unit cap":
        (_MD, _finding_keys(edu_daily_cap_df, _MD, MEMBER_COL, "srv_date")),
    "Rule 12: S9480 on too few days in a calendar week, a QUESTION":
        ("member week", _finding_keys(s9480_intensity_df, "member week", MEMBER_COL,
                                      "week_start")),
    "Rule 16: bed hold or therapeutic leave above the contract year limit":
        ("member contract year",
         _finding_keys(bed_hold_df[bed_hold_df.get("over_the_limit", False) == True]  # noqa: E712
                       if len(bed_hold_df) else bed_hold_df,
                       "member contract year", MEMBER_COL, "contract_year_start")),
    "Rule 18: 98960 above the rolling window unit cap":
        ("member", _finding_keys(edu_rolling_df, "member", MEMBER_COL)),
    "Rule 19: H0001 above 12 a year on at least one year basis":
        ("member", _finding_keys(h0001_year_df, "member", MEMBER_COL)),
    "Rule 20: H0002 above 24 a year on at least one year basis":
        ("member", _finding_keys(h0002_year_df, "member", MEMBER_COL)),
}
_MEMBER_DAY_KEYS = set()
for _k, (_kind, _ks) in RULE_KEY_SETS.items():
    if _kind == _MD:
        _MEMBER_DAY_KEYS |= _ks
DISTINCT_MEMBER_DAYS_ACROSS_RULES = len(_MEMBER_DAY_KEYS)
_OTHER_KEYS = {kind: set() for kind in {k for k, _ in RULE_KEY_SETS.values()} if kind != _MD}
for _k, (_kind, _ks) in RULE_KEY_SETS.items():
    if _kind != _MD:
        _OTHER_KEYS[_kind] |= _ks


def _overlaps(label):
    """Which other rules name the same findings as this one, and how many of its rows nothing
    else names. Computed from this run's findings, so it is true for the data in front of the
    reader rather than true in general."""
    kind, keys = RULE_KEY_SETS.get(label, (None, set()))
    if not keys:
        return "", None
    other_sets = [s2 for o, (k2, s2) in RULE_KEY_SETS.items() if o != label and k2 == kind]
    union_of_others = set().union(*other_sets) if other_sets else set()
    others = sorted(o for o, (k2, s2) in RULE_KEY_SETS.items()
                    if o != label and k2 == kind and (keys & s2))
    return ("; ".join(o.split(":")[0] for o in others) if others else "nothing else names these",
            len(keys - union_of_others))


rule_summary = pd.DataFrame(
    [{"test": k, "rows": v} for k, v in RULE_COUNTS.items()])
rule_summary["rows_numeric"] = pd.to_numeric(rule_summary["rows"], errors="coerce")
_FINDING_PREFIXES = tuple(f"Rule {i}:" for i in range(1, 21))
rule_summary["note"] = [
    RULE_NOTE_OVERRIDES[k] if k in RULE_NOTE_OVERRIDES
    else "the test could not run at all; see the Data_Availability sheet" if str(v) == "NOT RUN"
    else "a finding" if k.startswith(_FINDING_PREFIXES)
    else "a decoy or a permitted pattern, counted so you can see how much of it there is"
    for k, v in zip(rule_summary.test, rule_summary.rows)]
rule_summary["what_each_row_counts"] = [
    RULE_KEY_SETS.get(k, ("not a count of findings", None))[0] for k in rule_summary.test]
rule_summary["overlaps_with"] = [_overlaps(k)[0] for k in rule_summary.test]
rule_summary["rows_no_other_rule_names"] = [_overlaps(k)[1] for k in rule_summary.test]
if (rule_summary["rows_numeric"].fillna(0) == 0).all():
    rule_summary = pd.concat([rule_summary, pd.DataFrame([{
        "test": "EVERY TEST FOUND NOTHING OR COULD NOT RUN", "rows": 0, "rows_numeric": 0,
        "note": "A test that silently finds nothing is the failure mode to fear. Before concluding "
                "there is nothing here, check the Data_Availability sheet for missing columns and "
                "for tests marked NOT RUN, check that the provider IDs are right, and check that "
                "focus_only is off.",
        "what_each_row_counts": "", "overlaps_with": "", "rows_no_other_rule_names": None}])],
        ignore_index=True)

# The header rows, so the first thing read on this sheet is the warning not to add the counts up,
# and the deduplicated totals that should be used instead.
_header = [
    {"test": "HOW TO READ THIS SHEET: THE COUNTS BELOW ARE NOT ADDITIVE",
     "rows": "", "rows_numeric": None,
     "note": "Several of these tests name the SAME member day, so adding the counts together "
             "produces a finding total larger than the number of member days involved. Rules 4 "
             "and 6 both test the one per diem per member per day rule, one keyed on the "
             "servicing provider and one on the billing entity. Rule 7 fires on member days that "
             "often break the per diem rule as well. Use the deduplicated total in the next row "
             "when you need one number. The overlaps_with column on each row says which other "
             "rules name the same findings in THIS run.",
     "what_each_row_counts": "", "overlaps_with": "", "rows_no_other_rule_names": None},
    {"test": "USE THIS TOTAL: distinct member days named by any rule in this workbook",
     "rows": DISTINCT_MEMBER_DAYS_ACROSS_RULES,
     "rows_numeric": DISTINCT_MEMBER_DAYS_ACROSS_RULES,
     "note": "One member and one date counted once, however many rules named it. This is the "
             "number to quote. It covers only the tests whose rows ARE member days; the rows "
             "below it counted in member weeks, member contract years or members are listed "
             "separately because they cannot be added into a member day total.",
     "what_each_row_counts": _MD, "overlaps_with": "", "rows_no_other_rule_names": None},
]
for _kind in sorted(_OTHER_KEYS):
    _header.append({
        "test": f"Separate total: distinct {_kind}s named by any rule, which do NOT add into the "
                f"member day total",
        "rows": len(_OTHER_KEYS[_kind]), "rows_numeric": len(_OTHER_KEYS[_kind]),
        "note": f"A {_kind} is a different unit from a member day and the two cannot be added "
                f"together.",
        "what_each_row_counts": _kind, "overlaps_with": "", "rows_no_other_rule_names": None})
rule_summary = pd.concat([pd.DataFrame(_header), rule_summary], ignore_index=True)
print(rule_summary[["test", "rows", "what_each_row_counts", "overlaps_with",
                    "rows_no_other_rule_names"]].to_string(index=False))
print(f"\nDeduplicated: {DISTINCT_MEMBER_DAYS_ACROSS_RULES} distinct member days are named by at "
      f"least one rule in this workbook. The per rule counts above are NOT additive.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 27. Write the workbook
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
     f"An investigation of {len(TARGETS)} named servicing provider"
     + ("s" if len(TARGETS) != 1 else "") + ": "
     + "; ".join(f"{p} ({TARGET_NAMES_GIVEN.get(p) or 'no name supplied with the request'})"
                 for p in TARGETS)
     + f". {len(FOCUS_CODES)} procedure codes"
     + (f" and {len(FOCUS_DIAGS)} primary diagnoses are" if HAVE_FOCUS_DIAGS else " are")
     + " used as a lens."],
    ["Run date", str(TODAY)],
    ["Period covered", f"{DATA_START} to {DATA_END}" if DATA_START else "no claims found"],
    ["READ THIS FIRST: which instrument fits this request",
     f"The focus codes are a {FOCUS_SHAPE.upper()} set. {FOCUS_SHAPE_BASIS}. "
     + ("So the DAILY CENSUS and the EPISODES are the headline of this workbook, and the hours in "
        "Part D are NOT the right instrument. A per diem carries no minutes: one unit is one day "
        "of care whatever happened inside it. A provider billing nothing but per diems shows "
        "close to zero hours however much they are billing, and that is the arithmetic working "
        "correctly rather than a quiet provider. Open H_Census_Summary and I_LOS_Distribution "
        "before you open D_Day_Summary."
        if SHAPE_IS_PER_DIEM else
        "So the DAY is the headline of this workbook, as it always has been. Open D_Day_Summary "
        "after the profile.")],
    ["READ THIS SECOND: person or organisation", ORG_DAY_WARNING],
    ["Where to start",
     "Open 'A_Provider_Profile'. It has one column for each provider so you can compare them at a "
     "glance. Read the person_or_organisation row first: it decides whether a day total means one "
     "person's day or a whole organisation's. Then open 'E_Provider_Type_Check' for the single "
     "statement about what this provider type billed, and 'E_Rule_Summary' for the published rule "
     "tests, because those need no statistical argument."],
    ["The provider type statement", PROVIDER_TYPE_STATEMENT],
    ["The most important thing about these numbers",
     f"The {len(FOCUS_CODES)} focus codes"
     + (f" and the {len(FOCUS_DIAGS)} focus diagnoses are" if HAVE_FOCUS_DIAGS else " are")
     + " a LENS, not a filter. Every claim for "
     "these providers is in this workbook, whatever the code. That is on purpose. If the data "
     "were cut down to the focus codes, the hours in a day would come out too low, every "
     "percentage would be 100% by construction, and the same day conflict tests could not run at "
     "all. The Criteria sheet spells this out."],
    ["The focus lens has two halves",
     "Treatment and level of care codes say where a member is being treated and how intensively. "
     "Assessment, screening and education codes are the front door that decides who is admitted. "
     "B_Focus_Codes is grouped into those two blocks and B_Focus_Code_Groups compares each block "
     "against the whole book. Do not compare a number from one block against a number from the "
     "other; they answer different questions."],
    ["Focus diagnoses in this run",
     ", ".join(FOCUS_DIAGS) if HAVE_FOCUS_DIAGS else
     "NONE WERE SUPPLIED with the request. The diagnosis sheets say so rather than showing an "
     "empty table, and the code by diagnosis cross tabulation is not produced at all rather than "
     "written out as a grid of zeros. A grid of zeros reads as 'checked, nothing found' when the "
     "truth is that nobody asked the question."],
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
    ["B_Focus_Codes", "Each focus code for each provider, grouped into the treatment block and "
                      "the assessment block: lines, units, dollars, "
                      "members, service days, units per member day, dollars per unit, and what "
                      "share of this provider's whole book the code represents. A code the "
                      "provider never billed appears as a row of zeros rather than being missing, "
                      "so you can tell 'checked, none found' from 'not checked'."],
    ["B_Focus_Diagnoses", "The same for the focus primary diagnoses, when any were supplied."],
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
    ["DO NOT ADD THE RULE COUNTS TOGETHER",
     "Several tests name the same member day, so the counts on E_Rule_Summary are NOT additive. "
     "Rules 4 and 6 both test the one per diem per member per day rule, one keyed on the "
     "servicing provider and one on the billing entity; rule 7 fires on member days that often "
     "break the per diem rule as well. The top of that sheet carries a deduplicated total of "
     "distinct member days named by any rule, and that is the number to quote. Each row also says "
     "which other rules name the same findings and how many of its rows nothing else names."],
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
    ["Do not add E_Multi_Per_Diem_Days and E_One_Per_Diem_Per_Day together",
     "They test the same published rule and will often name the same member days. The older "
     "sheet groups by the servicing provider; the newer one groups by the billing entity, which "
     "is the right key for an organisation whose clinicians bill under several servicing "
     "identifiers, and it carries the dollars beyond the largest per diem. For a per diem focus "
     "set, read E_One_Per_Diem_Per_Day."],
    ["E_One_Per_Diem_Per_Day", "THE CENTRAL TEST when the focus codes are per diems. AHCCCS "
                               "allows one per diem behavioural health service per member per "
                               "day, effective 2024-02-06. This sheet lists the member days where "
                               "ONE billing entity billed two or more of them, with every per "
                               "diem code on the day and the dollars beyond the single largest "
                               "per diem. Where the second per diem came from a different entity "
                               "the row is on E_One_Per_Diem_Cross_Ent instead and is NOT a "
                               "finding against this provider."],
    ["E_Level_Of_Care_Conflicts", "Member days where one billing entity billed two levels of care "
                                  "that cannot coexist. A member cannot be in twenty four hour "
                                  "residential care and attending an intensive outpatient "
                                  "programme the same day. Check every row against "
                                  "E_Conflict_Matrix, which is the table the test was built "
                                  "from."],
    ["E_Conflict_Matrix", "The levels of care that cannot coexist, written out, with what changes "
                          "when a DIFFERENT entity bills the second one. Some combinations are "
                          "permitted cross entity and getting that wrong is the main false "
                          "positive risk in this workbook."],
    ["E_H2012_Wrong_Tier", "H2012 is billed per HOUR, up to five hours a day, even though it sits "
                           "in the middle of a per diem focus set. Above five hours the service "
                           "belongs on H2015 and then H2016. More than five units in a member day "
                           "means the service exceeded the tier it was billed under."],
    ["E_Doc_2023_07_17_Expanded", "The 2023-07-17 expanded documentation list, which is a second "
                                  "control separate from the 2023-05-03 one. H2012 is on it in "
                                  "the 'above 2 hourly units' group. Like the other one it uses "
                                  "the date the CLAIM ARRIVED, not the date of service, and if "
                                  "this table has no receipt date the test does NOT run."],
    ["E_S9480_Once_A_Day", "S9480 is payable once a day. One unit is one day of intensive "
                           "outpatient psychiatric programme."],
    ["E_S9480_Prior_Auth", "S9480 lines on or after the date prior authorization was required, "
                           "carrying no authorization number. Two dates apply: 2025-07-01 for fee "
                           "for service and 2024-10-15 for Arizona Complete Health. IHS and 638 "
                           "facilities are EXEMPT and are removed, and the sheet says on every "
                           "row whether the exemption could be applied at all."],
    ["E_Min_Intensity_S9480", "QUESTIONS, NOT FINDINGS. S9480 has a published minimum of at least "
                              "two days a week, and a claim can count days, so this one is a "
                              "rule. It also requires three hours a day, which a per diem claim "
                              "cannot show, so every row is a question for the service record. "
                              "Weeks where the shortfall is just where the calendar falls have "
                              "been EXCLUDED from this sheet, not merely labelled, and are on "
                              "E_Min_Intensity_Edges."],
    ["E_Min_Intensity_Edges", "The short S9480 weeks that were taken off the rule sheet because "
                              "they are a calendar edge: the programme carries on into the week "
                              "before or after, or the week sits at the start or end of the "
                              "extract. Nothing here is a finding. They are kept so nothing is "
                              "hidden, and the count removed is on E_Rule_Summary."],
    ["There is NO H0035 intensity rule, and that is deliberate",
     "H0035 has a published minimum of twenty HOURS a week. A per diem claim carries no hours at "
     "all, so that minimum cannot be tested and this workbook does not try. An earlier version "
     "turned it into a number of days by dividing by the longest day of programming it was "
     "willing to believe, and that divisor was ours rather than AHCCCS's. A row built on it would "
     "have sat on these rule sheets beside nine published rules where anybody skimming would take "
     "it for one. It has been removed. I_H0035_Days_Per_Week shows the plain distribution of "
     "H0035 days per member per week instead, with no threshold on it, and the service records "
     "are what answer whether the twenty hours were met."],
    ["I_H0035_Days_Per_Week", "How many member weeks had one day of H0035, how many had two, and "
                              "so on, with the members, units and dollars at each. A DISTRIBUTION, "
                              "not a test: no threshold, no flag, no finding count. The published "
                              "twenty hours a week is printed beside it as context for reading "
                              "it."],
    ["E_H0018_Place_Of_Service", "Behavioural health residential facilities must bill place of "
                                 "service 56 from 2023-12-01. These H0018 lines carry something "
                                 "else."],
    ["E_Provider_Type_Check", "WHAT THIS PROVIDER TYPE BILLED. H0018 is the residential facility "
                              "per diem and provider type B8 is the type that bills it. This "
                              "workbook does NOT assert that another type may not: the AHCCCS "
                              "code to provider type matrix settles that, and checking it is the "
                              "first thing to do with this sheet."],
    ["E_Bed_Hold_And_Leave", "Therapeutic leave (revenue code 0183) and bed hold (0189) days per "
                             "contract year, against the published limits. Needs a revenue code "
                             "column, and the age split needs a date of birth. Without either the "
                             "sheet says so."],
    ["E_98960_Daily_Cap and E_98960_Rolling_Cap",
     "Two hard limits on the education and training codes, both effective 2023-04-01: four units "
     "a day and sixty units in any rolling thirty days. They catch different things. A member can "
     "stay under four every single day and still be far above sixty in a month, which is why the "
     "rolling one is computed its own way. Both are unit counts, so both are exact."],
    ["E_H0001_Per_Year and E_H0002_Per_Year",
     "H0001 is limited to 12 a year and H0002 to 24 a year for members aged 11 and over. AHCCCS "
     "DOES NOT SAY WHICH YEAR. Both sheets therefore show the calendar year count and the benefit "
     "year count SIDE BY SIDE. A member above the limit on one basis and below it on the other is "
     "marked AMBIGUOUS and is not a finding: that member is a question about which year AHCCCS "
     "meant."],
    ["H_Daily_Census, H_Census_Summary, H_Census_Distribution",
     "THE HEADLINE WHEN THE FOCUS CODES ARE PER DIEMS. For every calendar day, how many distinct "
     "members this provider had on a per diem, overall and per code. Every calendar day gets a "
     "row including the empty ones. Read H_Census_Summary first: it has the mean, the median, the "
     "peak and the dates the peak happened on. THE NUMBER TO COMPARE THE PEAK AGAINST IS THE "
     "FACILITY'S LICENSED CAPACITY, which is the state's own record and is not in this notebook."],
    ["I_Stays, I_LOS_Distribution, I_Longest_Stays, I_H0018_Long_Stays, I_Readmissions",
     "How long each member stayed. A stay is consecutive billed days on one per diem code. "
     "I_LOS_Distribution gives the whole distribution rather than an average, because an average "
     "length of stay hides the thing worth seeing. I_H0018_Long_Stays exists because H0018 is "
     "SHORT TERM residential, so a long unbroken run of it is a question on the face of the code. "
     "The long stay cut is a percentile of THIS PROVIDER'S OWN stays: it is descriptive and is "
     "NOT a published limit. I_Readmissions lists members who left and came back repeatedly."],
    ["J_Servicing_Providers", "Who the servicing providers are under the billing entities on this "
                              "provider's claims, and how often the organisation's own identifier "
                              "is the servicing provider on its own claims. That last one is why "
                              "a day total for an organisation cannot be read as one person's "
                              "day. It is a description of how the claims were filled in, not a "
                              "finding."],
    ["K_Monthly_Dollars", "Paid dollars by month beside a $500,000 a month reference figure. THAT "
                          "FIGURE IS NOT AN AHCCCS CONTROL. It is what the state's forensic audit "
                          "recommended as a review trigger for Integrated Clinics. A month above "
                          "it is not a finding."],
    ["L_S9480_Pricing_Era", "A check on the EXTRACT rather than on the provider. AHCCCS paid "
                            "S9480 at 58.66% of billed charges under By Report pricing until a "
                            "fixed $370.41 took effect on 2022-10-01, so watching when paid over "
                            "billed stops clustering dates that change in this data."],
    ["M_Funnel, M_Funnel_Summary, M_Funnel_Lag, M_H0002_Screening",
     "A DESCRIPTION OF BILLING ORDER, and nothing more. The front door codes decide who is "
     "admitted; the per diems and day treatment are what they are admitted to. These sheets show, "
     "for each member, how long after the front door treatment started, who went through the "
     "front door and never started treatment, and who started treatment here with no front door "
     "service from anybody. Integrated care providers assess and then treat their own patients "
     "every day, entirely properly. A tight lag distribution means something is running to a "
     "schedule, and whether that schedule is a well run programme or something else is not a "
     "question claims data can answer."],
    ["Before naming anybody",
     "Check the person_or_organisation row on A_Provider_Profile. Check the Criteria sheet to "
     "confirm the unit basis of the codes involved is still right. Check E_Provider_Type_Check "
     "against the AHCCCS code to provider type matrix. For the same day tests confirm that the "
     "billing entity really is the facility. For the census, get the licensed capacity from "
     "enrolment. For H0001 and H0002, settle which year AHCCCS meant. And remember that these "
     "tests only see what these providers and their billing entities billed."],
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
    _write(w, instrument_df, "A_Which_Instrument", "")
    _write(w, profile_side_by_side, "A_Provider_Profile",
           "Neither provider has any claim line in this table. Check the IDs.")
    _write(w, provider_type_check_df, "E_Provider_Type_Check",
           "No provider type could be read from the claims, so nothing could be checked against "
           "the AHCCCS code to provider type matrix.")
    _write(w, focus_shape_df, "B_Focus_Code_Basis", _EMPTY_WARN)
    _write(w, focus_codes_df, "B_Focus_Codes", _EMPTY_WARN)
    _write(w, focus_group_summary_df, "B_Focus_Code_Groups", _EMPTY_WARN)
    _write(w, focus_diags_df, "B_Focus_Diagnoses",
           "NO FOCUS DIAGNOSES WERE SUPPLIED with this request, so there was nothing to look for. "
           "This is not a test that found nothing." if not HAVE_FOCUS_DIAGS else _EMPTY_WARN)
    _write(w, code_by_diag_df, "B_Code_By_Diagnosis",
           "NO FOCUS DIAGNOSES WERE SUPPLIED with this request, so there is nothing to cross "
           "tabulate. A grid of zeros is deliberately NOT written here: it would read as "
           "'checked, nothing found' when the truth is that nobody asked the question."
           if not HAVE_FOCUS_DIAGS else _EMPTY_WARN)
    _write(w, mod_mix_df, "B_Focus_Modifier_Mix",
           "No modifiers on any focus code line, or no focus code lines at all.")

    # The census and the episodes lead when the focus codes are per diems, and the day leads when
    # they are timed. Both are always written; only the order changes, so nothing is ever absent.
    def _write_day_section():
        _write(w, day_summary_df, "D_Day_Summary",
               HOURS_INSTRUMENT_STATUS + ". " + HOURS_INSTRUMENT_WHY if not len(day_summary_df)
               else _EMPTY_WARN)

    def _write_census_section():
        _write(w, census_summary_df, "H_Census_Summary",
               "No per diem code with a published unit basis was billed, so there is no day of "
               "care to count and no census to build. H0019 and the other codes AHCCCS has never "
               "published a unit for cannot build one.")
        _write(w, daily_census_df, "H_Daily_Census",
               "No per diem code with a published unit basis was billed.")
        _write(w, census_distribution_df, "H_Census_Distribution",
               "No census was built, so there is no distribution.")
        _write(w, los_distribution_df, "I_LOS_Distribution",
               "No stay could be built: no per diem code with a published unit basis was billed.")
        _write(w, long_stay_cut_df, "I_Long_Stay_Cut",
               "No stay could be built, so no tail could be marked.")
        _write(w, stays_df, "I_Stays", "No stay could be built.")
        _write(w, longest_stays_df, "I_Longest_Stays", "No stay could be built.")
        _write(w, h0018_long_stays_df, "I_H0018_Long_Stays",
               "This provider billed no H0018, or no H0018 stay could be built. Check "
               "B_Focus_Codes to see which.")
        _write(w, readmissions_df, "I_Readmissions",
               f"No member left and returned within {READMIT_WINDOW_DAYS} days "
               f"{READMIT_MIN_RETURNS} times or more.")
        # A distribution, not a test. There is no threshold on this sheet and nothing on it is a
        # finding, because the published H0035 minimum is in hours and a claim cannot carry them.
        _write(w, h0035_days_per_week_df, "I_H0035_Days_Per_Week",
               "H0035 was never billed, so there is no distribution of days per week to show. "
               "This sheet is a distribution and not a test: it has no threshold on it and "
               "nothing on it is a finding.")

    if SHAPE_IS_PER_DIEM:
        _write_census_section()
        _write_day_section()
    else:
        _write_day_section()
        _write_census_section()

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
    _write(w, one_per_diem_df, "E_One_Per_Diem_Per_Day",
           "No member day carried two or more per diem codes from the SAME billing entity. Cross "
           "entity collisions, which are not a finding against this provider, are on "
           "E_One_Per_Diem_Cross_Ent.")
    _write(w, one_per_diem_cross_entity_df, "E_One_Per_Diem_Cross_Ent",
           "No member day carried per diems from more than one billing entity.")
    _write(w, conflict_matrix_df, "E_Conflict_Matrix", "")
    _write(w, level_conflicts_df, "E_Level_Of_Care_Conflicts",
           "No member day carried two levels of care that cannot coexist from the same billing "
           "entity. Check E_Conflict_Matrix for what was tested.")
    _write(w, level_conflicts_cross_entity_df, "E_Level_Conflicts_Cross_Ent",
           "No member day carried conflicting levels of care from different billing entities.")
    _write(w, h2012_tier_df, "E_H2012_Wrong_Tier",
           f"No member day went above {H2012_TIER_MAX_UNITS:g} units of H2012, or H2012 was never "
           f"billed. Check B_Focus_Codes to see which.")
    _write(w, doc_expanded_df, "E_Doc_2023_07_17_Expanded",
           EXPANDED_DOC_STATUS if not RECEIPT_AVAILABLE else
           "No member day went above the 2023-07-17 expanded documentation thresholds. "
           + _EMPTY_WARN)
    _write(w, s9480_once_df, "E_S9480_Once_A_Day",
           f"S9480 was never billed above {S9480_UNITS_PER_DAY:g} unit for one member in one day, "
           f"or was never billed at all.")
    _write(w, s9480_pa_df, "E_S9480_Prior_Auth",
           PA_TEST_STATUS if not PA_COL else
           "Every S9480 line on or after the applicable date carried a prior authorization "
           "number, or was exempt as an IHS or 638 line, or there were none on or after the "
           "date.")
    _write(w, s9480_intensity_df, "E_Min_Intensity_S9480",
           f"Every member week with S9480 had it on at least {S9480_MIN_DAYS_PER_WEEK} days, or "
           f"every short week was a calendar edge and is on E_Min_Intensity_Edges, or S9480 was "
           f"never billed.")
    _write(w, s9480_intensity_edge_df, "E_Min_Intensity_Edges",
           "No short S9480 week was a calendar edge. A week is a calendar edge when the programme "
           "carries on into the week before or after, or when it sits at the first or last week "
           "of the extract.")
    _write(w, h0018_pos_df, "E_H0018_Place_Of_Service",
           POS_TEST_STATUS if not POS_CODE_COL else
           f"Every H0018 line on or after {POS_56_DATE} carried place of service "
           f"{BHRF_POS_CODE}, or H0018 was never billed on or after that date.")
    _write(w, bed_hold_df, "E_Bed_Hold_And_Leave",
           LEAVE_TEST_STATUS if not REV_CODE_COL else
           f"No line carried revenue code {REV_HOME_PASS} or {REV_BED_HOLD}, so there was no "
           f"therapeutic leave or bed hold day to count.")
    _write(w, edu_daily_cap_df, "E_98960_Daily_Cap",
           f"No member day went above {EDU_DAILY_UNIT_CAP:g} units of an education and training "
           f"code, or none was billed on or after {EDU_LIMIT_DATE}.")
    _write(w, edu_rolling_df, "E_98960_Rolling_Cap",
           f"No member went above {EDU_ROLLING_UNIT_CAP:g} units of an education and training "
           f"code in any rolling {EDU_ROLLING_WINDOW_DAYS} days.")
    _write(w, h0001_year_df, "E_H0001_Per_Year",
           f"No member went above {PER_YEAR_LIMITS['H0001']:g} H0001 on EITHER year basis, or "
           f"H0001 was never billed.")
    _write(w, h0002_year_df, "E_H0002_Per_Year",
           H0002_TEST_STATUS if not DOB_COL else
           f"No member aged {H0002_MIN_AGE} or over went above {PER_YEAR_LIMITS['H0002']:g} H0002 "
           f"on EITHER year basis, or H0002 was never billed.")
    _write(w, org_servicing_df, "J_Servicing_Providers",
           ORG_SCOPE_STATUS if ORG_SCOPE_STATUS != "run" else
           "No line was billed by any of the billing entities on these providers' claims.")
    _write(w, org_self_servicing_df, "J_Organisation_Summary",
           ORG_SCOPE_STATUS if ORG_SCOPE_STATUS != "run" else "")
    _write(w, monthly_dollars_df, "K_Monthly_Dollars", _EMPTY_WARN)
    _write(w, s9480_pricing_df, "L_S9480_Pricing_Era",
           PRICING_ERA_STATUS if not BILL_COL else
           "No S9480 line carries a billed amount above zero, so paid over billed could not be "
           "computed and the pricing era could not be dated from this data.")
    _write(w, funnel_summary_df, "M_Funnel_Summary",
           "These providers billed neither a front door code nor a treatment code.")
    _write(w, funnel_lag_stats_df, "M_Funnel_Lag",
           "No member went from a front door service to a treatment service with these "
           "providers.")
    _write(w, funnel_lag_distribution_df, "M_Funnel_Lag_Distribution",
           "No member went from a front door service to a treatment service with these "
           "providers.")
    _write(w, funnel_df, "M_Funnel",
           "These providers billed neither a front door code nor a treatment code.")
    _write(w, h0002_screening_df, "M_H0002_Screening",
           "These providers never billed H0002, so there is no screening to report.")
    _write(w, member_overlap_df, "F_Member_Overlap",
           "Only one provider was analysed, so there is no overlap to report.")
    _write(w, overlap_members_df, "F_Overlap_Members",
           "Only one provider was analysed, so there is no overlap to report."
           if len(TARGETS) < 2 else "The providers share no members at all.")
    _write(w, overlap_days_df, "F_Overlap_Member_Days",
           "Only one provider was analysed, so there are no shared member days."
           if len(TARGETS) < 2 else
           "The providers share no members, so there are no shared member days.")
    _write(w, shared_entities_df, "F_Shared_Billing_Entities",
           "Only one provider was analysed, so there is no billing entity in common to report."
           if len(TARGETS) < 2 else "The providers have no billing entity in common.")
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
        # An organisation's day is not a clinician's day, so the sheet is not called one. The
        # name is the first thing a reader sees and a wrong name invites exactly the reading
        # this notebook spends a page warning against.
        _write(w, CLIN_DAY_SHEETS.get(pid),
               pfx(pid, "Clinician_Days" if IS_PERSON.get(pid) else "Provider_Days"),
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
# MAGIC ## 28. Email the workbook
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


_census_line = (
    f"<li><b>Daily census.</b> Peak {int(CENSUS_PEAK)} members on a per diem in one day, on "
    f"{', '.join(str(d) for d in CENSUS_PEAK_DATES[:3])}"
    f"{' and other dates' if len(CENSUS_PEAK_DATES) > 3 else ''}. Compare that against the "
    f"facility's LICENSED CAPACITY, which is the state's own record and is not in this "
    f"workbook.</li>"
    if CENSUS_AVAILABLE else
    "<li>No daily census could be built: no per diem code with a published unit basis was "
    "billed.</li>")
_stay_line = (
    f"<li><b>Stays.</b> {N_STAYS:,} built. Members who left and returned within "
    f"{READMIT_WINDOW_DAYS} days {READMIT_MIN_RETURNS} times or more: "
    f"{len(readmissions_df)}.</li>" if N_STAYS else
    "<li>No stay could be built.</li>")

body = f"""
<p>Deep dive on {len(TARGETS)} servicing provider{'s' if len(TARGETS) != 1 else ''}, run
{RUN_TAG}.</p>
{_profile_rows()}
<p><b>Which instrument fits this request.</b> The focus codes are a <b>{FOCUS_SHAPE}</b> set.
{FOCUS_SHAPE_BASIS}. {HOURS_INSTRUMENT_STATUS}: {HOURS_INSTRUMENT_WHY}</p>
<p><b>{ORG_DAY_WARNING}</b></p>
<p><b>Provider type against the codes billed.</b> {PROVIDER_TYPE_STATEMENT}</p>
<p>The {len(FOCUS_CODES)} focus codes ({', '.join(FOCUS_CODES)})
{('and the ' + str(len(FOCUS_DIAGS)) + ' focus diagnoses (' + ', '.join(FOCUS_DIAGS) + ') are')
 if HAVE_FOCUS_DIAGS else 'are'} used as a lens.
{'' if HAVE_FOCUS_DIAGS else
 'No focus diagnoses were supplied with this request, so every diagnosis sheet says so rather than '
 'showing an empty table, and the diagnosis measure was dropped from the peer comparison.'}
Every claim for these providers is in the workbook, not only those codes, so that every share has a
denominator and the same day conflict tests can run at all.</p>
<p>Published rule tests, which need no peer comparison.
<b>The counts below are not additive</b>: several tests name the same member day, and rules 4 and 6
both test the one per diem per member per day rule from different keys. Across every rule in the
workbook, <b>{DISTINCT_MEMBER_DAYS_ACROSS_RULES} distinct member days</b> are named by at least one
of them, and that is the figure to quote. The E_Rule_Summary sheet says which rules overlap with
which.</p>
{_rule_rows()}
<p style="color:#666">There is deliberately no H0035 minimum intensity test. Its published minimum
is twenty HOURS a week and a per diem claim carries none, so turning it into a number of days would
need a divisor nobody has published. The plain distribution of H0035 days per member per week is on
I_H0035_Days_Per_Week instead, with no threshold on it. The S9480 weekly minimum IS published in
days, so it stays a rule, with {S9480_EDGE_WEEKS_REMOVED} short weeks excluded because the
shortfall was where the calendar falls rather than a short programme; those are on
E_Min_Intensity_Edges.</p>
<ul>
  {_census_line}
  {_stay_line}
  <li>Months paid above the ${MONTHLY_PAID_REFERENCE:,.0f} figure: <b>{MONTHS_OVER_REFERENCE}</b>
      of {len(monthly_dollars_df)}. That figure is the state forensic audit's RECOMMENDED review
      trigger for Integrated Clinics, not an AHCCCS control, and a month above it is not a
      finding.</li>
  <li>Members whose H0002 screening and treatment were billed by the same entity:
      <b>{H0002_SAME_ENTITY_N}</b> of {len(h0002_screening_df)}. That is common and legitimate in
      integrated care and is reported so it can be seen.</li>
  <li>Members seen by every provider named: <b>{len(SHARED_MEMBERS)}</b>. Days when more than one
      billed the same member: <b>{_same_day}</b>. Billing entities in common:
      <b>{len(SHARED_ENTITIES)}</b>. Referring providers in common:
      <b>{len(SHARED_REFERRERS)}</b>.</li>
  <li>{'Peer measures marked contaminated: <b>' + ', '.join(sorted(set(CONTAMINATED_MEASURES))) + '</b>. Do not let those percentiles decide anything on their own.' if CONTAMINATED_MEASURES else 'No peer measure crossed the contamination threshold this run.'}</li>
</ul>
<p>Open the <b>Start_Here</b> sheet first; it explains every other sheet in plain language.</p>
<p style="color:#666">Cautions worth repeating. H0031, H0001 and H0002 are billed per session and
AHCCCS states no duration for H0031, so it adds no hours to any day total and that is correct.
H0019 keeps every dollar it was paid and is deliberately kept out of every day count, census figure
and length of stay, because AHCCCS has never published what one of its units covers. H2012 is billed
per HOUR, not per day, even though it sits in the middle of a per diem focus set. The H0001 and
H0002 yearly limits are reported on two year bases because AHCCCS does not say which year it means,
and a member above on one basis and below on the other is a question rather than a finding. The long
stay cut is this provider's own distribution and is not a published limit. And AHCCCS permits
therapy billed by a separate provider on the same day as a residential per diem, so the same day
tests only report the same billing entity.</p>
<p style="color:#666">These are leads for review based only on claims data. They do not establish
fraud, intent, or whether a service was medically necessary.</p>
<p style="color:#666">File: {OUTPUT_PATH}</p>
"""


def send_workbook():
    if not EMAIL_TO:
        return "skipped: no recipients configured"
    user, pwd = _secret(SMTP_USER_KEY), _secret(SMTP_PASS_KEY)
    msg = EmailMessage()
    # The DEDUPLICATED figure, not the sum of the rule counts. Several tests name the same member
    # day, so summing them would put a double counted number in the subject line of the email the
    # whole review starts from.
    msg["Subject"] = (f"Provider deep dive - {', '.join(TARGETS)} - {RUN_TAG} - "
                      f"{DISTINCT_MEMBER_DAYS_ACROSS_RULES} distinct member days named by a rule")
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
# MAGIC ## 29. How to point this at other providers, and how to schedule it
# MAGIC
# MAGIC 1. Click **Schedule** at the top right of this notebook, or just run it with different
# MAGIC    parameters.
# MAGIC 2. Under Parameters set `targets` to the servicing provider IDs you want, comma separated. Set
# MAGIC    `focus_codes` to the codes you want highlighted, and `focus_diags` to the diagnoses, or
# MAGIC    leave `focus_diags` empty if the request gives none. Set `email_to`.
# MAGIC 3. If you add a provider ID, add its name to `TARGET_NAMES_GIVEN` in section 1 as well, so the
# MAGIC    workbook can tell you whether the name on the claims matches the name you were given. If no
# MAGIC    name came with the request, leave it out and the workbook says there was nothing to check.
# MAGIC    That is the only code edit this notebook needs.
# MAGIC 4. Leave `focus_shape` on `auto`. The notebook works out from the paid dollars whether it is
# MAGIC    looking at a timed set or a per diem set and leads with the right section. Override it only
# MAGIC    when you want to force the other view of a set you already understand.
# MAGIC
# MAGIC **To reproduce the earlier timed code run exactly**, set `targets` to `123531,120020`,
# MAGIC `focus_codes` to `H2038,T1016,H0004,H0031` and `focus_diags` to `F1020,F1520,F4310,F1010`.
# MAGIC The notebook detects a timed set, leads with the day section as it did before, and produces
# MAGIC the same answers for those tests. The per diem sections are still written and will be thin,
# MAGIC which is correct for those providers rather than a gap.
# MAGIC
# MAGIC **Six things to do before relying on the output.**
# MAGIC
# MAGIC 1. Read **A_Which_Instrument** first. It says whether the focus codes are a timed set or a per
# MAGIC    diem set, and therefore whether the hours or the census is the number that means something.
# MAGIC    A near zero hour total against a per diem provider is the arithmetic working, not a quiet
# MAGIC    provider.
# MAGIC 2. Read the **Data_Availability** and **Analysis_Status** sheets. Anything listed as a
# MAGIC    missing column means a part of this workbook could not be computed, and those sheets say
# MAGIC    exactly which part, whether it was skipped outright or only weakened, and why.
# MAGIC 3. Read the **person_or_organisation** row on A_Provider_Profile. If a provider turns out to be
# MAGIC    an organisational NPI rather than a person, every hour figure on the day sheets is that
# MAGIC    organisation's whole staff and must not be read as one person's day.
# MAGIC 4. Take **E_Provider_Type_Check** to the AHCCCS code to provider type matrix. That matrix, not
# MAGIC    this notebook, settles whether a provider type may bill a code.
# MAGIC 5. Get the **licensed capacity** out of provider enrolment before you use the census peak for
# MAGIC    anything. This notebook does not hold capacity and does not guess at it.
# MAGIC 6. Read **E_Rule_Summary** twice. Once for a test producing more rows than anybody can work,
# MAGIC    and once for a test producing none at all. A zero is not a pass; in this project a zero has
# MAGIC    twice meant a broken test rather than a clean provider.
# MAGIC
# MAGIC **Two things worth settling by hand, once.** Take three rows from E_One_Per_Diem_Per_Day and
# MAGIC three from E_Level_Of_Care_Conflicts and pull the actual records. That is the only way to find
# MAGIC out whether the unit basis and the billing entity logic behave the way this notebook assumes on
# MAGIC real Arizona claims, and it costs an afternoon. Then settle which year AHCCCS means in the
# MAGIC H0001 and H0002 yearly limits, because every row on those two sheets depends on it and this
# MAGIC notebook reports both answers precisely because nobody has said.
