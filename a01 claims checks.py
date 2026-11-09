# Databricks notebook source
# MAGIC %md
# MAGIC # a01 · Claims checks the open readings wait on (CPU cluster with Unity Catalog)
# MAGIC
# MAGIC Read only: it writes nothing. Run it after 03 (it needs kb_table_rows) and read the last cell.
# MAGIC
# MAGIC 1. **Formats.** Do the claim values look like B2's values (provider type "02" vs "2", POS "11" vs "11.0")? If the
# MAGIC    match rate in cell 2 is low, fix the normalisation before trusting anything below.
# MAGIC 2. **Fill rates.** The telehealth and PA records are *conditional* until POS_CODE and EVENT_PA_NO are measured.
# MAGIC    EVENT_PA_NO should be filled far more often on codes the PA file marks as requiring PA (1, 3) than on codes
# MAGIC    it marks as not requiring it (4); if it is not, the field does not mean what the PA records assume.
# MAGIC 3. **B2-MOD, B2-POS, B2-PT.** How often do paid lines on B2 codes carry a modifier, a place of service or a
# MAGIC    provider type B2 does not list for the code? This is what AHCCCS's own claim edits let through, the evidence
# MAGIC    for whether B2's lists can be read as complete (a line outside them is a violation) or only as a reference.
# MAGIC
# MAGIC Only the newest B2 version in kb_table_rows is used, so the check is limited to recent claims (SINCE).
# MAGIC The claims table writes -1 in empty modifier slots: a modifier counts only when it is two letters or digits,
# MAGIC and -1 or zeros count as empty in EVENT_PA_NO and the diagnosis columns.

# COMMAND ----------

# MAGIC %run ./_settings

# COMMAND ----------

CLAIMS = "main.prod_input.all_data_C_A"
SINCE = "2026-04-01"            # service dates from here on; the B2 version loaded is dated 2026-10-01
FORMS = ("A",)                  # FORM_TYP: A = professional (CMS-1500)
WHERE = "1 = 1"                 # extra filter on the claims table, e.g. "PMT_AMT > 0"
TOP = 25


def t(name: str) -> str:
    return config.table(name)


def pt(col: str) -> str:        # provider type: B2 writes two characters ("02", "C2")
    return (f"CASE WHEN trim(CAST({col} AS STRING)) RLIKE '^[0-9]$' THEN lpad(trim(CAST({col} AS STRING)), 2, '0') "
            f"ELSE upper(trim(CAST({col} AS STRING))) END")


def pos(col: str) -> str:       # place of service: two digits, no decimal tail
    return f"lpad(regexp_replace(trim(CAST({col} AS STRING)), '\\\\.0+$', ''), 2, '0')"


def mod(col: str) -> str:       # a modifier is two letters or digits; the claims write -1 for an empty slot
    v = f"upper(trim(CAST({col} AS STRING)))"
    return f"CASE WHEN {v} RLIKE '^[A-Z0-9]{{2}}$' THEN {v} END"


def blank(col: str) -> str:     # the value, or NULL when empty, -1 or all zeros
    v = f"trim(CAST({col} AS STRING))"
    return f"CASE WHEN {v} IS NULL OR {v} IN ('', '-1') OR {v} RLIKE '^0+$' THEN NULL ELSE {v} END"


def newest(spec: str) -> str:
    row = spark.sql(f"""SELECT r.source_version_id FROM (SELECT DISTINCT source_version_id FROM {t('table_rows')}
                        WHERE spec = '{spec}') r JOIN {t('source_versions')} v USING (source_version_id)
                        ORDER BY v.version_date DESC LIMIT 1""").collect()  # noqa: F821
    if not row:
        raise ValueError(f"no {spec} rows in {t('table_rows')}: run 03 first")
    return row[0][0]


B2_VERSION, PA_VERSION = newest("b2"), newest("pa_guidelines")
forms = ", ".join(f"'{f}'" for f in FORMS)
spark.sql(f"""
    CREATE OR REPLACE TEMP VIEW b2_rows AS
    SELECT d.code, d.provider_type, d.modifiers, d.places_of_service FROM (
      SELECT from_json(data, 'struct<code:string, provider_type:string, modifiers:array<string>,
                                      places_of_service:array<string>>') AS d
      FROM {t('table_rows')} WHERE spec = 'b2' AND source_version_id = '{B2_VERSION}')""")  # noqa: F821
spark.sql("""
    CREATE OR REPLACE TEMP VIEW b2_code AS
    SELECT code, array_distinct(flatten(collect_list(modifiers))) AS mods,
           array_distinct(flatten(collect_list(places_of_service))) AS poss, collect_set(provider_type) AS types,
           collect_list(struct(provider_type, modifiers, places_of_service)) AS by_type
    FROM b2_rows GROUP BY code""")  # noqa: F821
spark.sql(f"""
    CREATE OR REPLACE TEMP VIEW lines AS
    SELECT upper(trim(CAST(PROC_CD AS STRING))) AS code, {pt('SProv_Type')} AS rtype, {pt('BProv_Type')} AS btype,
           {pos('POS_CODE')} AS pos,
           filter(array({mod('PROC_MOD_1')}, {mod('PROC_MOD_2')}, {mod('PROC_MOD_3')}, {mod('PROC_MOD_4')}),
                  x -> x IS NOT NULL) AS mods,
           {blank('EVENT_PA_NO')} AS pa_no, CAST(PMT_AMT AS DOUBLE) AS paid,
           CAST(BProv_ID AS STRING) AS bprov
    FROM {CLAIMS}
    WHERE CAST(FORM_TYP AS STRING) IN ({forms}) AND to_date(Svc_Begin_Dt) >= DATE'{SINCE}' AND ({WHERE})""")  # noqa: F821
spark.sql("""
    CREATE OR REPLACE TEMP VIEW b2_lines AS
    SELECT l.*, b.mods AS b2_mods, b.poss AS b2_poss, b.types AS b2_types,
           filter(b.by_type, r -> r.provider_type IN (l.rtype, l.btype)) AS matched
    FROM lines l JOIN b2_code b ON b.code = l.code""")  # noqa: F821
n_lines = spark.sql("SELECT count(*) FROM lines").collect()[0][0]  # noqa: F821
print(f"B2 {B2_VERSION} | PA {PA_VERSION} | {n_lines:,} claim lines since {SINCE} on forms {FORMS}")
if n_lines == 0:
    print("No lines: check SINCE, FORMS, WHERE and whether Svc_Begin_Dt reads as a date (to_date)")

# COMMAND ----------

# MAGIC %md ## 1 · Formats: do claim values look like B2's?

# COMMAND ----------

display(spark.sql("""
    SELECT count(*) AS lines_on_b2_codes,
           round(avg(CASE WHEN array_contains(b2_types, rtype) OR array_contains(b2_types, btype) THEN 1 ELSE 0 END), 4)
             AS share_type_listed,
           round(avg(CASE WHEN pos IS NULL THEN NULL WHEN array_contains(b2_poss, pos) THEN 1 ELSE 0 END), 4)
             AS share_pos_listed,
           round(avg(CASE WHEN size(mods) = 0 THEN NULL WHEN size(array_except(mods, b2_mods)) = 0 THEN 1 ELSE 0 END), 4)
             AS share_mods_listed
    FROM b2_lines"""))  # noqa: F821
for col in ("rtype", "btype", "pos"):
    display(spark.sql(f"SELECT {col}, count(*) AS lines FROM b2_lines GROUP BY {col} ORDER BY lines DESC LIMIT 15"))  # noqa: F821
display(spark.sql("SELECT m, count(*) AS lines FROM b2_lines LATERAL VIEW explode(mods) e AS m GROUP BY m "
                  "ORDER BY lines DESC LIMIT 15"))  # noqa: F821

# COMMAND ----------

# MAGIC %md ## 2 · Fill rates (all lines in scope) and whether EVENT_PA_NO follows the PA file

# COMMAND ----------

diag = [c for c in spark.table(CLAIMS).columns if any(k in c.upper() for k in ("DIAG", "ICD", "DX_"))]  # noqa: F821
print("diagnosis-like columns:", diag or "none found")
extra = "".join(f", round(avg(CASE WHEN {blank(c)} IS NULL THEN 0 ELSE 1 END), 4) AS fill_{c}"
                for c in diag if not c.upper().endswith(("_DESC", "_KEY")))
display(spark.sql(f"""
    SELECT count(*) AS lines,
           round(avg(CASE WHEN {pos('POS_CODE')} IS NULL OR {pos('POS_CODE')} IN ('00', '') THEN 0 ELSE 1 END), 4)
             AS fill_pos,
           round(avg(CASE WHEN {mod('PROC_MOD_1')} IS NULL THEN 0 ELSE 1 END), 4) AS fill_modifier_1,
           round(avg(CASE WHEN {blank('EVENT_PA_NO')} IS NULL THEN 0 ELSE 1 END), 4) AS fill_pa_no{extra}
    FROM {CLAIMS}
    WHERE CAST(FORM_TYP AS STRING) IN ({forms}) AND to_date(Svc_Begin_Dt) >= DATE'{SINCE}' AND ({WHERE})"""))  # noqa: F821
spark.sql(f"""
    CREATE OR REPLACE TEMP VIEW pa_code AS
    SELECT d.code, max(d.pa_code) AS pa_code FROM (
      SELECT from_json(data, 'struct<code:string, pa_code:string>') AS d FROM {t('table_rows')}
      WHERE spec = 'pa_guidelines' AND source_version_id = '{PA_VERSION}') GROUP BY d.code""")  # noqa: F821
display(spark.sql("""
    SELECT coalesce(p.pa_code, 'not in PA file') AS pa_file_says, count(*) AS lines,
           round(avg(CASE WHEN l.pa_no IS NULL OR l.pa_no RLIKE '^0+$' THEN 0 ELSE 1 END), 4) AS share_with_pa_no
    FROM lines l LEFT JOIN pa_code p ON p.code = l.code GROUP BY 1 ORDER BY 1"""))  # noqa: F821
first_dx = next((c for c in diag if c.upper() in ("DIAG_CD_1", "DIAG_CD1", "PRIN_DIAG_CD")), None)
if first_dx:                    # real ICD-10-CM codes, or a placeholder?
    display(spark.sql(f"""
        SELECT round(avg(CASE WHEN v RLIKE '^[A-Z][0-9][0-9A-Z]' THEN 1 ELSE 0 END), 4) AS share_icd10_shaped,
               round(avg(CASE WHEN v RLIKE '^F' THEN 1 ELSE 0 END), 4) AS share_f_codes,
               count(DISTINCT v) AS distinct_codes
        FROM (SELECT upper(replace({blank(first_dx)}, '.', '')) AS v FROM {CLAIMS}
              WHERE CAST(FORM_TYP AS STRING) IN ({forms}) AND to_date(Svc_Begin_Dt) >= DATE'{SINCE}' AND ({WHERE}))
        WHERE v IS NOT NULL"""))  # noqa: F821
    display(spark.sql(f"""SELECT {first_dx} AS value, count(*) AS lines FROM {CLAIMS}
        WHERE CAST(FORM_TYP AS STRING) IN ({forms}) AND to_date(Svc_Begin_Dt) >= DATE'{SINCE}' AND ({WHERE})
        GROUP BY 1 ORDER BY lines DESC LIMIT 15"""))  # noqa: F821

# COMMAND ----------

# MAGIC %md ## 3 · B2-MOD: modifiers B2 does not list for the code

# COMMAND ----------

spark.sql("""
    CREATE OR REPLACE TEMP VIEW mod_check AS
    SELECT *, array_except(mods, b2_mods) AS off_any_type,
           array_except(mods, array_distinct(flatten(transform(matched, r -> r.modifiers)))) AS off_own_type
    FROM b2_lines WHERE size(mods) > 0""")  # noqa: F821
display(spark.sql("""
    SELECT count(*) AS lines_with_modifier,
           sum(CASE WHEN size(off_any_type) > 0 THEN 1 ELSE 0 END) AS lines_off_list_any_type,
           round(avg(CASE WHEN size(off_any_type) > 0 THEN 1 ELSE 0 END), 4) AS share_off_any_type,
           round(avg(CASE WHEN size(matched) > 0 AND size(off_own_type) > 0 THEN 1 ELSE 0 END), 4) AS share_off_own_type,
           round(sum(CASE WHEN size(off_any_type) > 0 THEN paid ELSE 0 END), 2) AS paid_off_any_type,
           count(DISTINCT CASE WHEN size(off_any_type) > 0 THEN bprov END) AS billing_providers_off
    FROM mod_check"""))  # noqa: F821
display(spark.sql(f"""
    SELECT code, m AS modifier_not_in_b2, count(*) AS lines, count(DISTINCT bprov) AS billing_providers,
           round(sum(paid), 2) AS paid
    FROM mod_check LATERAL VIEW explode(off_any_type) e AS m GROUP BY code, m ORDER BY lines DESC LIMIT {TOP}"""))  # noqa: F821

# COMMAND ----------

# MAGIC %md ## 4 · B2-POS: places of service B2 does not list for the code

# COMMAND ----------

display(spark.sql("""
    SELECT count(*) AS lines_with_pos,
           sum(CASE WHEN NOT array_contains(b2_poss, pos) THEN 1 ELSE 0 END) AS lines_off_list,
           round(avg(CASE WHEN NOT array_contains(b2_poss, pos) THEN 1 ELSE 0 END), 4) AS share_off_any_type,
           round(avg(CASE WHEN size(matched) > 0 AND NOT exists(matched, r -> array_contains(r.places_of_service, pos))
                          THEN 1 ELSE 0 END), 4) AS share_off_own_type,
           round(sum(CASE WHEN NOT array_contains(b2_poss, pos) THEN paid ELSE 0 END), 2) AS paid_off_list,
           count(DISTINCT CASE WHEN NOT array_contains(b2_poss, pos) THEN bprov END) AS billing_providers_off
    FROM b2_lines WHERE pos IS NOT NULL AND pos <> '00'"""))  # noqa: F821
display(spark.sql(f"""
    SELECT code, pos AS pos_not_in_b2, count(*) AS lines, count(DISTINCT bprov) AS billing_providers,
           round(sum(paid), 2) AS paid
    FROM b2_lines WHERE pos IS NOT NULL AND pos <> '00' AND NOT array_contains(b2_poss, pos)
    GROUP BY code, pos ORDER BY lines DESC LIMIT {TOP}"""))  # noqa: F821

# COMMAND ----------

# MAGIC %md ## 5 · B2-PT: H0xxx and H2xxx lines whose provider types B2 does not list for the code

# COMMAND ----------

display(spark.sql("""
    SELECT count(*) AS h_code_lines,
           round(avg(CASE WHEN NOT array_contains(b2_types, rtype) THEN 1 ELSE 0 END), 4) AS share_rendering_unlisted,
           round(avg(CASE WHEN NOT array_contains(b2_types, btype) THEN 1 ELSE 0 END), 4) AS share_billing_unlisted,
           round(avg(CASE WHEN size(matched) = 0 THEN 1 ELSE 0 END), 4) AS share_neither_listed,
           round(sum(CASE WHEN size(matched) = 0 THEN paid ELSE 0 END), 2) AS paid_neither_listed
    FROM b2_lines WHERE code RLIKE '^H[02][0-9]{3}$'"""))  # noqa: F821
display(spark.sql(f"""
    SELECT code, rtype, btype, count(*) AS lines, count(DISTINCT bprov) AS billing_providers, round(sum(paid), 2) AS paid
    FROM b2_lines WHERE code RLIKE '^H[02][0-9]{{3}}$' AND size(matched) = 0
    GROUP BY code, rtype, btype ORDER BY lines DESC LIMIT {TOP}"""))  # noqa: F821

# COMMAND ----------

# MAGIC %md
# MAGIC ## Reading the results
# MAGIC
# MAGIC First check section 1: if the share of lines whose provider type, POS or modifiers B2 lists is low across the
# MAGIC board, the formats differ and the numbers below overstate what is off the list.
# MAGIC
# MAGIC For each of B2-MOD, B2-POS and B2-PT, the share of lines off the list (any provider type) says what AHCCCS's own
# MAGIC claim edits allow. The cut-offs are a judgment call, not a standard:
# MAGIC
# MAGIC | share off the list | what it suggests | reading |
# MAGIC |---|---|---|
# MAGIC | under about 0.5% | AHCCCS edits enforce the list; what gets through is rare | sign as complete: few findings, each worth a look |
# MAGIC | about 0.5% to 2% | partly enforced, or enforced for some codes only | look at the top pairs before deciding |
# MAGIC | over about 2%, across many providers | AHCCCS pays outside the list | reject as complete; keep the list as a review signal |
# MAGIC
# MAGIC A line off the list that AHCCCS paid passed every claim edit. Calling it a violation needs a source that says the
# MAGIC list is complete, which is what the reading claims and what FFS chapter 19 (B2 "is intended to be a reference
# MAGIC document") argues against. Send the five summary rows back and the readings can be settled.
