-- EVENT_PA_NO checks for d02 (run in a01 after section 2: it reuses the pa_code view that section builds)
-- Same window and form as a01: professional lines (FORM_TYP A), service dates from 2024-10-01, paid rows only.

-- 0. The lines: one row per claim row; -1, blanks and zeros in EVENT_PA_NO count as no number
CREATE OR REPLACE TEMP VIEW pa_lines AS
SELECT trim(CAST(ClaimID AS STRING)) AS claim_id,
       upper(trim(CAST(PROC_CD AS STRING))) AS code,
       coalesce(CAST(CLM_TYP AS STRING), '(blank)') AS clm_typ,
       CASE WHEN trim(CAST(BProv_Type AS STRING)) RLIKE '^[0-9]$' THEN lpad(trim(CAST(BProv_Type AS STRING)), 2, '0')
            ELSE upper(trim(CAST(BProv_Type AS STRING))) END AS btype,
       trim(CAST(BProv_ID AS STRING)) AS bprov,
       trim(CAST(MEMBER_KEY AS STRING)) AS member,
       CASE WHEN trim(CAST(EVENT_PA_NO AS STRING)) IN ('', '-1') OR trim(CAST(EVENT_PA_NO AS STRING)) RLIKE '^0+$'
            THEN NULL ELSE trim(CAST(EVENT_PA_NO AS STRING)) END AS pa_no,
       CAST(QUANTITY_PAID AS DOUBLE) AS units,
       CAST(PMT_AMT AS DOUBLE) AS paid
FROM main.prod_input.all_data_C_A
WHERE CAST(FORM_TYP AS STRING) = 'A' AND to_date(Svc_Begin_Dt) >= DATE'2024-10-01' AND CAST(PMT_AMT AS DOUBLE) > 0;

-- A1. Is the number there when the PA file says PA is required? By claim type and PA code.
--     pa_code 3 (required for acute and LTC) is the clean test; 1 and 2 depend on the member's program.
--     claim_fill counts a number on any line of the claim (a number kept on one line only shows here).
SELECT l.clm_typ, coalesce(p.pa_code, 'not in PA file') AS pa_file_says, count(*) AS lines,
       round(avg(CASE WHEN l.pa_no IS NULL THEN 0 ELSE 1 END), 3) AS line_fill,
       round(avg(CASE WHEN c.claim_pa IS NULL THEN 0 ELSE 1 END), 3) AS claim_fill,
       round(sum(CASE WHEN l.pa_no IS NULL THEN l.paid ELSE 0 END), 0) AS paid_with_no_number
FROM pa_lines l
LEFT JOIN pa_code p ON p.code = l.code
JOIN (SELECT claim_id, max(pa_no) AS claim_pa FROM pa_lines GROUP BY claim_id) c ON c.claim_id = l.claim_id
GROUP BY 1, 2 ORDER BY 1, 2;

-- A2. FFS lines on pa_code 3 codes: where the blanks sit, by billing provider type
SELECT l.btype, count(*) AS lines,
       round(avg(CASE WHEN l.pa_no IS NULL THEN 0 ELSE 1 END), 3) AS line_fill,
       round(sum(CASE WHEN l.pa_no IS NULL THEN l.paid ELSE 0 END), 0) AS paid_with_no_number
FROM pa_lines l JOIN pa_code p ON p.code = l.code
WHERE l.clm_typ = 'F' AND p.pa_code = '3'
GROUP BY 1 ORDER BY lines DESC LIMIT 20;

-- B1. Does one PA number show on more than one member or billing provider? (a PA is found by member and NPI)
SELECT count(*) AS pa_numbers,
       sum(CASE WHEN members > 1 THEN 1 ELSE 0 END) AS on_2plus_members,
       sum(CASE WHEN bprovs > 1 THEN 1 ELSE 0 END) AS on_2plus_billing_providers
FROM (SELECT pa_no, count(DISTINCT member) AS members, count(DISTINCT bprov) AS bprovs
      FROM pa_lines WHERE pa_no IS NOT NULL GROUP BY pa_no);

-- B2. The numbers behind B1, largest first
SELECT pa_no, clm_typs, members, billing_providers, codes, lines, units, paid
FROM (SELECT pa_no, array_join(array_sort(collect_set(clm_typ)), ' ') AS clm_typs,
             count(DISTINCT member) AS members, count(DISTINCT bprov) AS billing_providers,
             count(DISTINCT code) AS codes, count(*) AS lines, sum(units) AS units, round(sum(paid), 0) AS paid
      FROM pa_lines WHERE pa_no IS NOT NULL GROUP BY pa_no)
WHERE members > 1 OR billing_providers > 1
ORDER BY members DESC, billing_providers DESC, paid DESC LIMIT 50;

-- C. Is there an authorization table (number, member, provider, code, approved units, dates, status)?
--    system.information_schema spans every catalog you can read; use main.information_schema if this one is closed.
SELECT table_catalog, table_schema, table_name, column_name, data_type
FROM system.information_schema.columns
WHERE lower(column_name) RLIKE 'pa_no|pa_num|pa_id|prior_auth|auth_no|auth_num|auth_id|authoriz'
ORDER BY 1, 2, 3, 4;
