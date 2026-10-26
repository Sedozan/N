# =====================================================================================
# llm_summarization_gpt_v3.4  :  wiring patch for AHCCCS feature dictionary v2
#
# Minimal-diff. Existing variable names and function signatures are kept. The new
# dictionary fields are composed into the existing Clinical_Context column, which is
# already passed through build_structured_context -> build_prompt.
#
# Step 0: export the Dictionary sheet to the CSV the notebook already reads:
#   d = pd.read_excel('AHCCCS_feature_dictionary_v2.xlsx', sheet_name='Dictionary').fillna('')
#   d = d.rename(columns={'Display_Label': 'Replacement'})
#   d['Purpose'] = d['Description']
#   d['Clinical_Context'] = d['Description']
#   d['Thresholds'] = ''                      # tier ladders now live in Description
#   d.to_csv(f'/Workspace/Users/{ROOT}/Projects/all_projects_predictions/data/feature_mapping.csv',
#            index=False)
# =====================================================================================


# -------------------------------------------------------------------------------------
# PATCH 1  (CELL 31, append at the end of the cell)
# -------------------------------------------------------------------------------------
NEW_COLS = ['Suspicious_Pattern', 'Benign_Explanations', 'Narrative_Caveat', 'Narrate_Role',
            'Validation_Status', 'Applies_To', 'Concept_Group', 'Value_Template', 'Value_Type']

def _prefix_dict(col):
    if col not in mapping_df.columns:
        print(f"WARNING: {col} not in feature_mapping.csv")
        return {}
    vals = mapping_df[col].fillna('').astype(str)
    return dict(sorted(zip(mapping_df['Feature_Root'], vals), key=sort_key, reverse=True))

new_field_dicts = {c: _prefix_dict(c) for c in NEW_COLS}

def get_mapping_field(feature, col):
    """Longest-prefix lookup, same matching rule as get_clinical_context."""
    if pd.isna(feature):
        return ""
    d = new_field_dicts.get(col, {})
    s = str(feature)
    if s in d:
        return d[s]
    for root, v in d.items():
        if s.startswith(root):
            return v
    return ""

def get_mapping_root(feature):
    s = str(feature)
    for root in mapping_dict:                 # already sorted longest-first
        if s == root or s.startswith(root):
            return root
    return ""

# Tier and band labels arrive inside the feature name ('..._class_Rare Anomaly').
# Translate them so words like 'Anomaly' or 'Suspicious' never reach the narrative.
TIER_ROOTS = {'average_procedure_zscore_class', 'global_procedure_zscore_class', 'mileage_zscore_class',
              'not_mileage_zscore_class', 'procedure_volume_zscore_class', 'reservation_zscore_class',
              'wait_time_zscore_class'}
TIER_PLAIN = {
    'extreme low underuse': 'far below peers', 'very low underuse': 'well below peers',
    'slightly below avg underuse': 'below peers', 'normal': 'in line with peers',
    'mild anomaly': 'somewhat above peers', 'anomaly': 'well above peers',
    'rare anomaly': 'far above peers', 'extreme anomaly': 'among the most extreme above peers',
}
WAIT_BANDS = {  # unusual_wait_time_class: fixed hours, NOT peer comparisons (1 T2007 unit = 30 min)
    'normal': 'up to 2 hours', 'high': '2 to 4 hours', 'extremely high': '4 to 8 hours',
    'abnormal': '8 to 24 hours', 'suspicious': 'more than 24 hours', 'no wait time': 'no wait time',
}

def plain_level(feature):
    """Return the plain-English level for a categorical/tier feature, or '' if none."""
    s, root = str(feature), get_mapping_root(feature)
    if not root or len(s) <= len(root):
        return ''
    raw = s[len(root):].lstrip('_').replace('_', ' ').strip()
    key = raw.lower()
    if root in TIER_ROOTS:
        return TIER_PLAIN.get(key, raw)
    if root == 'unusual_wait_time_class':
        return WAIT_BANDS.get(key, raw)
    if root in ('visit_frequency_class', 'weeks_with_unusual_visits_class'):
        m = re.search(r'above\s*(\d+)', key)
        if m:
            return f"top {100 - int(m.group(1))}% of providers"
        if 'below' in key:
            return 'below the top 5% of providers'
    return raw

CATEGORY_DISPLAY = {
    'NEMT NOT IHS': 'NEMT (non-IHS)', 'NEMT IHS': 'NEMT (IHS)', 'DME MEDICAL SUPPLIES': 'DME medical supplies',
    'DME AND APPLIANCES': 'DME and appliances', 'NURSING HOME': 'nursing home',
    'BH OUTPATIENT CLINIC': 'BH outpatient clinic', 'BH RESIDENTIAL TREATMENT': 'BH residential treatment',
    'BH PROFESSIONAL': 'BH professional',
}


# -------------------------------------------------------------------------------------
# PATCH 2  (CELL 33, insert right after `df = df[df['Impactful_Feature'] != 'Index']`
#           and BEFORE the groupby/agg)
# -------------------------------------------------------------------------------------
for c in NEW_COLS:
    df[c] = df['Feature'].apply(lambda f, c=c: get_mapping_field(f, c))

# (0) plain-English level in the label the LLM sees
df['Level_Plain'] = df['Feature'].apply(plain_level)
_root_label = df['Feature'].apply(lambda f: mapping_dict.get(get_mapping_root(f), ''))
_has_lvl = (df['Level_Plain'] != '') & (_root_label != '')
df.loc[_has_lvl, 'Impactful_Feature'] = _root_label[_has_lvl] + ': ' + df.loc[_has_lvl, 'Level_Plain']

# (a) drop raw fields, peer baselines, date parts, and features whose definition is unconfirmed
df = df[(df['Narrate_Role'] != 'Do not narrate') & (df['Validation_Status'] != 'Hold until confirmed')]

# (b) drop features that do not apply to the provider's claim category
def _applies(applies_to, category):
    a = str(applies_to).strip().upper()
    cat = str(category).strip().upper()
    if a in ('', 'ALL CATEGORIES'):
        return True
    if a.startswith('ALL EXCEPT'):
        return cat not in [x.strip() for x in a.replace('ALL EXCEPT', '').split(';')]
    return cat in [x.strip() for x in a.split(';')]

df = df[[_applies(a, c) for a, c in zip(df['Applies_To'], df['Category'])]]

# (c) one line per concept per provider; Driver outranks Context only, then Impact_Score.
#     The surviving line notes how many related measures also contributed.
_has_group = df['Concept_Group'].str.strip() != ''
_grp_keys = [provider_id, 'Category', 'Concept_Group']
df['Related_Measures'] = 0
df.loc[_has_group, 'Related_Measures'] = (
    df[_has_group].groupby(_grp_keys)['Feature'].transform('count') - 1)
_role_rank = df['Narrate_Role'].map({'Driver': 0, 'Context only': 1}).fillna(2)
_dedup = (df[_has_group].assign(_rr=_role_rank[_has_group])
            .sort_values(['_rr', 'Impact_Score'], ascending=[True, False])
            .drop_duplicates(subset=_grp_keys, keep='first')
            .drop(columns='_rr'))
df = pd.concat([_dedup, df[~_has_group]]).sort_index()

# (c2) PATCH 6 (optional): deterministic value sentence, filled by Python, never by the LLM.
#      Needs provider_feature_values with columns
#      [provider_id, 'Category', 'Feature', 'value', 'peer_median', 'peer_pct', 'n_claims'],
#      where peer_median / peer_pct are computed across providers in the SAME claim category.
#      Without that table, Value_Sentence stays empty and nothing else changes.
MIN_CLAIMS_FOR_VALUE = 20      # below this, a percentage rests on too few claims to quote

def _ordinal(n):
    n = int(round(float(n)))
    suf = 'th' if 10 <= n % 100 <= 20 else {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')
    return f"{n}{suf}"

def _fmt(x, vtype, feature=''):
    try:
        if x is None or pd.isna(x):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(x, str):
        return x
    x = float(x)
    if vtype == 'Dollars':
        return f"${x:,.0f}"
    if vtype in ('Share (0-1)', 'Category level', 'Tier label', 'Yes/No flag'):
        return f"{x:.1%}" if 0 <= x <= 1 else f"{x:,.0f}"   # provider-level share of lines
    if vtype in ('Count', 'Days', 'Miles', 'Units'):
        return f"{x:,.0f}" if x.is_integer() else f"{x:,.1f}"
    return f"{x:,.2f}"

def render_value_sentence(row):
    tpl = row.get('Value_Template', '')
    if not tpl:
        return ''
    if row.get('n_claims') is not None and not pd.isna(row.get('n_claims')) \
            and row['n_claims'] < MIN_CLAIMS_FOR_VALUE:
        return ''
    vals = {
        'value': _fmt(row.get('value'), row['Value_Type'], row['Feature']),
        'peer_median': _fmt(row.get('peer_median'), row['Value_Type'], row['Feature']),
        'peer_pct': _ordinal(row['peer_pct']) if row.get('peer_pct') is not None
                    and not pd.isna(row.get('peer_pct')) else None,
        'level': row.get('Level_Plain', ''),
        'category': CATEGORY_DISPLAY.get(str(row['Category']).upper(), str(row['Category'])),
    }
    needed = set(re.findall(r'\{(\w+)\}', tpl))
    if any(vals.get(k) in (None, '') for k in needed):
        return ''                                   # never send a half-filled sentence
    out = tpl.format(**vals)
    if row['Value_Type'] == 'Units' and 'wait' in str(row['Feature']).lower():
        hrs = float(row['value']) * 0.5             # T2007: 1 unit = 30 minutes
        out = out.replace(f"{vals['value']} wait-time units",
                          f"{vals['value']} wait-time units (about {hrs:,.1f} hours)", 1)
    return out

df['Value_Sentence'] = ''
if 'provider_feature_values' in globals():
    df = df.merge(provider_feature_values, on=[provider_id, 'Category', 'Feature'], how='left')
    df['Value_Sentence'] = df.apply(render_value_sentence, axis=1)

# (d) compose the LLM-facing text into the existing Clinical_Context slot.
#     Order keeps the caveat ahead of anything long, so nothing important is cut.
def _compose(row):
    parts = [row['Clinical_Context'] or row['Impactful_Feature_Purpose'],
             row['Value_Sentence'],
             row['Suspicious_Pattern'],
             f"Other explanations: {row['Benign_Explanations']}" if row['Benign_Explanations'] else '',
             f"Note: {row['Narrative_Caveat']}" if row['Narrative_Caveat'] else '',
             f"({int(row['Related_Measures'])} closely related measure(s) also contributed.)"
             if row['Related_Measures'] > 0 else '']
    text = ' '.join(p.strip() for p in parts if p and str(p).strip())
    if row['Narrate_Role'] == 'Context only':
        text = '(context only, not a reason for the flag) ' + text
    return text

df['Clinical_Context'] = df.apply(_compose, axis=1)


# -------------------------------------------------------------------------------------
# PATCH 3  (CELL 14, build_structured_context)
# -------------------------------------------------------------------------------------
# 3a. TOP SIGNAL must be a Driver. Replace
#         top_idx = int(np.argmax(scores_arr[:min(top_num, n)]))
#     with
#         _cands = [i for i in range(min(top_num, n))
#                   if not (clinical_contexts and i < len(clinical_contexts)
#                           and str(clinical_contexts[i]).startswith('(context only'))]
#         top_idx = (max(_cands, key=lambda i: scores_arr[i]) if _cands
#                    else int(np.argmax(scores_arr[:min(top_num, n)])))
#
# 3b. Composed text is 400 to 1,100 characters; the old [:200] cut off the direction,
#     the other explanations and the caveat. In format_grouped replace
#         line += f" — {item['description'][:200]}"
#     with
#         line += f" : {item['description'][:1200]}"
#
# 3c. score_summary wording (no ML vocabulary reaches the LLM):
#     "The top feature (score: X) dominates at Nx the median feature score (Y), indicating a
#      concentrated deviation."  ->  "One pattern stands out: the strongest signal (X) is N times
#      the typical signal (Y)."
#     "Feature contributions are distributed across multiple signals (...), suggesting a broad
#      pattern of unusual billing."  ->  "Several patterns contributed at similar levels
#      (strongest X, typical Y)."


# -------------------------------------------------------------------------------------
# PATCH 4  (CELL 16, build_prompt)
# -------------------------------------------------------------------------------------
# 4a. system_msg role: "You are a Medicaid fraud investigator writing case notes..."
#     -> "You are a Medicaid program integrity analyst writing case notes..."
#     ('fraud' in the system prompt invites the word your quality_check forbids.)
#
# 4b. Add to CONSTRAINTS (once, instead of on every feature line):
PAID_AND_DIRECTION_RULES = (
    "- All figures come from PAID claims. Write 'was paid for' or 'paid claims'. Paid is not proof "
    "of delivery: never say a service was or was not delivered, submitted, attempted, or denied.\n"
    "- Only state that the provider is higher or lower than peers when a number in the feature "
    "line, PEER BASELINES or PROVIDER PROFILE shows it; otherwise say the pattern was unusual "
    "compared with peers.\n"
    "- Describe at most 3 distinct patterns, strongest first.\n"
    "- Items marked '(context only...)' may be mentioned as 'In addition, ...' but never as the "
    "reason the provider was flagged.\n"
    "- When an 'Other explanations' note plausibly applies, mention it in one short clause.\n"
    "- Do not judge medical necessity; describe the paid-claims pattern only.\n"
)
#
# 4c. Opening template, in system_msg AND in the TASK line of user_msg:
#     "...was flagged because its [CATEGORY] billing..." -> "...because its [CATEGORY] paid claims..."
#
# 4d. Tier headers in user_msg:
#     "TIER 1 DRIVERS (primary contributors to anomaly score)" -> "TIER 1 PATTERNS (main reasons the
#      provider stood out)";  "TIER 2 DRIVERS (supporting contributors)" -> "TIER 2 PATTERNS (supporting)"


# -------------------------------------------------------------------------------------
# PATCH 5  (CELL 7 SCHEME_DEFINITIONS, and CELL 16 build_scheme_detection_prompt)
# -------------------------------------------------------------------------------------
# The prompt itself pushes medical-necessity judgments the dictionary avoids.
#   Upcoding:        "...pairing a diagnosis that does not clinically justify the equipment/service..."
#                 -> "...payment for a more expensive code than peers use for similar patients, or a
#                     diagnosis rarely paired with that equipment/service among peers..."
#   Modifier misuse: "...without clinical justification" -> "...in ways peers rarely use"
#   Service upcreep: "...without clinical justification" -> "...faster than peers serving similar patients"
#   Billing for non-covered services:
#                    "...submitting claims for services on holidays, weekends, or outside covered
#                     benefit periods"
#                 -> "...payment for services outside the member's covered benefit or eligibility, or
#                     for non-covered trip purposes. Weekend or holiday dates are a timing pattern, not
#                     non-covered, and never apply to nursing home or BH residential providers."
#   build_prompt TASK example:
#     OLD: "'pairing a sleep disorder diagnosis with nebulizer supplies suggests upcoding because the
#           diagnosis does not clinically justify this equipment.'"
#     NEW: "'pairing a sleep disorder diagnosis with nebulizer supplies is a combination peers are
#           rarely paid for, a pattern seen in upcoding.'"
#
#   Scheme-detection example mappings:
#     OLD: "- wait-time or mileage quantities far above peers -> excessive visits\n"
#     NEW: "- wait-time or mileage quantities far above peers -> mileage inflation\n"
#          "- trips with no same-day medical claim for the member -> phantom billing\n"
#          "- paid units or frequency above a policy limit -> frequency abuse\n"
#     OLD: "- holiday/weekend billing, unusual service timing -> billing for non-covered services\n"
#     NEW: "- trips to non-covered destinations, services outside eligibility -> billing for
#           non-covered services\n"
#     (Pharmacy trips are a covered NEMT purpose under AMPM 310-BB; do not map them to a scheme.)


# -------------------------------------------------------------------------------------
# SECURITY (CELL 37): a Hugging Face token is hard-coded in login("hf_..."). Rotate it and
# read it from a Databricks secret instead:
#     login(dbutils.secrets.get(scope="<scope>", key="hf_token"))
# -------------------------------------------------------------------------------------
