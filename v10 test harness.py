"""End-to-end test of llm_summarization_gpt_v4.0.ipynb with mock data in the real file layouts and a fake GPT-OSS.
Runs the notebook's own cells (skips %pip and replaces the model load). No GPU or Databricks needed.

Usage:  pip install pandas numpy openai-harmony
        python v4_test_harness.py <notebook.ipynb> <feature_mapping_v2_2.csv> [normal|batch|raw_referrals|resume|unmapped]
"""
import sys, os, json, types, contextlib, re, tempfile, shutil
import numpy as np
import pandas as pd

NB, MAPPING, MODE = sys.argv[1], sys.argv[2], (sys.argv[3] if len(sys.argv) > 3 else 'normal')
T = tempfile.mkdtemp(prefix='v4test_')
ROOT = 'sedo.senou@azahcccs.gov'
BASE = f'{T}/Workspace/Users/{ROOT}'
DATA = f'{BASE}/Projects/all_projects_predictions/data'
EXT = f'{BASE}/data/external data'
os.makedirs(DATA); os.makedirs(EXT)
SD = '2026-10-12'
open(f'{DATA}/scoring_date.txt', 'w').write(SD)
shutil.copy(MAPPING, f'{DATA}/feature_mapping_v2_4.csv')
rng = np.random.default_rng(11)
NEMT, NH = 'NEMT NOT IHS', 'NURSING HOME'

# ── Scoring files: 6-month and early (half of the 6-month lines repeated in the early file) ──
def scoring(cat, folder, pids, extra):
    rows = []
    for i, pid in enumerate(pids):
        n = 40 + 3 * i
        for k in range(n):
            r = {'SProv_ID': pid, 'UNIQUE_KEY': f'{pid}-{k}', 'claim_id': f'{pid}-c{k // 2}', 'MEMBER_KEY': f'm{pid}-{k % 7}',
                 'PMT_AMT': float(30 + i + k % 5), 'QUANTITY_PAID': 1 + k % 3, 'PROC_CD': ['A0130', 'T2007'][k % 2],
                 'proc_cd_category': ['Transportation', 'Wait time'][k % 2], 'diag_1_level_I': 'Endocrine diseases',
                 'GENDER': 'F' if k % 3 else 'M', 'POS_DESC': ['Home', 'Office'][k % 2] if pid != pids[0] else 'Home',
                 'is_major_holiday_service': float(k % 15 == 0), 'line_item_cnt': 2 + k % 3,
                 'global_procedure_zscore_class': 'Rare Anomaly' if (i == 0 and k % 4 == 0) else 'Normal',
                 'unrelated_column': 'x'}
            r.update(extra(i, pid, k))
            rows.append(r)
    df = pd.DataFrame(rows)
    d = f'{BASE}/Projects/{folder}/scoring_data'
    os.makedirs(d)
    df.to_csv(f'{d}/rolling_6m_score_monthly_scoring_file.csv.gz', index=False, compression='gzip')
    early = pd.concat([df.iloc[::2], df.iloc[:5].assign(UNIQUE_KEY=lambda x: x['UNIQUE_KEY'] + '-new')])
    early.to_csv(f'{d}/early_score_monthly_scoring_file.csv.gz', index=False, compression='gzip')
    return len(df.iloc[::2])

nemt_pids = list(range(2001, 2031))
# avg_trips_per_patient: provider-level constant; 2001 highest, 2006 and 2008 lowest
trips = {pid: 9.0 - 0.2 * i for i, pid in enumerate(nemt_pids)}
trips[2006] = 0.5; trips[2008] = 0.4; trips[2004] = 8.9     # 2004 above the confirmation percentile
OVERLAP = scoring(NEMT, 'nemt_not_ihs', nemt_pids, lambda i, pid, k: {'avg_trips_per_patient': trips[pid]})
OVERLAP += scoring(NH, 'nursing_home', list(range(3001, 3021)), lambda i, pid, k: {'avg_trips_per_patient': np.nan})

# ── Exploded importance file ──
imp = []
def add(pid, cat, level, feat, rank, score, tier='top_50', fcat='Billing Pattern', agree=1, tf=2, models=2,
        which_tf='rolling_3m_score,rolling_6m_score'):
    imp.append({'Category': cat, 'Servicing_Provider_ID': pid, 'Analysis_Level': level, 'Importance_Tier': tier,
                'Analysis_Levels_In_Agreement': agree, 'Which_Analysis_Levels': level, 'Feature_Name': feat,
                'Impactful_Feature': feat.replace('_', ' '), 'Feature_Category': fcat, 'Feature_Rank_The_Provider': rank,
                'Overall_Importance_Score': score, 'Timeframe_Weighted_Importance': score / 1.15,
                'Avg_Importance_Score': score / 1.2, 'Peak_Importance_Score': 1.0, 'Avg_Entities_Affected': 1,
                'Max_Entity_Affected': 1, 'Pct_Of_Entities_Affected': np.nan, 'Scoring_Runs_Detected_In': 2,
                'Timeframes_Detected_In': tf, 'Models_Detected_In': models, 'Which_Timeframe_Flagged': which_tf,
                'Which_Model_Flagged': 'saved_models', 'Share_Of_Provider_Signal': 0.1, 'Cumulative_Signal_Coverage': 0.1,
                'Scoring_Date': SD})
DEMO = 'Patient Demographics'
def good(pid, cat=NEMT):
    add(pid, cat, 'provider', 'avg_trips_per_patient', 1, 1.05, tf=4, models=4,
        which_tf='early_score,final_score,rolling_3m_score,rolling_6m_score')
    add(pid, cat, 'patient', 'avg_trips_per_patient', 4, 0.30, agree=2)            # same feature, other level
    add(pid, cat, 'claim', 'POS_DESC_Home', 2, 0.80, tier='top_80', fcat='Provider Characteristics')
    add(pid, cat, 'claim', 'line_item_cnt_42', 3, 0.50, tier='top_80')
    add(pid, cat, 'patient', 'GENDER_F', 5, 0.20, tier='top_100', fcat=DEMO)
for pid in (2001, 2002, 2004, 2005, 2010, 2011, 2012):
    good(pid)
add(2001, NEMT, 'patient', 'avg_qty_per_patient', 6, 0.15, tier='top_80')       # level-dependent measure, patient level
# 2003: patient mix (like 130676)
for r, (lvl, f, s) in enumerate([('provider', 'GENDER_F', 1.06), ('provider', 'RESERVATION_IND_N', 0.95),
                                  ('patient', 'GENDER_M', 0.36), ('patient', 'patient_age_group_medicaid_adults', 0.30),
                                  ('provider', 'avg_trips_per_patient', 0.10)], start=1):
    add(2003, NEMT, lvl, f, r, s, fcat=DEMO if 'trips' not in f else 'Visit & Service Frequency')
# 2006: main pattern with no scheme link -> possible new pattern; avg_trips_per_patient is low -> background
add(2006, NEMT, 'provider', 'dominant_proc_pct', 1, 0.9); add(2006, NEMT, 'provider', 'avg_trips_per_patient', 2, 0.7)
# 2007: drivers only outside the 80% coverage -> no main pattern
add(2007, NEMT, 'provider', 'avg_billed_miles', 1, 0.9, tier='top_100'); add(2007, NEMT, 'patient', 'GENDER_M', 2, 0.4, tier='top_100', fcat=DEMO)
# 2008: linked driver with no value (possible scheme), harmless trips driver
add(2008, NEMT, 'provider', 'avg_billed_miles', 1, 0.9); add(2008, NEMT, 'provider', 'avg_trips_per_patient', 2, 0.8)
# 2009: held feature first, then a normal driver
add(2009, NEMT, 'provider', 'pct_round_trips', 1, 1.1); add(2009, NEMT, 'provider', 'pct_billed_ends_0_or_5', 2, 0.6)
# 2010 also has missing-value levels at the top
add(2010, NEMT, 'claim', 'POS_DESC_Null', 0, 1.3, fcat='Provider Characteristics')
add(2010, NEMT, 'claim', 'procmodif_proc_pair_-1,-1,-1,-1__0', 0, 1.2, fcat='Procedure & Modifier Behavior')
add(2050, NEMT, 'provider', 'avg_trips_per_patient', 1, 1.0)                     # not referred
add(2010, NEMT, 'provider', 'index', 0, 2.0)                                       # row-index artifact
imp.append({**imp[0], 'Scoring_Date': '2026-09-12'})                               # other scoring date
# Nursing home
for r, (lvl, f, s) in enumerate([('provider', 'GENDER_F', 1.06), ('provider', 'RESERVATION_IND_N', 0.95),
                                  ('patient', 'GENDER_M', 0.36), ('patient', 'patient_age_group_medicaid_adults', 0.30)], start=1):
    add(3001, NH, lvl, f, r, s, fcat=DEMO)
for pid in (3002, 3004, 3005, 3006):
    add(pid, NH, 'provider', 'pct_loa_bed_days', 1, 0.9); add(pid, NH, 'claim', 'avg_cost_per_bed_day', 2, 0.6, tier='top_80')
    add(pid, NH, 'provider', 'total_bed_days', 3, 0.3, tier='top_100')
add(3003, NH, 'claim', 'is_major_holiday_service', 1, 0.9); add(3003, NH, 'provider', 'total_bed_days', 2, 0.5)
if MODE == 'unmapped':
    add(2001, NEMT, 'provider', 'brand_new_feature_xyz', 9, 0.1)
# Same code pattern with 0 and -1 placeholders (9 such pairs in the real file): must take one main slot, not two
add(2011, NEMT, 'claim', 'procmodif_proc_pair_TN,0,0,0__A0130', 1, 1.2, fcat='Procedure & Modifier Behavior')
add(2011, NEMT, 'claim', 'procmodif_proc_pair_TN,-1,-1,-1__A0130', 1, 1.1, fcat='Procedure & Modifier Behavior')
pd.DataFrame(imp).to_csv(f'{DATA}/explained_top_features_{SD}.csv', index=False)

# ── Referrals ──
ref = [{'Servicing_Provider_ID': pid, 'Category': NEMT, 'Servicing_Provider_Anomaly_Score': 1 - 0.01 * (pid - 2001),
        'Provider_Perspective_Score': 0.5, 'Patient_Perspective_Score': 0.4, 'Visit_Perspective_Score': 0.3,
        'Claim_Perspective_Score': 0.2, 'Scoring_Date': SD} for pid in range(2001, 2014)]          # 2013: no features
ref += [{'Servicing_Provider_ID': pid, 'Category': NH, 'Servicing_Provider_Anomaly_Score': 1 - 0.01 * (pid - 3001),
         'Provider_Perspective_Score': 0.5, 'Patient_Perspective_Score': 0.9, 'Visit_Perspective_Score': 0.3,
         'Claim_Perspective_Score': 0.2, 'Scoring_Date': SD} for pid in range(3001, 3007)]
ref.append({**ref[0], 'Servicing_Provider_Anomaly_Score': 0.5})                      # duplicate row
ref_df = pd.DataFrame(ref)
if MODE == 'raw_referrals':
    for cat, folder, prefix in [(NEMT, 'nemt_not_ihs', 'nemt_not_ihs'), (NH, 'nursing_home', 'nh')]:
        d = f'{BASE}/Projects/{folder}/referrals'; os.makedirs(d)
        x = ref_df[ref_df.Category == cat].rename(columns={
            'Servicing_Provider_ID': 'SProv_ID', 'Servicing_Provider_Anomaly_Score': 'final_weighted_provider_score',
            'Provider_Perspective_Score': 'score_provider', 'Patient_Perspective_Score': 'score_patient',
            'Visit_Perspective_Score': 'score_visit', 'Claim_Perspective_Score': 'score_claim'})
        x['category'] = prefix; x['monthly_payout'] = 30000
        x.drop(columns=['Category', 'Scoring_Date']).to_csv(f'{d}/{prefix}_monthly_anomaly_detection_referrals.csv', index=False)
else:
    ref_df.to_csv(f'{DATA}/referred_providers_{SD}.csv', index=False)

# ── Reference tables ──
pd.DataFrame({'ICD10 Code': ['Z89512'], 'Description': ['Acquired absence of right leg below knee']}).to_csv(f'{EXT}/icd10_categories_description.csv', index=False)
pd.DataFrame({'diag_cd_root': ['Z89'], 'category_level_III': ['acquired absence of limb']}).to_csv(f'{EXT}/icd_10_diagnosis_code_root_and_category.csv', index=False)
pd.DataFrame({'HCPCS Codes': ['A0130', 'T2007'], 'Description': ['Non-emergency transportation: wheelchair van', 'Transportation waiting time']}).to_csv(f'{EXT}/hcpcs_code_description.csv', index=False)
pd.DataFrame({'CPT Code': ['99213'], 'Description': ['Office visit']}).to_csv(f'{EXT}/cpt_codes_description.csv', index=False)
pd.DataFrame({'Modifier': ['RR'], 'Description': ['Rental']}).to_csv(f'{EXT}/procedure_modifiers.csv', index=False)

# ── Stubs: torch, transformers, huggingface_hub, pyspark ──
class FT:
    def __init__(self, rows): self.rows = [list(r) for r in rows]
    def to(self, device): return self
    @property
    def shape(self): return (len(self.rows), len(self.rows[0]))
    def __getitem__(self, i): return FR(self.rows[i])
    def __iter__(self): return (FR(r) for r in self.rows)
class FR(list):
    def __getitem__(self, s):
        r = list.__getitem__(self, s)
        return FR(r) if isinstance(s, slice) else r
    def tolist(self): return list(self)
def stub(name, **attrs):
    m = types.ModuleType(name); m.__dict__.update(attrs); sys.modules[name] = m; return m
stub('torch', no_grad=contextlib.nullcontext, tensor=lambda rows: FT(rows),
     cuda=types.SimpleNamespace(is_available=lambda: False, empty_cache=lambda: None))
stub('transformers', AutoTokenizer=object, AutoModelForCausalLM=object)
stub('huggingface_hub', login=lambda *a, **k: None)
stub('pyspark'); stub('pyspark.sql', SparkSession=object)
stub('pyspark.sql.types', StructType=object, StructField=object, StringType=object)

# ── Fake GPT-OSS ──
from openai_harmony import Message, Role
STATE = {'calls': {}, 'registry': {}, 'batch_sizes': [], 'prompts': {}}
def good_note(pid, cat, user):
    m1 = re.search(r'MAIN PATTERNS[^\n]*\n\s+1\. ([^:\[\n]+)(?: \([^)]*\))?(?: \[[^\]]*\])?: ([^\n]+)', user)
    label, body = m1.group(1).strip(), m1.group(2)
    val = re.search(r'(The provider averaged .*?\)\.)|(Among .*? percentile[^.]*\.)', body)
    stab = re.search(r'(Present [^.;]+)', body)
    s = f"Provider {pid} was flagged because its {cat} paid claims stood out on one main measure: {label.lower()}. "
    s += (val.group(0) + ' ') if val else 'No provider value is available for this measure, so it was unusual compared with peers. '
    if stab:
        s += f"This pattern was {stab.group(1).lower()}. "
    tp = re.search(r'Total paid: (\$[\d,.]+) for this provider; peer median (\$[\d,.]+); (\d+(?:st|nd|rd|th)) percentile', user)
    if tp:
        s += (f"For size, the provider was paid {tp.group(1)} in total against a peer median of {tp.group(2)}, "
              f"at the {tp.group(3)} percentile. ")
    s += ("Other explanations, such as a larger patient base or longer appointments, could account for part of the difference. ")
    rec = re.search(r'RECORDS THAT WOULD CLARIFY THE MAIN PATTERNS: ([^\n]+)', user)
    if rec:
        s += 'Records that would clarify this: ' + rec.group(1)
    return s
def respond(pid, cat, user, attempt):
    if 'EVIDENCE BRIEF:' in user and 'PRE-CLASSIFIED CANDIDATES' in user:          # Layer 2
        STATE.setdefault('layer2', []).append(pid); STATE.setdefault('l2prompts', {})[pid] = user
        if pid == '2004':      # format noise: hyphen and case in the name, 'Medium evidence', a markdown table row, a header
            return {'final': 'scheme | evidence_strength | rationale\n---|---|---\nExcessive-Visits | Medium evidence | trips per rider far above peers\n'
                             '| upcoding | high | menu scheme, not linked |\nkickback | high | invented', 'n': 40}
        if pid == '2006':
            return {'final': 'The provider shows unusual billing.', 'n': 40}      # no pipe lines: unusable answer
        if pid == '3002':
            return {'final': '- Excessive_Visits (medium) | medium | fuzzy name', 'n': 40}
        return {'final': 'excessive visits | medium | trips per rider far above peers', 'n': 40}
    if pid == '2002':
        return {'final': (f"Provider {pid} was flagged because its {cat} billed claims show a markedly high score of 0.860, "
                          "about 4 times the peer level, consistent with upcoding. Services were not delivered on 12,345 trips, "
                          "which appears deliberate. The provider should be investigated further."), 'n': 120}
    if pid == '2004' and attempt == 1:
        return {'final': None, 'n': 'limit'}
    if pid == '2005':
        return {'raise': True}
    return {'final': good_note(pid, cat, user), 'n': 150}
class FakeTok:
    pad_token_id, eos_token_id, padding_side = 0, 0, 'right'
    def apply_chat_template(self, messages, return_tensors=None, add_generation_prompt=True, return_dict=True, **kw):
        tid = 100000 + len(STATE['registry'])
        STATE['registry'][tid] = {'messages': messages}
        ids = [tid] * 10
        return {'input_ids': FT([ids]), 'attention_mask': FT([[1] * 10])} if return_tensors else {'input_ids': ids}
    def decode(self, ids, skip_special_tokens=False):
        spec = STATE['registry'][ids[0]]
        out = '<|channel|>analysis<|message|>Thinking<|end|>'
        return out + ('<|start|>assistant<|channel|>final<|message|>' + spec['final'] + '<|return|>' if spec.get('final') else '')
class FakeModel:
    device = 'cpu'
    def generate(self, input_ids, attention_mask=None, max_new_tokens=800, do_sample=False, pad_token_id=None):
        STATE['batch_sizes'].append(len(input_ids.rows))
        outs = []
        for row in input_ids.rows:
            messages = STATE['registry'][row[-1]]['messages']
            user = messages[1]['content']
            pid, cat = re.search(r'PROVIDER: (\S+) \| CLAIM CATEGORY: ([^\n]+)', user).groups()
            if not ('EVIDENCE BRIEF:' in user and 'PRE-CLASSIFIED CANDIDATES' in user):
                STATE['calls'][pid] = STATE['calls'].get(pid, 0) + 1
                STATE['prompts'][pid] = messages
            spec = respond(pid, cat.strip(), user, STATE['calls'].get(pid, 0))
            if spec.get('raise'):
                raise RuntimeError('CUDA out of memory (simulated)')
            n = max_new_tokens if spec['n'] == 'limit' else spec['n']
            tid = 200000 + len(STATE['registry'])
            STATE['registry'][tid] = spec
            outs.append(row + [tid] * n)
        return FT(outs)
class FakeHarmony:
    def parse_messages_from_completion_tokens(self, tokens, role=None, strict=True):
        assert role == Role.ASSISTANT
        spec = STATE['registry'][tokens[0]]
        msgs = [Message.from_role_and_content(Role.ASSISTANT, 'Thinking').with_channel('analysis')]
        if spec.get('final'):
            msgs.append(Message.from_role_and_content(Role.ASSISTANT, spec['final']).with_channel('final'))
        return msgs
    def stop_tokens_for_assistant_actions(self): return []

# ── Run the notebook cells ──
SETTING_CHANGE = {}
def run(ns_extra=None):
    nb = json.load(open(NB))
    ns = {'__name__': '__main__', 'dbutils': None}
    for i, c in enumerate(nb['cells']):
        if c['cell_type'] != 'code':
            continue
        src = ''.join(c['source']).replace('/Workspace/Users/', f'{T}/Workspace/Users/')
        if src.lstrip().startswith('%pip'):
            continue
        if 'AutoModelForCausalLM.from_pretrained' in src:
            ns.update(tokenizer=FakeTok(), model=FakeModel(), harmony_enc=FakeHarmony())
            continue
        for a, b in SETTING_CHANGE.items():
            src = src.replace(a, b)
        if MODE == 'batch' and 'GEN_BATCH_SIZE = 1' in src:
            src = src.replace('GEN_BATCH_SIZE = 1\n', 'GEN_BATCH_SIZE = 3\n')
        src = src.replace('NARRATE_FLOOR = 10 ', 'NARRATE_FLOOR = 5 ')         # small fixture: floor 5 like v4
        src = src.replace('LAYER2_AUDIT_RATE = 0.05 ', 'LAYER2_AUDIT_RATE = 1.0 ')  # validate every narrative in tests
        try:
            exec(compile(src, f'cell_{i}', 'exec'), ns)
        except ValueError as e:
            if MODE == 'unmapped' and 'no dictionary row' in str(e):
                print('EXPECTED STOP at cell', i, ':', str(e).splitlines()[0]); print('ALL PASS (unmapped stop)'); sys.exit(0)
            raise
    return ns

ns = run()
if MODE == 'unmapped':
    sys.exit('FAIL: run did not stop on an unmapped feature')
out, feat, prov = ns['output_df'], ns['feat'], ns['providers']
if os.environ.get('DUMP'):
    _d = os.environ['DUMP']; os.makedirs(_d, exist_ok=True)
    for _pid, _m in STATE['prompts'].items():
        open(f'{_d}/prompt_{_pid}.txt', 'w').write(_m[0]['content'] + '\n\n=== USER ===\n' + _m[1]['content'])
    for _pid, _u in STATE.get('l2prompts', {}).items():
        open(f'{_d}/layer2_{_pid}.txt', 'w').write(_u)
    out.to_csv(f'{_d}/output.csv', index=False)
o = out.set_index(out['Servicing_Provider_ID'].astype(str))
f = feat.assign(pid=feat['Servicing_Provider_ID'].astype(str))
def note_type(pid): return o.at[str(pid), 'Note_Type']
checks = {
    'one note per referred provider (19), none for 2050': len(out) == 19 and '2050' not in o.index,
    'patient mix 2003 and 3001, no LLM call': note_type(2003) == 'patient_mix' and note_type(3001) == 'patient_mix'
        and '2003' not in STATE['calls'] and '3001' not in STATE['calls'],
    'patient mix note text': 'patient mix' in o.at['3001', 'LLM_Summary'],
    'NEMT: 5 selected (floor), patient mix and no-main do not take slots':
        set(prov[(prov.Category == NEMT) & prov.Narrate_Selected]['Servicing_Provider_ID']) == {'2001', '2002', '2004', '2005', '2006'},
    '2007 has no main pattern -> evidence_only': note_type(2007) == 'evidence_only',
    '2008-2012 below quota -> evidence_list with values': all(note_type(p) == 'evidence_list' for p in (2008, 2009, 2010, 2011, 2012))
        and 'stood out on: (1)' in o.at['2010', 'LLM_Summary'] and 'percentile' in o.at['2010', 'LLM_Summary'],
    'evidence notes differ by provider': len({o.at[str(p), 'LLM_Summary'].split('stood out on')[1] for p in (2008, 2009, 2010)}) == 3,
    '2013 referred without features -> evidence_only': note_type(2013) == 'evidence_only',
    'NH: 3003 (holiday not applicable to NH) evidence_only; 3002/4/5/6 narrative':
        note_type(3003) == 'evidence_only' and all(note_type(p) == 'narrative' for p in (3002, 3004, 3005, 3006)),
    '2001 passes QA and confirms a scheme with strength': o.at['2001', 'QA_Status'] == 'pass'
        and re.search(r'(excessive visits|phantom billing) \((high|medium|low) evidence\)', o.at['2001', 'Detected_Schemes']) is not None,
    '2001 mixed stability -> STABILITY line names the patterns': 'STABILITY: Main pattern 1 was present in most review windows; main patterns 2 and 3 were present only in recent review windows.' in STATE['prompts']['2001'][1]['content'],
    '2001 persistent pattern -> stability in prompt and note': 'resent in most review windows' in STATE['prompts']['2001'][1]['content']
        and o.at['2001', 'Main_Pattern_Stability'].startswith('persistent') and 'present in most review windows' in o.at['2001', 'LLM_Summary'],
    '2001 note quotes the main pattern value (9.00 lines per rider)': '9.00 paid claim lines per rider' in o.at['2001', 'LLM_Summary'],
    '2002 planted defects flagged': o.at['2002', 'QA_Status'] == 'needs_review'
        and all(k in o.at['2002', 'QA_Flags'] for k in ('billed', 'intensifier', 'delivery claim', 'recommendation',
                                                         'ungrounded_number', 'unsupported_scheme', 'score', 'intent')),
    '2004 retried and passed': STATE['calls'].get('2004', 0) >= 2 and o.at['2004', 'QA_Status'] == 'pass',
    '2004 Layer 2 ran, invented scheme rejected, linked scheme kept': '2004' in STATE.get('layer2', [])
        and o.at['2004', 'Layer2_Used'] in (True, 'True') and 'kickback' in o.at['2004', 'Layer2_Rejected']
        and 'excessive visits' in o.at['2004', 'Detected_Schemes'] and o.at['2004', 'Layer2_Dropped'] == '',
    '2004 menu scheme without a link goes to Layer2_Suggested, not Detected': o.at['2004', 'Layer2_Suggested'] == 'upcoding'
        and 'upcoding' not in o.at['2004', 'Detected_Schemes'],
    '2006 unusable Layer 2 answer -> deterministic schemes kept, flagged': o.at['2006', 'Layer2_Used'] in (False, 'False')
        and 'no usable answer' in o.at['2006', 'Layer2_Note'],
    'quarantined narratives (2002, 2005) get an evidence fallback in the client file':
        set(ns['client_df'].loc[ns['client_df']['Note_Type'] == 'evidence_fallback', 'Servicing_Provider_ID'].astype(str)) == {'2002', '2005'}
        and 'stood out on' in ns['client_df'].loc[ns['client_df']['Servicing_Provider_ID'].astype(str) == '2002', 'LLM_Summary'].iloc[0],
    'client file carries no score/model/scoring words': not ns['client_df']['LLM_Summary'].str.contains(r'\b(scor(e|es|ing)|models?|anomal)', case=False, regex=True).any(),
    'Scheme_Strength follows Layer 2 (2005 not confirmed -> empty)': o.at['2005', 'Scheme_Strength'] in ('', None) or pd.isna(o.at['2005', 'Scheme_Strength']),
    '3002 fuzzy scheme name matched and rejected as unlinked (NH)': 'excessive visits' in o.at['3002', 'Layer2_Rejected'],
    'Layer 2 never raises evidence above the deterministic strength': o.at['2004', 'Detected_Schemes'].startswith('excessive visits (medium'),
    'Layer 2 prompt carries the brief, not the narrative': all('Provider 2001 was flagged' not in m for m in [json.dumps(STATE.get('l2prompts', {}))]),
    '2005 linked scheme below the gate: Layer 2 cannot add it to Detected, lands in Layer2_Added and Possible':
        'excessive visits' in o.at['2005', 'Layer2_Added'] and 'excessive visits' in o.at['2005', 'Possible_Schemes']
        and 'evidence)' not in o.at['2005', 'Detected_Schemes'],
    '2005 generation failure kept the run going, evidence note written': o.at['2005', 'QA_Status'] == 'generation_failed'
        and 'stood out on' in o.at['2005', 'LLM_Summary'],
    '2006 no scheme link -> possible new pattern; low trips driver is background':
        o.at['2006', 'Detected_Schemes'] == ns['NEW_PATTERN_LABEL']
        and f[(f.pid == '2006') & (f.Feature == 'avg_trips_per_patient')]['Role'].tolist() == ['Context only'],
    '2008 linked driver without value -> not confirmed, possible listed':
        o.at['2008', 'Detected_Schemes'].startswith('Not confirmed') and 'mileage inflation' in o.at['2008', 'Possible_Schemes'],
    '2009 held feature not shown, next driver is main':
        f[(f.pid == '2009') & (f.Feature == 'pct_round_trips')]['Role'].tolist() == ['Not narrated']
        and f[(f.pid == '2009') & (f.Feature == 'pct_billed_ends_0_or_5')]['Shown_As'].tolist() == ['Main'],
    '2010 missing-value levels not narrated, index dropped':
        set(f[(f.pid == '2010') & f.Feature.str.contains('Null|__0')]['Role']) == {'Not narrated'}
        and not (f.Feature == 'index').any(),
    'same feature on two levels kept once (best level provider), N_Levels 2':
        f[(f.pid == '2001') & (f.Feature == 'avg_trips_per_patient')]['Best_Level'].tolist() == ['provider']
        and f[(f.pid == '2001') & (f.Feature == 'avg_trips_per_patient')]['N_Levels'].tolist() == [2],
    'count level line_item_cnt_42 labelled with its number':
        f[(f.pid == '2001') & (f.Feature == 'line_item_cnt_42')]['Label'].str.contains('42').all(),
    'duplicate scoring lines removed': ns['scoring_stats']['duplicate_lines_removed'] == OVERLAP,
    'value sentence for avg_trips_per_patient (2001) with 99th-ish percentile':
        'percentile' in f[(f.pid == '2001') & (f.Feature == 'avg_trips_per_patient')]['Value_Sentence'].iloc[0],
    'no scores or ML words in any prompt': not any(re.search(r'\b(scor(e|es|ing)|anomaly|z-score|shap|importance|model)\b', m[1]['content'], re.I)
                                                   for m in STATE['prompts'].values()),
    'unrelated scoring column not read': 'unrelated_column' not in ns['df_scoring_master'].columns,
    'records sentence reaches the prompt and the note': 'RECORDS THAT WOULD CLARIFY' in STATE['prompts']['2001'][1]['content']
        and 'Records that would clarify this' in o.at['2001', 'LLM_Summary'],
    'quarantine holds needs_review only': set(ns['quarantine_df']['QA_Status']) <= {'needs_review', 'generation_failed'}
        and '2002' in set(ns['quarantine_df']['Servicing_Provider_ID'].astype(str)),
    'QA: percentile is not a percent (98% with only 98th in the prompt is ungrounded)':
        any(x.startswith('ungrounded_number:98%') for x in ns['quality_check'](
            o.at['2001', 'LLM_Summary'].replace('98th percentile', '98%', 1), '2001', NEMT, STATE['prompts']['2001'][1]['content'], ['excessive visits'])),
    'QA: a policy-limit percentile from a description is not a provider percentile':
        'ungrounded_number:90th' in ns['quality_check'](
            o.at['2001', 'LLM_Summary'] + ' The facility was at the 90th percentile.', '2001', NEMT,
            STATE['prompts']['2001'][1]['content'] + ' Description: at or above the 90th percentile (top 10%).', ['excessive visits'],
            None, ns['facts_text'](ns['briefs'][('2001', NEMT)])),
    'QA: exemplar sentences copied into a note are flagged': 'exemplar_copied' in ns['quality_check'](
            o.at['2001', 'LLM_Summary'] + ' Riders in dialysis or day programs legitimately take several round trips a week, which could account for part of the difference.',
            '2001', NEMT, STATE['prompts']['2001'][1]['content'], ['excessive visits'], None, None, None, STATE['prompts']['2001'][0]['content']),
    'QA: note that skips main pattern 1 is flagged': 'main_pattern_missing' in ns['quality_check'](
            re.sub(r'paid claim lines per rider', 'wait time', o.at['2001', 'LLM_Summary'], flags=re.I), '2001', NEMT,
            STATE['prompts']['2001'][1]['content'], ['excessive visits'], None, None, ('Paid claim lines per rider', 9.0)),
    'QA: all-stopword label does not trip the same-sentence rule': 'pass' in ns['quality_check'](
            'Provider 2001 was flagged because its NEMT NOT IHS paid claims showed 14.00 paid claim lines per patient against a peer median of 6.50, at the 97th percentile among peers. '
            + ' '.join(['The pattern was unusual compared with peers.'] * 12), '2001', NEMT,
            'PROVIDER: 2001 | CLAIM CATEGORY: NEMT NOT IHS\nThe provider averaged 14.00 paid claim lines per patient (peer median 6.50; 97th percentile among peers).', [],
            {'Paid claim lines per patient': [14.0, 6.5, 97.0]}),
    'schemes table has 2001': '2001' in set(ns['schemes_df']['Servicing_Provider_ID'].astype(str)),
}
if MODE == 'batch':
    checks['batched generation used'] = max(STATE['batch_sizes']) > 1
for k, v in checks.items():
    print(('PASS ' if v else 'FAIL ') + k)
checks['level-dependent measure names its analysis level'] = '(measured per patient' in STATE['prompts']['2001'][1]['content']
checks['Main_Pattern_Confidence column filled'] = o.at['2001', 'Main_Pattern_Confidence'] in ns['CONFIDENCE_ORDER']
checks['code levels with 0 and -1 placeholders take one slot'] = (
    f[(f.pid == '2011') & f.Feature.str.startswith('procmodif')]['Shown_As'].tolist().count('Main') == 1)
if MODE == 'resume':
    ckpt = ns['CHECKPOINT_FILE']
    with open(ckpt, 'a') as fh:
        fh.write('{"summary": {"Servicing_Provider_ID": "20')          # crash in the middle of a line
    STATE['calls'].clear()
    ns2 = run()
    print(('PASS ' if set(STATE['calls']) == {'2005'} else 'FAIL ')
          + f"rerun reuses written notes, retries the failed one, survives a cut-off line (calls: {STATE['calls']})")
    print(('PASS ' if len(ns2['output_df']) == 19 else 'FAIL ') + 'resume output complete')
    STATE['calls'].clear()
    SETTING_CHANGE['MAX_BACKGROUND = 3\n'] = 'MAX_BACKGROUND = 0\n'
    ns3 = run()
    print(('PASS ' if {'2001', '2002', '3002'} <= set(STATE['calls']) else 'FAIL ')
          + f"changed settings regenerate the notes (calls: {sorted(STATE['calls'])})")
json.dump({'calls': STATE['calls'], 'prompts': STATE['prompts']}, open('./fake_llm_state.json', 'w'), indent=1, default=str)
out.to_csv('./output_df.csv', index=False)
feat.to_csv('./features.csv', index=False)
print('FAILED' if not all(checks.values()) else 'ALL PASS', '| TEMP', T)
