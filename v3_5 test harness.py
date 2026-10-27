"""End-to-end test of llm_summarization_gpt_v3.5.ipynb with mock data and a fake GPT-OSS.
Runs the notebook's own cells in order (skips %pip and the real model-load cell). No GPU or Databricks needed.

Usage:  pip install pandas numpy openai-harmony
        python v3.5_test_harness.py llm_summarization_gpt_v3.5.ipynb feature_mapping.csv [normal|adversarial|unmapped]
normal: clean note, planted defects, cut-off answer + retry, total failure, no-driver providers
adversarial: hidden drivers, '_0' flags, Normal vs Rare tiers, scaled columns, duplicate/blank/float IDs
unmapped: an unknown feature must stop the run"""
import sys, os, json, types, contextlib, re, gzip, tempfile, shutil, random
import numpy as np
import pandas as pd

NB, MAPPING_CSV, MODE = sys.argv[1], sys.argv[2], (sys.argv[3] if len(sys.argv) > 3 else 'normal')
T = tempfile.mkdtemp(prefix='v35test_')
ROOT = 'sedo.senou@azahcccs.gov'
BASE = f'{T}/Workspace/Users/{ROOT}'
DATA = f'{BASE}/Projects/all_projects_predictions/data'
os.makedirs(DATA); os.makedirs(f'{BASE}/data/external data')
open(f'{DATA}/scoring_date.txt', 'w').write('2026-10-01')
shutil.copy(MAPPING_CSV, f'{DATA}/feature_mapping.csv')
rng = np.random.default_rng(7)

# ── Mock scoring files (claim lines) for all 8 categories ─────────────────────────────────────
CATS = {'nemt_not_ihs': 'NEMT NOT IHS', 'nemt_ihs': 'NEMT IHS', 'dme_supplies': 'DME MEDICAL SUPPLIES',
        'dme_and_appliances': 'DME AND APPLIANCES', 'nursing_home': 'NURSING HOME',
        'bh_outpatient_clinic': 'BH OUTPATIENT CLINIC', 'bh_residential_treatment': 'BH RESIDENTIAL TREATMENT',
        'bh_professional_counseling': 'BH PROFESSIONAL'}
TIERS = ['Normal', 'Mild Anomaly', 'Anomaly', 'Rare Anomaly']
lines_by_cat = {}
for folder, cat in CATS.items():
    rows = []
    for p in range(30):
        pid = 1000 + p + 100 * list(CATS).index(folder)
        n_lines = 60 + 5 * p
        trips = round(2 + p / 10, 2)
        for k in range(n_lines):
            rows.append({'SProv_ID': pid, 'claim_id': f'{pid}-{k // 2}', 'UNIQUE_KEY': f'{pid}-{k}',
                         'PMT_AMT': float(20 + p + (k % 7)), 'QUANTITY_PAID': 1 + (k % 3),
                         'MEMBER_KEY': f'm{pid}-{k % (5 + p // 3)}',
                         'PROC_CD': ['A0130', 'T2007', 'A0425', 'E1390'][k % 4],
                         'proc_cd_category': ['Transportation', 'Wait time', 'Mileage', 'DME'][k % 4],
                         'diag_1_level_I': ['Diseases of the genitourinary system', 'Endocrine diseases'][k % 2],
                         'global_procedure_zscore_class': TIERS[3] if (p >= 25 and k % 3 == 0) else TIERS[k % 2],
                         'wait_time_zscore_class': 'Insanely Rare' if (p == 29 and k % 5 == 0) else 'Normal',
                         'is_christmas_day': 1 if (k % 40 == 0 or (p >= 28 and k % 9 == 0)) else 0,
                         'avg_trips_per_patient': trips,                      # provider-level, constant
                         'mileage_zscore': float(rng.normal()),               # varies within provider
                         'POS_DESC': ['Home', 'Office'][k % 2],
                         'is_major_holiday_service': 0 if p == 22 else int(k % 10 == 0),
                         'avg_pmt_per_patient': p / 29.0})                    # min-max scaled column
    df = pd.DataFrame(rows)
    d = f'{BASE}/Projects/{folder}/scoring_data'
    os.makedirs(d)
    df.to_csv(f'{d}/rolling_3m_score_monthly_scoring_file.csv.gz', index=False, compression='gzip')
    lines_by_cat[cat] = df

# ── Mock importance file ──────────────────────────────────────────────────────────────────────
imp = []
def add(pid, cat, feat, score, tier='Tier 1'):
    imp.append({'Servicing_Provider_ID': pid, 'Feature': feat, 'Impact_Score': score,
                'Feature_Detailed': 'Wait time z-score vs global mean (Rare Anomaly = z 10-20)',
                'Feature_Category': 'Transportation Patterns', 'Importance_Tier': tier,
                'Scoring_Date': '2026-10-01', 'Category': cat})
NEMT, DME = 'NEMT NOT IHS', 'DME MEDICAL SUPPLIES'
# P_GOOD (1029, NEMT): unsorted input, tier, insanely rare label, yes/no, provider-level numeric, dedup
for f, s in [('is_christmas_day', 0.30), ('avg_trips_per_patient', 0.70), ('index', 0.99),
             ('global_procedure_zscore_class_Rare Anomaly', 0.90), ('wait_time_zscore_class_Insanely Rare', 0.80),
             ('is_major_holiday_service', 0.20), ('PROC_CD_T2007', 0.40), ('PROC_CD_A0130', 0.35),
             ('weekly_provider_visit_count', 0.95), ('CLAIM_ON_CASE_DETAILS', 0.97), ('GENDER_F', 0.25)]:
    add(1029, NEMT, f, s)
# P_BAD (1028, NEMT): planted defects in the model output
for f, s in [('global_procedure_zscore_class_Rare Anomaly', 0.8), ('avg_trips_per_patient', 0.6), ('PROC_CD_T2007', 0.3)]:
    add(1028, NEMT, f, s)
# P_TRUNC (1027, NEMT): first answer cut off, retry succeeds
for f, s in [('avg_trips_per_patient', 0.6), ('PROC_CD_A0130', 0.5)]:
    add(1027, NEMT, f, s)
# P_FAIL (1026, NEMT): never reaches the final channel
for f, s in [('avg_trips_per_patient', 0.6)]:
    add(1026, NEMT, f, s)
# P_CONTEXT (1025, NEMT): only context-only features -> evidence_only
for f, s in [('GENDER_F', 0.6), ('diag_1_level_I_Endocrine diseases', 0.5)]:
    add(1025, NEMT, f, s)
# P_HELD (1024, NEMT): only held / not narrated -> evidence_only with no features
for f, s in [('weekly_provider_visit_count', 0.9), ('CLAIM_ON_CASE_DETAILS', 0.8)]:
    add(1024, NEMT, f, s)
# P_DME (1229, DME): applies-to drop (NEMT trips feature), released pair hold, code decoding
for f, s in [('avg_trips_per_patient', 0.9), ('diag_cd_1_proc_cd_pair_Z89512__L5620', 0.8),
             ('PROC_CD_E1390', 0.5), ('global_procedure_zscore_class_Anomaly', 0.4),
             ('diag_cd_1_proc_cd_pair_F4312__90837', 0.45)]:     # full code not in table: group fallback
    add(1229, DME, f, s)
if MODE == 'unmapped':
    add(1029, NEMT, 'patient_visit_gap_days_p90', 0.5)      # numeric root + unknown suffix: must stop the run
if MODE == 'adversarial':
    # 1023: shown drivers unlinked; linked drivers only below the cut (TOP_NUM forced to 5 below)
    for f, s in [('billing_growth_rate', 0.9), ('new_proc_code_count', 0.8), ('pct_round_trips', 0.7),
                 ('pct_claims_any_anomaly', 0.6), ('pct_trips_pharmacy_only', 0.5),
                 ('avg_claims_per_active_day', 0.02), ('avg_trips_per_active_day', 0.02), ('pct_trips_under_1mi', 0.02)]:
        add(1023, NEMT, f, s)
    # 1022: '_0' yes/no level (provider has the fewest holiday lines) plus one real driver
    add(1022, NEMT, 'is_major_holiday_service_0', 0.9); add(1022, NEMT, 'avg_trips_per_patient', 0.5)
    # 1021: tier 'Normal' scored above 'Rare Anomaly'; scaled dollars column; duplicate and float-ID rows
    add(1021, NEMT, 'global_procedure_zscore_class_Normal', 0.95); add(1021, NEMT, 'global_procedure_zscore_class_Rare Anomaly', 0.90)
    add(1021, NEMT, 'avg_pmt_per_patient', 0.6); add(1021, NEMT, 'avg_pmt_per_patient', 0.55)
    add('1021.0', NEMT, 'avg_trips_per_patient', 0.4)
    add('', NEMT, 'avg_trips_per_patient', 0.4)                          # blank provider ID
pd.DataFrame(imp).to_csv(f'{DATA}/all_feature_importance_normalized_and_tiers.csv', index=False)

# ── Reference tables ──────────────────────────────────────────────────────────────────────────
pd.DataFrame({'ICD10 Code': ['Z89512'], 'Description': ['Acquired absence of right leg below knee']}).to_csv(
    f'{BASE}/data/external data/icd10_categories_description.csv', index=False)
pd.DataFrame({'diag_cd_root': ['Z89', 'F43'],
              'category_level_III': ['acquired absence of limb', 'reaction to severe stress, and adjustment disorders'],
              'category_level_II': ['persons with potential health hazards', 'anxiety, dissociative, stress-related disorders'],
              'category_level_I': ['factors influencing health status', 'mental, behavioral and neurodevelopmental disorders']}
             ).to_csv(f'{BASE}/data/external data/icd_10_diagnosis_code_root_and_category.csv', index=False)
pd.DataFrame({'HCPCS Codes': ['L5620', 'E1390', 'T2007', 'A0130'],
              'Description': ['Addition to lower extremity prosthesis', 'Oxygen concentrator',
                              'Transportation waiting time, one-half hour increments', 'Non-emergency transportation: wheelchair van']}).to_csv(
    f'{BASE}/data/external data/hcpcs_code_description.csv', index=False)
pd.DataFrame({'CPT Code': ['99213'], 'Description': ['Office visit']}).to_csv(
    f'{BASE}/data/external data/cpt_codes_description.csv', index=False)
pd.DataFrame({'Modifier': ['RR'], 'Description': ['Rental']}).to_csv(
    f'{BASE}/data/external data/procedure_modifiers.csv', index=False)

# ── Stub modules that are not installed here ──────────────────────────────────────────────────
class FakeTensor:
    def __init__(self, rows): self.rows = rows
    def to(self, device): return self
    @property
    def shape(self): return (len(self.rows), len(self.rows[0]))
    def __getitem__(self, i): return FakeRow(self.rows[i])
class FakeRow(list):
    def __getitem__(self, s):
        r = list.__getitem__(self, s)
        return FakeRow(r) if isinstance(s, slice) else r
    def tolist(self): return list(self)

def stub(name, **attrs):
    m = types.ModuleType(name); m.__dict__.update(attrs); sys.modules[name] = m; return m
stub('torch', no_grad=contextlib.nullcontext, Tensor=FakeTensor)
stub('transformers', AutoTokenizer=object, AutoModelForCausalLM=object, pipeline=None, BitsAndBytesConfig=object)
stub('huggingface_hub', snapshot_download=None, login=lambda *a, **k: None)
stub('pyspark'); stub('pyspark.sql', SparkSession=object)
stub('pyspark.sql.types', StructType=object, StructField=object, StringType=object, DoubleType=object, DateType=object)
stub('pyspark.sql.functions', col=None)

# ── Fake GPT-OSS ──────────────────────────────────────────────────────────────────────────────
from openai_harmony import Message, Role
STATE = {'messages': None, 'calls': {}, 'reasoning_effort': set(), 'registry': {}, 'prompts': {}}

def good_note(pid, cat, user):
    label = re.search(r'MAIN PATTERNS[^\n]*\n\s+1\. ([^:\[\n]+)', user).group(1).strip()
    tp = re.search(r'Total paid: (\$[\d,.]+) for this provider; peer median (\$[\d,.]+); (\d+(?:st|nd|rd|th)) percentile', user)
    s = f"Provider {pid} was flagged because its {cat} paid claims stood out on one main measure: {label.lower()}. "
    if tp:
        s += (f"The provider was paid {tp.group(1)} in total, compared with a peer median of {tp.group(2)}, "
              f"which places it at the {tp.group(3)} percentile among peers in the same claim category. ")
    s += ("The pattern was unusual compared with peers, and the measures behind it are listed on the evidence card. "
          "Other explanations, such as a larger patient base or longer appointments, could account for part of the difference.")
    return s

def respond(pid, cat, user, attempt):
    if pid == '1028':
        return {'final': (f"Provider {pid} was flagged because its NEMT NOT IHS billed claims show a markedly high "
                          "score of 0.860, and paid wait time about 4 times the peer level, consistent with upcoding. "
                          "Services were not delivered on 12,345 trips, which appears deliberate. The provider should be "
                          "investigated further for these reasons."), 'n': 120}
    if pid == '1027' and attempt == 1:
        return {'final': None, 'n': 'limit'}
    if pid == '1026':
        return {'final': None, 'n': 'limit', 'raise': True}
    return {'final': good_note(pid, cat, user), 'n': 150}

class FakeTokenizer:
    def apply_chat_template(self, messages, return_tensors=None, add_generation_prompt=True, return_dict=True, **kw):
        STATE['messages'] = messages
        STATE['reasoning_effort'].add(kw.get('reasoning_effort'))
        return {'input_ids': FakeTensor([[1] * 10]), 'attention_mask': FakeTensor([[1] * 10])}
    def decode(self, ids, skip_special_tokens=False):
        spec = STATE['registry'][ids[0]]
        out = '<|channel|>analysis<|message|>Thinking about the provider<|end|>'
        if spec['final']:
            out += '<|start|>assistant<|channel|>final<|message|>' + spec['final'] + '<|return|>'
        return out

class FakeModel:
    device = 'cpu'
    def generate(self, input_ids, attention_mask=None, max_new_tokens=800, do_sample=False):
        user = STATE['messages'][1]['content']
        pid, cat = re.search(r'PROVIDER: (\S+) \| CLAIM CATEGORY: ([^\n]+)', user).groups()
        STATE['calls'][pid] = STATE['calls'].get(pid, 0) + 1
        STATE['prompts'][pid] = STATE['messages']
        spec = respond(pid, cat.strip(), user, STATE['calls'][pid])
        n = max_new_tokens if spec['n'] == 'limit' else spec['n']
        tid = 10_000 + len(STATE['registry'])
        STATE['registry'][tid] = spec
        return FakeTensor([[1] * 10 + [tid] * n])

class FakeHarmony:
    def parse_messages_from_completion_tokens(self, tokens, role=None, strict=True):
        assert role == Role.ASSISTANT, f'role should be Role.ASSISTANT, got {role!r}'
        spec = STATE['registry'][tokens[0]]
        if spec.get('raise'):
            raise RuntimeError('incomplete message')
        msgs = [Message.from_role_and_content(Role.ASSISTANT, 'Thinking about the provider').with_channel('analysis')]
        if spec['final']:
            msgs.append(Message.from_role_and_content(Role.ASSISTANT, spec['final']).with_channel('final'))
        return msgs

# ── Run the notebook cells ────────────────────────────────────────────────────────────────────
nb = json.load(open(NB))
ns = {'__name__': '__main__', 'dbutils': None}
for i, c in enumerate(nb['cells']):
    if c['cell_type'] != 'code':
        continue
    src = ''.join(c['source']).replace('/Workspace/Users/', f'{T}/Workspace/Users/')
    if src.lstrip().startswith('%pip'):
        continue
    if 'AutoModelForCausalLM.from_pretrained' in src:
        ns.update(tokenizer=FakeTokenizer(), model=FakeModel(), harmony_enc=FakeHarmony())
        continue
    try:
        exec(compile(src, f'cell_{i}', 'exec'), ns)
        if MODE == 'adversarial' and 'TOP_NUM = 20' in src:
            ns['TOP_NUM'] = 5
    except Exception as e:
        if MODE == 'unmapped' and isinstance(e, ValueError) and 'no dictionary row' in str(e):
            print('EXPECTED STOP at cell', i, ':', str(e).splitlines()[0])
            print(str(e).splitlines()[1])
            sys.exit(0)
        raise
if MODE == 'unmapped':
    sys.exit('FAIL: run did not stop on an unmapped feature')

json.dump({'reasoning_effort': sorted(map(str, STATE['reasoning_effort'])), 'calls': STATE['calls'],
           'prompts': STATE['prompts']}, open('./fake_llm_state.json', 'w'), indent=1)
out = ns['output_df']; aud = ns['audit_df']; feats = ns['features_narrated']
out.to_csv('./output_df.csv', index=False)
aud.to_json('./audit.jsonl', orient='records', lines=True)
feats.to_csv('./features_narrated.csv', index=False)
if MODE == 'adversarial':
    o = out.set_index(out[ns['provider_id']].astype(str))
    f = feats.assign(pid=feats[ns['provider_id']].astype(str))
    checks = {
        'IDs are clean strings, blank dropped': set(o.index) >= {'1021', '1022', '1023'} and not any(
            i.endswith('.0') or i in ('', 'nan') for i in o.index),
        '1023: no scheme from drivers outside the prompt': 'phantom' not in o.at['1023', 'Detected_Schemes'],
        '1022: _0 holiday level moved to background': f[(f.pid == '1022') & (f.Feature == 'is_major_holiday_service_0')
                                                        ]['Narrate_Role'].tolist() == ['Context only'],
        '1022: holiday value describes holiday lines (low percentile)': all(
            int(re.search(r'(\d+)(st|nd|rd|th) percentile', v).group(1)) < 50
            for v in f[(f.pid == '1022') & (f.Feature == 'is_major_holiday_service_0')]['Value_Sentence']),
        "1021: 'Rare Anomaly' stays a main pattern": o.at['1021', 'Narrative_Features'].split(' | ')[0].endswith('far above peers (highest level)'),
        "1021: 'Normal' level is background": f[(f.pid == '1021') & (f.Feature == 'global_procedure_zscore_class_Normal')
                                                ]['Narrate_Role'].tolist() == ['Context only'],
        '1021: scaled dollars column gets no value sentence': f[(f.pid == '1021') & (f.Feature == 'avg_pmt_per_patient')
                                                               ]['Value_Sentence'].fillna('').eq('').all(),
        '1021: duplicate and float-ID rows collapsed': len(f[(f.pid == '1021') & (f.Feature == 'avg_pmt_per_patient')]) == 1,
        'scaled column detected': ns['value_coverage'].get('out_of_range_looks_scaled', 0) >= 1,
    }
    for k, v in checks.items():
        print(('PASS ' if v else 'FAIL ') + k)
    print('1023 schemes:', o.at['1023', 'Detected_Schemes'])
    print('1022 holiday value:', f[(f.pid == '1022') & (f.Feature == 'is_major_holiday_service_0')]['Value_Sentence'].tolist())
print('TEMP', T)
