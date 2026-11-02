"""Reported weight loss, star ratings, and review expressions."""
from __future__ import annotations
import argparse
import copy
import hashlib
import html
import importlib.metadata
import json
import math
import os
import re
import sys
import unicodedata
from datetime import datetime
from pathlib import Path
from typing import Any

for name in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS'):
    os.environ.setdefault(name, '2')
import numpy as np
import pandas as pd
from scipy import linalg, stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent



CATEGORIES = ['medical', 'supplement', 'food', 'drink']


EXPECTED = dict(medical=14804, supplement=100127, food=8042, drink=43084)


ALIASES = {
    'product_id': ['商品ID', '製品ID', 'product_id', 'productid', 'item_id'],
    'review_id': ['口コミID', 'レビューID', 'review_id', 'reviewid'],
    'text': ['口コミ本文', '口コミ内容', 'レビュー本文', 'review_text', 'text', 'body'],
    'rating': ['評価', '星評価', '評価点', '星', 'rating', 'score', 'stars'],
    'date': ['投稿日時', '投稿日', '投稿年月日', 'review_date', 'date', 'created_at'],
    'parent_category': ['親カテゴリ名', '親カテゴリー名', '親カテゴリ', 'parent_category'],
    'child_category': ['子カテゴリ名', '子カテゴリー名', '子カテゴリ', 'child_category'],
}


DEFAULT_CONFIG = {'columns': {'review_product_id': None,
             'product_product_id': None,
             'review_id': None,
             'text': None,
             'rating': None,
             'date': None,
             'parent_category': None,
             'child_category': None},
 'categories': {'medical': {'field': 'parent_category',
                            'values': ['医療ダイエット', '医療ダイエット（メディカルダイエット）']},
                'supplement': {'field': 'child_category', 'values': ['ダイエットサプリ']},
                'food': {'field': 'parent_category', 'values': ['ダイエット食品']},
                'drink': {'field': 'parent_category', 'values': ['ダイエットドリンク', 'ダイエット飲料']}},
 'missing_rating_labels': ['',
                           '評価なし',
                           '評価無し',
                           '未評価',
                           '無評価',
                           '評価未入力',
                           'なし',
                           '未回答',
                           '不明',
                           'nan',
                           'none',
                           'null',
                           'na',
                           'n/a',
                           '-',
                           '―',
                           'ー'],
 'allow_zero_as_missing': False,
 'strict_join': False,
 'drop_exact_duplicate_rows': True,
 'assume_bare_weight_is_kg': True,
 'min_weight_kg': 30.0,
 'max_weight_kg': 250.0,
 'max_abs_weight_change_kg': 50.0,
 'discordance_min_category_n': 30,
 'discordance_font_file': None}


PATTERNS = {
    'effect_positive_phrase': r'効果(?:が|は)?(?:あった|ありました|あり(?!ま?せん))|痩せた(?!い)|やせた(?!い)|体重(?:が|は)?減(?:った|りました)|減量できた|効いた',
    'effect_negative_phrase': r'効果(?:が|は|も)?(?:ない|なかった|なし|ありません|感じられな)|痩せ(?:ない|なかった|ません)|やせ(?:ない|なかった|ません)|体重(?:が|は)?減ら(?:ない|なかった)|変化(?:が|は)?(?:ない|なし|なかった)|効か(?:ない|なかった)',
    'symptom_mention': r'副作用|下痢|腹痛|吐き気|頭痛|便秘|体調不良',
    'cancellation_mention': r'解約|キャンセル',
    'price_mention': r'価格|値段|料金|コスパ|費用|金額|割高|割安',
    'effect_mention': r'効果|痩せ|やせ|体重|減量',
    'subscription_mention': r'定期購入|定期契約|定期コース|定期便',
    'cancellation_difficulty_phrase': r'解約(?:が|は)?(?:できな|出来な|難し|困難)|解約し(?:にく|づら)|キャンセル(?:が|は)?(?:できな|出来な)',
}


def save_json(path: Path, value: Any) -> None:
    def clean(v):
        if isinstance(v, dict): return {str(k):clean(x) for k,x in v.items()}
        if isinstance(v, (list,tuple)): return [clean(x) for x in v]
        if isinstance(v, np.ndarray): return clean(v.tolist())
        if isinstance(v, np.generic): return clean(v.item())
        if isinstance(v, float) and not math.isfinite(v): return None
        if isinstance(v, Path): return str(v)
        return v
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(clean(value),ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')


def file_sha256(path: Path) -> str:
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''): h.update(block)
    return h.hexdigest()


def normalize_key(value: Any) -> str:
    return re.sub(r'[\s_\-]+', '', unicodedata.normalize('NFKC', str(value)).strip().lower())


def save_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding='utf-8-sig')


def resolve_column(df: pd.DataFrame, logical: str, explicit: str | None = None, required: bool = True) -> Any:
    if explicit is not None:
        if explicit not in df.columns:
            raise ValueError(f'設定した列 {explicit!r} がありません。列名: {list(df.columns)}')
        return explicit
    allowed = {normalize_key(x) for x in ALIASES[logical]}
    found = [x for x in df.columns if normalize_key(x) in allowed]
    if len(found) == 1: return found[0]
    if len(found) > 1:
        raise ValueError(f'{logical} の候補が複数あります: {found}。config["columns"] で1列を指定してください。')
    if required:
        raise ValueError(f'{logical} の列を特定できません。config["columns"] を指定してください。列名: {list(df.columns)}')
    return None


def id_series(series: pd.Series) -> pd.Series:

    s = series.astype('string').fillna('').map(lambda v: unicodedata.normalize('NFKC', v).strip())
    return s.str.replace(r'^([+-]?\d+)\.0+$', r'\1', regex=True).astype('string')


def normalized_text(value: Any) -> str:
    if pd.isna(value): return ''
    s = html.unescape(str(value)).replace('\r\n', '\n').replace('\r', '\n')
    s = re.sub(r'<br\s*/?>', '\n', s, flags=re.I)
    s = re.sub(r'<[^>]+>', ' ', s)
    return unicodedata.normalize('NFKC', s).strip()


def parse_rating(value: Any, missing_labels: list[str] | None = None,
                 allow_zero: bool = False) -> tuple[float, str]:
    if pd.isna(value): return np.nan, 'missing'
    s=unicodedata.normalize('NFKC',str(value)).strip().replace('\ufe0f','')
    labels=DEFAULT_CONFIG['missing_rating_labels'] if missing_labels is None else missing_labels
    if s.casefold() in {unicodedata.normalize('NFKC',str(x)).strip().casefold() for x in labels}:
        return np.nan, 'missing'
    compact=re.sub(r'\s+','',s)
    if re.fullmatch(r'★{1,5}☆{0,4}',compact) and len(compact)<=5:
        return float(compact.count('★')),'valid'
    m=re.fullmatch(r'(?:評価|星|★)?([0-9]+(?:\.0+)?)(?:点|つ|/5)?',compact)
    if m:
        value=float(m.group(1))
        if value in {1.,2.,3.,4.,5.}: return value,'valid'
        if value==0 and allow_zero: return np.nan,'missing'
    return np.nan,'invalid'


def parse_measure(value: Any, kind: str, assume_bare: bool=True) -> tuple[float,str]:

    if pd.isna(value): return np.nan,'missing'
    s=unicodedata.normalize('NFKC',str(value)).strip().casefold()
    if s in set(DEFAULT_CONFIG['missing_rating_labels'])|{'回答なし','--','---','不明'}:
        return np.nan,'missing'
    number=r'[+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?'
    unit_patterns={
        'weight':r'kg|kgs|kilograms?|キログラム|キロ|g|grams?|グラム|lb|lbs|ポンド',
        'height':r'cm|センチメートル|センチ|m|メートル',
        'age':r'歳|才|years?|才',
    }
    m=re.fullmatch(rf'({number})\s*({unit_patterns[kind]})?',s)
    if not m: return np.nan,'unparsed'
    v=float(m.group(1).replace(',','')); unit=m.group(2)
    if not np.isfinite(v) or v<=0: return np.nan,'nonpositive_or_nonfinite'
    if not unit and not assume_bare: return np.nan,'unit_not_confirmed'
    status='valid_assumed_unit' if not unit else 'valid_explicit_unit'
    if kind=='weight' and unit in {'g','gram','grams','グラム'}:
        v/=1000; status='converted_g_to_kg'
    if kind=='weight' and unit in {'lb','lbs','ポンド'}:
        v*=0.45359237; status='converted_lb_to_kg'
    if kind=='height' and unit in {'m','メートル'}:
        v*=100; status='converted_m_to_cm'
    return v,status


def attach_numeric_fields(rr: pd.DataFrame, raw: pd.DataFrame, cfg: dict, out: Path) -> None:
    audits=[]; formats=[]
    for source,target,kind in [('使用前体重','weight_before_kg','weight'),
                               ('使用後体重','weight_after_kg','weight'),
                               ('身長','height_cm','height'),('年齢','age_years','age')]:
        values=raw[source] if source in raw else pd.Series('',index=raw.index)
        unique=values.fillna('').astype(str).drop_duplicates()
        parsed={s:parse_measure(s,kind,cfg['assume_bare_weight_is_kg'] if kind=='weight' else True) for s in unique}
        mapped=values.fillna('').astype(str).map(parsed)
        rr[target]=[v[0] for v in mapped]
        rr[target+'_status']=[v[1] for v in mapped]
        for status,n in rr[target+'_status'].value_counts().items():
            audits.append({'column':source,'status':status,'n':int(n),'pct':100*n/len(rr)})
        invalid=rr[target].isna()&rr[target+'_status'].ne('missing')
        forms=values.loc[invalid].map(lambda v:re.sub(r'\d+(?:[.,]\d+)*','<num>',unicodedata.normalize('NFKC',str(v)))[:100])
        for fmt,n in forms.value_counts().items(): formats.append({'column':source,'format':fmt,'n':int(n)})
    rr['weight_loss_kg']=rr.weight_before_kg-rr.weight_after_kg
    rr['weight_pair_parsed']=rr[['weight_before_kg','weight_after_kg']].notna().all(axis=1)
    rr['weight_in_range']=rr.weight_before_kg.between(cfg['min_weight_kg'],cfg['max_weight_kg']) & rr.weight_after_kg.between(cfg['min_weight_kg'],cfg['max_weight_kg'])
    rr['weight_change_in_range']=rr.weight_loss_kg.abs().le(cfg['max_abs_weight_change_kg'])
    rr['weight_usable']=rr.weight_pair_parsed&rr.weight_in_range&rr.weight_change_in_range
    rr['loss_5kg']=rr.weight_loss_kg/5
    rr['loss_5kg_squared']=rr.loss_5kg**2
    rr['baseline_weight_10kg']=(rr.weight_before_kg-60)/10
    rr['weight_loss_percent']=100*rr.weight_loss_kg/rr.weight_before_kg
    rr['baseline_bmi']=rr.weight_before_kg/(rr.height_cm/100)**2

    rr['age_10years']=(rr.age_years-40)/10
    rr['height_10cm']=(rr.height_cm-160)/10
    rr['age_height_usable']=rr.age_years.between(18,100)&rr.height_cm.between(120,220)
    for source,target in [('使用期間','duration_group'),('性別','sex_group')]:
        s=raw[source].fillna('').astype(str).map(lambda x:unicodedata.normalize('NFKC',x).strip()) if source in raw else pd.Series('',index=raw.index)
        missing=s.str.casefold().isin(set(DEFAULT_CONFIG['missing_rating_labels'])|{'不明','回答なし'})
        s=s.mask(missing,'unknown')

        rr[target]=s.where(s.isin(s.value_counts().head(20).index),'other_rare_label')
        rr[target+'_known']=~missing
    save_csv(pd.DataFrame(audits),out/'data_checks'/'numeric_conversion_counts.csv')
    save_csv(pd.DataFrame(formats,columns=['column','format','n']),out/'data_checks'/'PRIVATE_unparsed_numeric_formats.csv')


def prepare_data(review: pd.DataFrame, product: pd.DataFrame, cfg: dict, out: Path) -> tuple[pd.DataFrame, dict]:
    if not isinstance(review, pd.DataFrame) or not isinstance(product, pd.DataFrame):
        raise TypeError('review と product は pandas.DataFrame にしてください。')
    if review.empty or product.empty: raise ValueError('空の入力データです。')
    if not review.columns.is_unique or not product.columns.is_unique:
        raise ValueError('重複した列名があります。列名を一意にしてください。')
    audit = {'input_review_rows': len(review), 'input_product_rows': len(product), 'warnings': [], 'schema': {}}
    r, p = review.copy(), product.copy()
    r.columns = r.columns.str.lstrip('\ufeff').str.strip()
    p.columns = p.columns.str.lstrip('\ufeff').str.strip()
    if not r.columns.is_unique or not p.columns.is_unique:
        raise ValueError('空白を除去すると列名が重複します。')
    for name, df in [('review', r), ('product', p)]:
        n = int(df.duplicated().sum())
        audit[name + '_exact_duplicate_rows'] = n
        if n and cfg['drop_exact_duplicate_rows']: df.drop_duplicates(inplace=True)
    r, p = r.reset_index(drop=True), p.reset_index(drop=True)
    c = cfg['columns']
    names = {
        'review_product_id': resolve_column(r, 'product_id', c['review_product_id']),
        'product_product_id': resolve_column(p, 'product_id', c['product_product_id']),
        'rating': resolve_column(r, 'rating', c['rating']),
        'text': resolve_column(r, 'text', c['text']),
        'review_id': resolve_column(r, 'review_id', c['review_id'], False),
        'date': resolve_column(r, 'date', c['date'], False),
        'parent_category': resolve_column(p, 'parent_category', c['parent_category'], False),
        'child_category': resolve_column(p, 'child_category', c['child_category'], False),
    }
    audit['schema'] = {k: str(v) if v is not None else None for k, v in names.items()}
    save_json(out / 'data_checks' / 'input_columns.json', audit['schema'])
    if names['parent_category'] is None and names['child_category'] is None:
        raise ValueError('product の親/子カテゴリ列が必要です。config["columns"] で指定してください。')
    for cat, rule in cfg['categories'].items():
        if names.get(rule['field']) is None:
            audit['warnings'].append(f'{cat}: {rule["field"]} がなく、このカテゴリは照合できません。')
    pids = id_series(p[names['product_product_id']])
    if pids.eq('').any(): raise ValueError('product の商品IDが空です。欠損キーを解消してください。')
    if pids.duplicated().any():
        raise ValueError('product に同一商品IDの複数行が残っています。カテゴリ等の不一致を手元で確認してください。結合は中止しました。')
    ids = id_series(r[names['review_id']]) if names['review_id'] else pd.Series('', index=r.index)
    if names['review_id'] and ids[ids.ne('')].duplicated().any():
        raise ValueError('同じ口コミIDの異なる行があります。重複か更新履歴かを確認するまで集計を中止します。')
    audit['missing_review_ids'] = int(ids.eq('').sum())
    if names['review_id'] is None:
        audit['warnings'].append('口コミIDがないため、行を単位に分析します。人物数は分かりません。')
    unique_ratings = r[names['rating']].drop_duplicates().tolist()

    rating_map = {str(v): parse_rating(v, cfg['missing_rating_labels'], cfg['allow_zero_as_missing']) for v in unique_ratings}
    parsed = r[names['rating']].map(lambda x: (np.nan, 'missing') if pd.isna(x) else rating_map[str(x)])
    rr = pd.DataFrame({
        'row_key': np.arange(len(r)), 'product_id': id_series(r[names['review_product_id']]),
        'rating': [x[0] for x in parsed], 'rating_status': [x[1] for x in parsed],
        'text': r[names['text']].fillna('').astype(str),
    })
    rr['text_clean'] = rr['text'].map(normalized_text)
    rr['text_chars'] = rr['text_clean'].str.len().astype('int64')
    rr['log_chars'] = np.log1p(rr['text_chars'])
    if names['date'] is not None:

        year = r[names['date']].astype('string').str.normalize('NFKC').str.extract(r'^\s*((?:19|20)\d{2})(?:[年/\-.]|$)', expand=False)
        rr['year'] = year.fillna('unknown')
        audit['date_year_unparsed_or_missing'] = int(year.isna().sum())
    else:
        rr['year'] = 'unknown'
        audit['warnings'].append('投稿日時列がないため、年固定効果は入りません。')
    attach_numeric_fields(rr, r, cfg, out)
    pp = pd.DataFrame({'product_id': pids})
    for name in ['parent_category', 'child_category']:
        pp[name] = p[names[name]].astype('string').fillna('') if names[name] is not None else ''
    joined = rr.merge(pp, how='left', on='product_id', validate='many_to_one', indicator=True, sort=False)
    assert len(joined) == len(rr), '結合で行数が変わりました。'
    audit['review_rows_after_deduplication'] = len(rr)
    audit['product_rows_after_deduplication'] = len(pp)
    audit['unmatched_review_rows'] = int(joined['_merge'].eq('left_only').sum())
    audit['matched_review_rows'] = len(joined) - audit['unmatched_review_rows']
    save_json(out / 'data_checks' / 'input_data_summary.json', audit)
    if audit['unmatched_review_rows'] and cfg['strict_join']:
        raise ValueError(f'{audit["unmatched_review_rows"]} 件が商品IDで結合できません。data_checks/input_data_summary.json を確認してください。')
    joined['category'] = 'other'
    masks = {}
    for cat, rule in cfg['categories'].items():
        masks[cat] = joined[rule['field']].fillna('').map(normalize_key).isin({normalize_key(v) for v in rule['values']}) & joined['_merge'].eq('both')
    if pd.DataFrame(masks).sum(axis=1).gt(1).any():
        raise ValueError('同じ口コミが複数カテゴリに一致しています。カテゴリ定義を確認してください。')
    for cat, mask in masks.items(): joined.loc[mask, 'category'] = cat
    joined['high_rating'] = np.where(joined['rating'].notna(), (joined['rating'] >= 4).astype(float), np.nan)
    audit['rating_status_all_rows'] = joined['rating_status'].value_counts().to_dict()
    audit['blank_text_all_rows'] = int(joined['text_chars'].eq(0).sum())
    audit['target_category_rows'] = int(joined['category'].isin(CATEGORIES).sum())
    if not audit['target_category_rows']: raise ValueError('対象カテゴリが0件です。親/子カテゴリの列とカテゴリ値を確認してください。')

    rating_audit = r[names['rating']].fillna('<MISSING>').astype(str).value_counts(dropna=False).rename_axis('raw_rating_label').reset_index(name='n')
    rating_audit['parsed_rating'] = rating_audit['raw_rating_label'].map(lambda s: parse_rating(s if s != '<MISSING>' else None, cfg['missing_rating_labels'], cfg['allow_zero_as_missing'])[0])
    rating_audit['parse_status'] = rating_audit['raw_rating_label'].map(lambda s: parse_rating(s if s != '<MISSING>' else None, cfg['missing_rating_labels'], cfg['allow_zero_as_missing'])[1])
    save_csv(rating_audit, out / 'data_checks' / 'PRIVATE_rating_conversion.csv')
    save_csv(joined.groupby(['parent_category', 'child_category', 'category'], dropna=False).size().reset_index(name='n'), out / 'data_checks' / 'product_category_mapping.csv')
    sample = joined['category'].isin(CATEGORIES)
    flow = [('input', len(review)), ('after_exact_duplicate_handling', len(rr)), ('matched_products', audit['matched_review_rows']), ('four_target_categories', int(sample.sum())), ('target_nonblank_text', int((sample & joined.text_chars.gt(0)).sum())), ('target_rated_nonblank_text', int((sample & joined.text_chars.gt(0) & joined.rating.notna()).sum()))]
    save_csv(pd.DataFrame(flow, columns=['stage', 'n']), out / 'data_checks' / 'sample_selection_counts.csv')
    if audit['unmatched_review_rows']:
        audit['warnings'].append('未結合の口コミは原データ監査には残し、カテゴリ別分析・回帰から除外。')
    observed = joined.groupby('category').size()
    save_csv(pd.DataFrame([{'category': k, 'observed_reviews': int(observed.get(k, 0)), 'paper_reviews': v, 'difference': int(observed.get(k, 0))-v} for k, v in EXPECTED.items()]), out/'data_checks'/'otsuka_sample_count_comparison.csv')
    save_json(out / 'data_checks' / 'input_data_summary.json', audit)
    return joined, audit


def independent_columns(x: pd.DataFrame, allow_constant: bool=False) -> tuple[pd.DataFrame,list[str]]:

    if x.empty: return x, []
    a=x.to_numpy(float)
    if not np.isfinite(a).all(): raise ValueError('説明変数に欠損/無限値があります。')
    norms=np.linalg.norm(a,axis=0)
    eligible=norms>1e-9
    if not allow_constant: eligible &= np.ptp(a,axis=0)>1e-10
    idx=np.flatnonzero(eligible)
    if not len(idx): return x.iloc[:,:0],list(x.columns)
    _,r,piv=linalg.qr(a[:,idx]/norms[idx],mode='economic',pivoting=True)
    rank=int((np.abs(np.diag(r))>1e-9).sum())
    keep=sorted(idx[piv[:rank]])
    return x.iloc[:,keep],[c for i,c in enumerate(x.columns) if i not in keep]


def regression_table(beta, covariance, terms, df_t=None) -> pd.DataFrame:
    beta=np.asarray(beta); diag=np.diag(covariance)
    if np.any(diag < -1e-9) or not np.isfinite(covariance).all():
        raise ValueError('共分散推定に異常があります。')
    se=np.sqrt(np.maximum(diag,0))
    statistic=np.divide(beta,se,out=np.full_like(beta,np.nan),where=se>1e-15)
    critical=stats.t.ppf(.975,df_t) if df_t else stats.norm.ppf(.975)
    p=2*(stats.t.sf(np.abs(statistic),df_t) if df_t else stats.norm.sf(np.abs(statistic)))
    return pd.DataFrame({'term':list(terms),'coefficient':beta,'cluster_se':se,
                         'ci_low':beta-critical*se,'ci_high':beta+critical*se,'p_value':p})


NEW_TEXT_LABELS = {
    'x_cancel_difficulty_expression': '解約困難の表現',
    'x_symptom_experience_expression': '症状発生・不調の表現',
    'x_symptom_negated_expression': '症状なしの表現',
    'x_price_negative_expression': '価格不満の表現',
    'x_price_positive_expression': '価格肯定の表現',
    'x_effect_positive_expression': '減量・効果肯定の表現',
    'x_effect_negative_expression': '減量・効果否定の表現',
    'x_usability_positive_expression': '使用感肯定の表現',
    'x_taste_positive_expression': '味の肯定表現',
    'x_service_positive_expression': '接客・対応肯定の表現',
    'x_continuation_positive_expression': '継続しやすい表現',
    'x_symptom_improvement_expression': '症状改善の表現',
    'x_usability_mention': '使用感への言及',
    'x_taste_mention': '味への言及',
    'x_service_mention': '接客・対応への言及',
    'x_continuation_mention': '継続への言及',
}


NEW_TEXT_ENGLISH_LABELS = {
    'x_cancel_difficulty_expression': 'Cancellation difficulty wording',
    'x_symptom_experience_expression': 'Symptom occurrence wording',
    'x_symptom_negated_expression': 'No-symptom wording',
    'x_price_negative_expression': 'Negative price wording',
    'x_price_positive_expression': 'Positive price wording',
    'x_effect_positive_expression': 'Positive effect wording',
    'x_effect_negative_expression': 'Negative effect wording',
    'x_usability_positive_expression': 'Positive usability wording',
    'x_taste_positive_expression': 'Positive taste wording',
    'x_service_positive_expression': 'Positive service wording',
    'x_continuation_positive_expression': 'Easy continuation wording',
    'x_symptom_improvement_expression': 'Symptom improvement wording',
    'x_usability_mention': 'Usability mention',
    'x_taste_mention': 'Taste mention',
    'x_service_mention': 'Service mention',
    'x_continuation_mention': 'Continuation mention',
}


NEW_TEXT_FEATURES = list(NEW_TEXT_LABELS)


_X_SYMPTOM = r'(?:副作用|下痢|腹痛|吐き気|頭痛|便秘|体調不良|胃痛|動悸|めまい|眩暈|発疹|じんましん|蕁麻疹)'


_X_PRICE = r'(?:価格|値段|料金|費用|金額|コスト)'


_X_NOCLAUSE = r'[^。！？!?;；\n、,]'


NEW_TEXT_PATTERNS = {
    'x_cancel_difficulty_expression': (
        r'(?:解約|キャンセル)(?:すること|手続き|の手続き)?(?:が|は|を)?(?:なかなか|全く|どうしても)?'
        r'(?:できな(?:い|かった)|出来な(?:い|かった)|できません(?:でした)?|出来ません(?:でした)?|'
        r'難しい|難しかった|困難(?:だ|で|だった)|面倒(?:だ|で|だった))'
        r'|(?:解約|キャンセル)し(?:にくい|にくかった|づらい|づらかった)'
        r'|(?:解約|キャンセル)(?:を)?(?:断られた|拒否された|受け付けてもらえなかった)'
        r'|(?:解約|キャンセル)(?:の)?(?:電話|窓口)(?:が|は)?(?:つながらない|繋がらない|つながらなかった|繋がらなかった)'
    ),
    'x_symptom_experience_expression': (
        _X_SYMPTOM + r'(?:が|は|も|を)?(?:少し|多少|強く|ひどく|酷く)?'
        r'(?:出た|出ました|あった|ありました|起きた|起きました|生じた|生じました|発生した|発生しました|'
        r'現れた|現れました|ひどい|酷い|つらい|辛い|する|します|した|しました)'
        r'|(?:下痢|便秘)(?:に)?(?:なった|なりました|なってしまった)'
        r'|体調(?:が|を)?(?:悪くなった|悪くなりました|崩した|崩しました)'
        r'|(?:お腹|胃|頭)(?:が|は)?(?:痛くなった|痛くなりました|痛い|痛かった)'
        r'|(?:気持ちが悪くなった|気持ち悪くなった|気分が悪くなった)'
    ),
    'x_symptom_negated_expression': (
        _X_SYMPTOM + r'(?:が|は|も|を)?(?:特に|全く|まったく|ほとんど|一切)?'
        r'(?:ない|なかった|なし|無し|無い|無かった|ありません(?:でした)?|'
        r'出なかった|出ません(?:でした)?|起きなかった|起きません(?:でした)?|'
        r'感じなかった|感じません(?:でした)?|ありませんでした|認められなかった)'
    ),
    'x_price_negative_expression': (
        _X_PRICE + r'(?:が|は|も)?(?:とても|かなり|少し|やや|ちょっと)?'
        r'(?:高い|高かった|高すぎる|高過ぎる|高すぎた|高過ぎた|高く感じた|高くて|不満(?:だ|で|だった))'
        r'|(?:コスパ|コストパフォーマンス)(?:が|は)?(?:悪い|悪かった|良くない|よくない)'
        r'|(?:割高(?:だ|で|だった)|高額(?:だ|で|だった))'
    ),
    'x_price_positive_expression': (
        _X_PRICE + r'(?:が|は|も)?(?:とても|かなり|比較的)?(?:安い|安かった|安くて|手頃(?:だ|で|だった)|手ごろ(?:だ|で|だった)|お手頃(?:だ|で|だった))'
        r'|(?:コスパ|コストパフォーマンス)(?:が|は)?(?:良い|いい|よい|良かった|よかった|最高)'
        r'|(?:割安(?:だ|で|だった)|お得(?:だ|で|だった))'
    ),
    'x_effect_positive_expression': (
        r'効果(?:が|は|も)?(?:確かに|少し|多少|とても)?(?:あった|ありました|あります|感じた|感じました|実感できた|実感しました)'
        r'|(?:痩せた(?![いく])|やせた(?![いく])|痩せました|やせました|減量できた|減量できました)'
        r'|体重(?:が|は)?(?:減った|減りました|落ちた|落ちました)'
    ),
    'x_effect_negative_expression': (
        r'効果(?:が|は|も|を)?(?:全く|まったく|あまり|ほとんど)?(?:ない|なかった|なし|無し|ありません(?:でした)?|感じなかった|感じません(?:でした)?|実感できなかった|実感できません(?:でした)?)'
        r'|(?:痩せ|やせ)(?:ない|なかった|ません(?:でした)?)'
        r'|体重(?:が|は)?(?:減らない|減らなかった|減りません(?:でした)?|変わらない|変わらなかった|変わりません(?:でした)?)'
    ),
    'x_usability_positive_expression': (
        r'(?:飲み|食べ|使い|溶け|溶かし)(?:やすい|やすかった|やすくて)'
        r'|(?:使い心地|使用感|飲み心地|飲みやすさ)(?:が|は|も)?(?:良い|いい|よい|良かった|よかった|快適)'
    ),
    'x_taste_positive_expression': (
        r'(?:美味しい|おいしい|美味しかった|おいしかった|美味しくて|おいしくて)'
        r'|味(?:が|は|も)?(?:良い|いい|よい|良かった|よかった|好き(?:だ|で|だった)|好み(?:だ|で|だった))'
    ),
    'x_service_positive_expression': (
        r'(?:接客|対応|スタッフ|店員|担当者|サポート)(?:の対応)?(?:が|は|も)?(?:とても|すごく)?'
        r'(?:丁寧(?:だ|で|だった)|親切(?:だ|で|だった)|良い|いい|よい|良かった|よかった|迅速(?:だ|で|だった))'
        r'|(?:丁寧|親切)(?:な対応|に対応してくれた|に対応していただいた)'
    ),
    'x_continuation_positive_expression': (
        r'(?:続け|継続し)(?:やすい|やすかった|やすくて)'
        r'|(?:無理なく|手軽に|楽に)(?:続けられる|続けられた|継続できる|継続できた)'
        r'|(?:続ける|継続する)(?:のが|ことが)?(?:楽(?:だ|で|だった)|簡単(?:だ|で|だった))'
    ),
    'x_symptom_improvement_expression': (
        r'(?:便秘|下痢|腹痛|頭痛|吐き気|胃痛)(?:が|は|も)?(?:改善した|改善しました|改善された|改善されました|治った|治りました|解消した|解消しました|解消された|解消されました|良くなった|よくなった|楽になった)'
        r'|(?:便通|お通じ)(?:が|は|も)?(?:良くなった|よくなった|改善した|改善しました)'
    ),
    'x_usability_mention': r'使い心地|使用感|飲み心地|飲みやす|食べやす|使いやす|飲みにく|食べにく|使いにく|溶けやす|溶けにく',
    'x_taste_mention': r'(?<![意趣興])味|美味|おいし|まずい|不味|甘い|苦い|酸っぱい',
    'x_service_mention': r'接客|対応|スタッフ|店員|担当者|サポート',
    'x_continuation_mention': r'続け|継続|続か|続き',
}


_X_DENY_SUFFIX = re.compile(
    r'^(?:(?:は|も)?(?:ない|なかった|なく|ありません|無い)|'
    r'わけ(?:で|じゃ)(?:は)?(?:ない|なかった|ありません)|'
    r'こと(?:は|が|も)(?:ない|なかった|ありません)|'
    r'とは(?:思わない|思いません|感じない|感じません|言えない|いえない|限らない)|'
    r'と(?:は)?(?:思わない|思いません)|'
    r'という(?:わけではない|ことはない|ことではない)|'
    r'と(?:言われた|聞いた|書かれていた)|'
    r'(?:という)?(?:噂|うわさ)|'
    r'か(?:心配|不安)|(?:から)?といって)'
)


_X_HYPOTHETICAL_PREFIX = re.compile(r'(?:もし|仮に|たとえば|例えば)[^。！？!?;；\n、,]{0,12}$')


_X_DENY_DIFFICULTY = re.compile(r'^(?:ことは(?:ない|なかった)|とは(?:思わない|思いません)|わけではない)')


_X_COMPILED = {key: re.compile(pattern) for key, pattern in NEW_TEXT_PATTERNS.items()}


def discordance_text_flags(text: str) -> dict[str, int]:

    text = normalized_text(text)
    flags = {}
    for feature, pattern in _X_COMPILED.items():
        present = False
        for match in pattern.finditer(text):
            if feature.endswith('_mention'):
                present = True
                break
            suffix = text[match.end():match.end()+30].lstrip()
            prefix = text[max(0, match.start()-16):match.start()]
            if _X_DENY_SUFFIX.search(suffix) or _X_HYPOTHETICAL_PREFIX.search(prefix):
                continue

            if suffix.startswith(('いと思', 'いです', 'ら', '場合', 'かもしれ', 'かも知れ')):
                if feature in {'x_effect_positive_expression', 'x_symptom_experience_expression'}:
                    continue
            if feature == 'x_cancel_difficulty_expression' and _X_DENY_DIFFICULTY.search(suffix):
                continue

            if feature == 'x_symptom_negated_expression' and suffix.startswith(('とは言えない', 'とも言えない')):
                continue
            present = True
            break
        flags[feature] = int(present)
    return flags


DISCORDANCE_CRITERIA = (
    ('rate_5pct_primary', 'weight_loss_percent', 5.0),
    ('rate_3pct_sensitivity', 'weight_loss_percent', 3.0),
    ('loss_1kg_sensitivity', 'weight_loss_kg', 1.0),
    ('loss_3kg_sensitivity', 'weight_loss_kg', 3.0),
)


DISCORDANCE_GROUPS = ['A', 'B', 'C', 'D', 'neutral_success', 'neutral_non_success']


DISCORDANCE_JAPANESE = False


DISCORDANCE_LABELS = {
    'A': ('A: 成功・高評価', 'A: Success / high rating'),
    'B': ('B: 成功・低評価', 'B: Success / low rating'),
    'C': ('C: 非成功・高評価', 'C: Below threshold / high rating'),
    'D': ('D: 非成功・低評価', 'D: Below threshold / low rating'),
    'neutral_success': ('成功・中立（★3）', 'Success / neutral (3 stars)'),
    'neutral_non_success': ('非成功・中立（★3）', 'Below threshold / neutral (3 stars)'),
}


DISCORDANCE_FEATURE_LABELS = {
    'cancellation_mention': ('解約への言及', 'Cancellation mention'),
    'x_cancel_difficulty_expression': ('解約困難の表現', 'Cancellation difficulty expression'),
    'symptom_mention': ('症状への言及', 'Symptom mention'),
    'x_symptom_experience_expression': ('症状経験を述べる表現', 'Symptom experience expression'),
    'x_symptom_negated_expression': ('症状を否定する表現', 'Negated symptom expression'),
    'price_mention': ('価格への言及', 'Price mention'),
    'x_price_negative_expression': ('価格不満の表現', 'Negative price expression'),
    'x_price_positive_expression': ('価格肯定の表現', 'Positive price expression'),
    'x_effect_positive_expression': ('減量・効果肯定の表現', 'Positive effect expression'),
    'x_effect_negative_expression': ('減量・効果否定の表現', 'Negative effect expression'),
    'x_usability_mention': ('使用感への言及', 'Usability mention'),
    'x_taste_mention': ('味への言及', 'Taste mention'),
    'x_service_mention': ('接客への言及', 'Service mention'),
    'x_continuation_mention': ('継続への言及', 'Continuation mention'),
    'x_usability_positive_expression': ('使用感肯定の表現', 'Positive usability expression'),
    'x_taste_positive_expression': ('味肯定の表現', 'Positive taste expression'),
    'x_service_positive_expression': ('接客肯定の表現', 'Positive service expression'),
    'x_continuation_positive_expression': ('続けやすさの表現', 'Ease of continuation expression'),
    'x_symptom_improvement_expression': ('症状改善を述べる表現', 'Symptom improvement expression'),
}


def configure_discordance_font(cfg: dict, out: Path) -> bool:

    global DISCORDANCE_JAPANESE
    from matplotlib import font_manager
    from matplotlib.ft2font import FT2Font
    wanted = ['Meiryo', 'Yu Gothic', 'Yu Gothic UI', 'MS Gothic', 'Noto Sans CJK JP',
              'Noto Sans JP', 'IPAexGothic', 'IPAGothic', 'TakaoGothic', 'Hiragino Sans']
    explicit = cfg.get('discordance_font_file')
    candidates = []
    if explicit:
        if not Path(explicit).is_file():
            raise ValueError('discordance_font_file が見つかりません。ローカルのフォントを指定してください。')
        font_manager.fontManager.addfont(str(explicit))
        candidates.append((font_manager.FontProperties(fname=str(explicit)).get_name(), str(explicit)))
    for family in wanted:
        for font in font_manager.fontManager.ttflist:
            if font.name == family:
                candidates.append((font.name, font.fname))
    selected = None
    required = set(map(ord, '減量成功非評価症状価格解約中立使用期間'))
    for family, path in candidates:
        try:
            if required.issubset(FT2Font(path).get_charmap()):
                selected = (family, path)
                break
        except (OSError, RuntimeError):
            continue
    DISCORDANCE_JAPANESE = selected is not None
    plt.rcParams['font.family'] = selected[0] if selected else 'DejaVu Sans'
    plt.rcParams['axes.unicode_minus'] = False
    note = ('インストール済み日本語フォントを使用。' if selected else
            'この工程では日本語フォントを検出できませんでした。作図用フォントはFigures.pyで別途確認します。CSVはUTF-8です。')
    save_json(out/'summary_tables'/'font_check.json', {
        'japanese_labels': DISCORDANCE_JAPANESE, 'font_family': selected[0] if selected else 'DejaVu Sans',
        'font_file': selected[1] if selected else None, 'note': note})
    return DISCORDANCE_JAPANESE


def _dc_label(japanese: str, english: str) -> str:
    return japanese if DISCORDANCE_JAPANESE else english


def _dc_group_label(group: str) -> str:
    return _dc_label(*DISCORDANCE_LABELS[group])


def _dc_ge(values: pd.Series, threshold: float) -> pd.Series:

    return values.ge(threshold) | pd.Series(
        np.isclose(values.to_numpy(float), threshold, rtol=0, atol=1e-10), index=values.index)


def _dc_groups(success: pd.Series, rating_band: pd.Series) -> pd.Series:
    result = pd.Series('ineligible', index=success.index, dtype='object')
    for code, flag, band in [('A',1,'high'),('B',1,'low'),('C',0,'high'),('D',0,'low'),
                             ('neutral_success',1,'neutral'),('neutral_non_success',0,'neutral')]:
        result.loc[success.eq(flag) & rating_band.eq(band)] = code
    return result


def prepare_discordance_fields(df: pd.DataFrame) -> pd.DataFrame:

    d = df.copy()
    usable = d['weight_usable'].fillna(False).astype(bool)

    rate = 100 * (d['weight_before_kg'] - d['weight_after_kg']) / d['weight_before_kg']
    usable &= np.isfinite(rate) & d['weight_before_kg'].gt(0)
    d['success_primary'] = np.nan
    d.loc[usable, 'success_primary'] = _dc_ge(rate.loc[usable], 5.0).astype(float)
    d['rating_band'] = 'missing'
    d.loc[d.rating.isin([1,2]), 'rating_band'] = 'low'
    d.loc[d.rating.eq(3), 'rating_band'] = 'neutral'
    d.loc[d.rating.isin([4,5]), 'rating_band'] = 'high'
    d['discordance_group'] = _dc_groups(d.success_primary, d.rating_band)
    return d


def _dc_save_figure(fig, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def _dc_group_summary(d: pd.DataFrame, group_col: str = 'discordance_group') -> pd.DataFrame:
    valid = d[group_col].isin(DISCORDANCE_GROUPS)
    n_total = int(valid.sum()); n_extreme = int(d[group_col].isin(['A','B','C','D']).sum())
    rows = []
    for group in DISCORDANCE_GROUPS:
        g = d[d[group_col].eq(group)]
        rows.append({
            'group':group,'definition_ja':DISCORDANCE_LABELS[group][0], 'n_reviews':len(g),
            'denominator_all_weight_rated':n_total,
            'share_all_weight_rated_pct':100*len(g)/n_total if n_total else np.nan,
            'denominator_extreme_ratings':n_extreme,
            'share_extreme_ratings_pct':100*len(g)/n_extreme if n_extreme and group in ['A','B','C','D'] else np.nan,
            'n_products':g.product_id.nunique(),
            'mean_loss_percent':g.weight_loss_percent.mean(),
            'median_loss_percent':g.weight_loss_percent.median(),
            'mean_loss_kg':g.weight_loss_kg.mean(), 'median_loss_kg':g.weight_loss_kg.median(),
            'mean_rating':g.rating.mean(), 'n_duration_known':int(g.duration_group_known.sum()),
            'n_nonblank_text':int(g.text_chars.gt(0).sum()),
        })
    return pd.DataFrame(rows)


def _dc_rates(groups: pd.Series) -> dict:
    counts = groups.value_counts()
    n = {code:int(counts.get(code,0)) for code in DISCORDANCE_GROUPS}
    success = n['A']+n['B']+n['neutral_success']
    nonsuccess = n['C']+n['D']+n['neutral_non_success']
    ab = n['A']+n['B']; cd = n['C']+n['D']; eligible = success+nonsuccess
    return {
        'n_eligible_weight_rated':eligible, 'n_success_all_ratings':success,
        'n_nonsuccess_all_ratings':nonsuccess, 'n_AB':ab, 'n_CD':cd,
        'n_B':n['B'],'n_C':n['C'],
        'B_pct_among_success_all_ratings':100*n['B']/success if success else np.nan,
        'B_pct_among_success_extreme_ratings':100*n['B']/ab if ab else np.nan,
        'C_pct_among_nonsuccess_all_ratings':100*n['C']/nonsuccess if nonsuccess else np.nan,
        'C_pct_among_nonsuccess_extreme_ratings':100*n['C']/cd if cd else np.nan,
        'B_pct_among_all_eligible':100*n['B']/eligible if eligible else np.nan,
        'C_pct_among_all_eligible':100*n['C']/eligible if eligible else np.nan,
    }


DISCORDANCE_MODEL_FEATURES = [
    'cancellation_mention', 'x_cancel_difficulty_expression', 'symptom_mention',
    'x_symptom_experience_expression', 'x_price_negative_expression',
    'x_effect_positive_expression', 'x_effect_negative_expression', 'log_chars',
]


DISCORDANCE_MODEL_CATEGORICAL = ['year', 'duration_group']


class DiscordanceModelError(ValueError):

    def __init__(self, message: str, details: dict):
        super().__init__(message)
        self.details = details


def absorbed_ols(df: pd.DataFrame, outcome: str, predictors: list[str],
                 categorical: list[str] | None = None) -> tuple[pd.DataFrame, dict, pd.DataFrame]:

    categorical = list(categorical or [])
    predictors = list(dict.fromkeys(predictors))
    required = [outcome, 'product_id'] + predictors + categorical
    missing = [column for column in required if column not in df]
    details = {'n_input_reviews': len(df), 'n_input_products': int(df.product_id.nunique()) if 'product_id' in df else 0,
               'outcome': outcome, 'requested_predictors': predictors,
               'categorical_controls': categorical, 'estimator': 'within-product OLS',
               'cluster_unit': 'product_id', 'iterative_convergence': 'not applicable: direct least squares'}
    if missing:
        raise DiscordanceModelError('必要な実測列がありません: ' + ', '.join(missing), details)
    d = df[required].copy()
    numeric = [outcome] + predictors
    for column in numeric:
        d[column] = pd.to_numeric(d[column], errors='coerce')
    invalid = ~np.isfinite(d[numeric].to_numpy(dtype=float)).all(axis=1)
    invalid |= d.product_id.isna().to_numpy() | d.product_id.astype(str).str.strip().eq('').to_numpy()
    details['missing_or_nonfinite_removed'] = int(invalid.sum())
    d = d.loc[~invalid].reset_index(drop=True)
    details['n_complete_before_singletons'] = len(d)
    details['n_products_complete_before_singletons'] = int(d.product_id.nunique())
    for column in categorical:
        d[column] = d[column].fillna('unknown').astype(str)
        d.loc[d[column].str.strip().eq(''), column] = 'unknown'
    counts = d.groupby('product_id', sort=False).size()
    singleton = d.product_id.map(counts).eq(1)
    details['singleton_reviews_removed'] = int(singleton.sum())
    details['singleton_products_removed'] = int(singleton.sum())
    d = d.loc[~singleton].reset_index(drop=True)
    n, g = len(d), int(d.product_id.nunique())
    details.update(n_reviews=n, n_products=g)
    if n < 30 or g < 2:
        raise DiscordanceModelError('推定を行わない: N<30 または商品クラスタ数<2。', details)
    x = d[predictors].astype(float)
    levels = {}
    for column in categorical:
        levels[column] = sorted(d[column].unique().tolist())
        x = pd.concat([x, pd.get_dummies(d[column], prefix=column, drop_first=True, dtype=float)], axis=1)
    details['categorical_levels'] = levels
    x, dropped_global = independent_columns(x)
    xw = x - x.groupby(d.product_id, sort=False).transform('mean')
    xw, dropped_within = independent_columns(xw)
    details['dropped_regressors'] = list(dict.fromkeys(dropped_global + dropped_within))
    details['drop_reason'] = 'constant, zero within-product variation, or rank deficiency; no p-value-based selection'
    yw = d[outcome] - d.groupby('product_id', sort=False)[outcome].transform('mean')
    a, y = xw.to_numpy(float), yw.to_numpy(float)
    k = a.shape[1]
    details.update(n_slopes=k, absorbed_product_parameters=g, residual_df=n-g-k)
    if not k:
        raise DiscordanceModelError('識別可能な商品内説明変数がありません。', details)
    if n <= g + k:
        raise DiscordanceModelError('商品固定効果を含む残差自由度が不足しています。', details)
    if np.var(y) < 1e-12:
        raise DiscordanceModelError('被説明変数の商品内変動がありません。', details)
    norms = np.linalg.norm(a, axis=0)
    z = a / norms
    condition = float(np.linalg.cond(z))
    details['scaled_condition_number'] = condition
    if not np.isfinite(condition) or condition > 1e10:
        raise DiscordanceModelError('説明変数の条件数が過大で安定して推定できません。', details)
    b_scaled = np.linalg.lstsq(z, y, rcond=None)[0]
    beta = b_scaled / norms
    u = y - z @ b_scaled
    codes = pd.factorize(d.product_id, sort=True)[0]
    scores = np.zeros((g, k))
    np.add.at(scores, codes, z * u[:, None])
    bread = np.linalg.inv(z.T @ z)
    correction = g / (g-1) * (n-1) / (n-g-k)
    covariance = correction * bread @ (scores.T @ scores) @ bread
    covariance = covariance / norms[:, None] / norms[None, :]
    table = regression_table(beta, covariance, xw.columns, g-1)
    table['n_reviews'] = n
    table['n_products'] = g
    table['residual_df'] = n-g-k
    table['cluster_t_df'] = g-1
    warnings_list = ['商品クラスタ数30未満: クラスタ漸近推論の精度に注意。'] if g < 30 else []
    details.update(status='estimated', cluster_t_df=g-1, cluster_correction=correction,
                   within_r_squared=1-float(u@u)/float(y@y),
                   warnings=warnings_list,
                   inference='product-clustered CR1; t reference with G-1 degrees of freedom',
                   interpretation='観察された口コミにおける条件付き関連。因果効果・医学的効能ではない。')
    if d[outcome].isin([0, 1]).all():

        fitted = (d[outcome]-yw) + (xw @ pd.Series(beta, index=xw.columns))
        details['lpm_fitted_outside_0_1_n'] = int(((fitted < 0) | (fitted > 1)).sum())
        details['lpm_note'] = 'LPM coefficients are probability-point differences; predictions may lie outside [0,1].'
    cov_frame = pd.DataFrame(covariance, index=xw.columns, columns=xw.columns)
    return table, details, cov_frame


def discordance_interaction_associations(table: pd.DataFrame, covariance: pd.DataFrame,
                                        info: dict) -> pd.DataFrame:

    beta = table.set_index('term').coefficient
    terms = list(covariance.index)
    specs = [
        ('cancellation_non_success', {'cancellation_mention': 1.}),
        ('cancellation_success', {'cancellation_mention': 1., 'success_x_cancellation': 1.}),
        ('success_minus_non_success', {'success_x_cancellation': 1.}),
    ]
    rows = []
    for label, weights in specs:
        absent = [term for term in weights if term not in terms]
        if absent:
            rows.append({'association': label, 'status': 'NOT_IDENTIFIED', 'reason': 'Dropped term(s): '+', '.join(absent),
                         'n_reviews': info['n_reviews'], 'n_products': info['n_products']})
            continue
        contrast = np.array([weights.get(term, 0.) for term in terms])
        coefficient = float(contrast @ beta.loc[terms].to_numpy())
        variance = float(contrast @ covariance.to_numpy() @ contrast)
        se = math.sqrt(max(variance, 0.))
        tdf = info['cluster_t_df']
        critical = stats.t.ppf(.975, tdf)
        pvalue = float(2*stats.t.sf(abs(coefficient/se), tdf)) if se > 1e-15 else np.nan
        rows.append({'association': label, 'status': 'estimated', 'coefficient': coefficient,
                     'cluster_se': se, 'ci_low': coefficient-critical*se, 'ci_high': coefficient+critical*se,
                     'p_value': pvalue, 'n_reviews': info['n_reviews'], 'n_products': info['n_products'],
                     'cluster_t_df': tdf, 'interpretation': '星の点数差、条件付き関連。介入効果ではない。'})
    return pd.DataFrame(rows)


def extract_features(df: pd.DataFrame, out: Path) -> pd.DataFrame:
    df = df.copy()
    for name, pattern in PATTERNS.items():
        df[name] = df.text_clean.str.contains(pattern, regex=True, na=False).astype('int8')
    return df


def extract_discordance_text(df: pd.DataFrame, cfg: dict, out: Path) -> pd.DataFrame:
    df = df.copy()
    strings = df.text_clean.fillna('').astype(str)
    cache = {}
    for text in strings.drop_duplicates():
        flags = discordance_text_flags(text)
        cache[text] = tuple(flags[key] for key in NEW_TEXT_FEATURES)
    values = np.asarray(strings.map(cache).tolist(), dtype=np.int8).reshape(len(df), len(NEW_TEXT_FEATURES))
    for i, name in enumerate(NEW_TEXT_FEATURES):
        df[name] = values[:, i]
    return df


def run_discordance_tables(df: pd.DataFrame, cfg: dict, out: Path) -> dict:
    dest = out / 'summary_tables'
    dest.mkdir(parents=True, exist_ok=True)
    eligible = df[df.discordance_group.isin(DISCORDANCE_GROUPS)]
    save_csv(_dc_group_summary(df), dest / 'group_counts_and_weight_change.csv')
    rates = _dc_rates(df.discordance_group)
    save_csv(pd.DataFrame([rates]), dest / 'mismatch_counts_and_rates.csv')
    keep = pd.Series(True, index=df.index)
    previous = len(df)
    flow = []
    for stage, condition in [
        ('all_target_matched_reviews', keep.copy()), ('valid_star_rating', df.rating.notna()),
        ('both_weights_parsed', df.weight_pair_parsed), ('both_weights_in_provisional_range', df.weight_in_range),
        ('weight_change_in_provisional_range', df.weight_change_in_range),
        ('finite_usable_weight_and_rating', df.discordance_group.isin(DISCORDANCE_GROUPS))]:
        keep &= condition.fillna(False)
        n = int(keep.sum())
        flow.append({'stage': stage, 'n_remaining': n, 'n_removed_at_stage': previous-n})
        previous = n
    save_csv(pd.DataFrame(flow), dest / 'weight_rating_sample_selection.csv')
    sensitivity = []
    for name, field, cutoff in DISCORDANCE_CRITERIA:
        success = pd.Series(np.nan, index=df.index)
        usable = df.success_primary.notna()
        success.loc[usable] = _dc_ge(df.loc[usable, field], cutoff).astype(float)
        sensitivity.append({'criterion': name, 'measure': field, 'threshold_ge': cutoff,
                            'is_primary': name == 'rate_5pct_primary', **_dc_rates(_dc_groups(success, df.rating_band))})
    save_csv(pd.DataFrame(sensitivity), dest / 'weight_loss_threshold_comparison.csv')
    for field, filename in [('duration_group', 'group_usage_duration.csv'),
                            ('category', 'group_product_categories.csv')]:
        t = eligible.groupby(['discordance_group', field], dropna=False).size().rename('n_reviews').reset_index()
        t['group_denominator'] = t.groupby('discordance_group').n_reviews.transform('sum')
        t['share_within_group_pct'] = 100*t.n_reviews/t.group_denominator
        save_csv(t, dest / filename)
    comparisons = []
    for comparison, first, second in [('B_vs_A', 'B', 'A'), ('C_vs_D', 'C', 'D')]:
        for feature in DISCORDANCE_FEATURE_LABELS:
            a = eligible.loc[eligible.discordance_group.eq(first) & eligible.text_chars.gt(0), feature]
            b = eligible.loc[eligible.discordance_group.eq(second) & eligible.text_chars.gt(0), feature]
            comparisons.append({'comparison': comparison, 'feature': feature,
                'n_first_with_text': len(a), 'n_second_with_text': len(b),
                'n_first_flagged': int(a.sum()), 'n_second_flagged': int(b.sum()),
                'first_prevalence_pct': 100*a.mean(), 'second_prevalence_pct': 100*b.mean(),
                'difference_percentage_points': 100*(a.mean()-b.mean())})
    save_csv(pd.DataFrame(comparisons), dest / 'group_text_expression_comparison.csv')
    categories = pd.DataFrame([{'category': cat, **_dc_rates(df.loc[df.category.eq(cat), 'discordance_group'])}
                               for cat in CATEGORIES])
    save_csv(categories, dest / 'category_mismatch_rates.csv')
    diagnostics = []
    for group in ['C', 'D']:
        g = eligible[eligible.discordance_group.eq(group)]
        for label, mask in [('gain', g.weight_loss_kg.lt(-1e-10)), ('unchanged', g.weight_loss_kg.abs().le(1e-10)),
                            ('positive_loss_below_5pct', g.weight_loss_kg.gt(1e-10))]:
            sub = g.loc[mask]
            diagnostics.append({'original_group': group, 'weight_change': label, 'n': len(sub),
                'group_denominator': len(g), 'share_pct': 100*len(sub)/len(g) if len(g) else np.nan,
                'mean_loss_percent': sub.weight_loss_percent.mean(), 'mean_rating': sub.rating.mean()})
    save_csv(pd.DataFrame(diagnostics), dest / 'weight_change_breakdown.csv')
    return rates


def run_discordance_models(df: pd.DataFrame, cfg: dict, out: Path) -> dict:
    dest = out / 'regression_results'
    dest.mkdir(parents=True, exist_ok=True)
    base = df[df.rating.notna() & df.success_primary.notna() & df.text_chars.gt(0)].copy()
    base['low_rating_discordance'] = base.rating.le(2).astype(float)
    base['success_x_cancellation'] = base.success_primary*base.cancellation_mention
    specs = [
        ('N1_A_success_low_FE', base[base.success_primary.eq(1)], 'low_rating_discordance', DISCORDANCE_MODEL_FEATURES),
        ('N1b_A_AB_low_FE', base[base.discordance_group.isin(['A', 'B'])], 'low_rating_discordance', DISCORDANCE_MODEL_FEATURES),
        ('N2_B_interaction_rating_FE', base, 'rating', ['success_primary', 'cancellation_mention', 'success_x_cancellation']
         + [x for x in DISCORDANCE_MODEL_FEATURES if x != 'cancellation_mention'])]
    filenames = {
        'N1_A_success_low_FE': 'low_rating_associations_5_percent_loss',
        'N1b_A_AB_low_FE': 'low_vs_high_ratings_5_percent_loss',
        'N2_B_interaction_rating_FE': 'rating_weight_loss_cancellation_interaction',
    }
    statuses = {}
    for name, sample, outcome, predictors in specs:
        try:
            table, info, covariance = absorbed_ols(sample, outcome, predictors, DISCORDANCE_MODEL_CATEGORICAL)
            focal = ['success_primary', 'cancellation_mention', 'success_x_cancellation'] if name.startswith('N2_') else ['cancellation_mention']
            info['unidentified_focal_terms'] = [term for term in focal if term not in set(table.term)]
            info['focus_identified'] = not info['unidentified_focal_terms']
            table['model'] = name
            table['model_status'] = info['status']
            save_csv(table, dest / (filenames[name]+'.csv'))
            save_csv(covariance.rename_axis('term').reset_index(), dest / (filenames[name]+'_covariance.csv'))
            if name.startswith('N2_'):
                save_csv(discordance_interaction_associations(table, covariance, info), dest / 'cancellation_associations_by_weight_loss.csv')
        except DiscordanceModelError as exc:
            info = {**exc.details, 'status': 'NOT_ESTIMABLE', 'reason': str(exc)}
        except (ValueError, np.linalg.LinAlgError) as exc:
            info = {'status': 'NOT_ESTIMABLE', 'n_input_reviews': len(sample), 'reason': str(exc)}
        statuses[name] = info
        save_json(dest / 'regression_details.json', statuses)
    save_csv(pd.DataFrame([{'model': name, **{k: v for k, v in info.items() if not isinstance(v, (list, dict))}}
                           for name, info in statuses.items()]), dest / 'regression_status.csv')
    return statuses


def plot_results(df: pd.DataFrame, out: Path) -> None:
    from Figures import make_figures
    make_figures(out, discordance_text_flags, PATTERNS)


def run_all(review: pd.DataFrame, product: pd.DataFrame, output_dir='outputs_private', config=None, demo=False) -> dict:
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    for key, value in (config or {}).items():
        if isinstance(value, dict) and isinstance(cfg.get(key), dict): cfg[key].update(value)
        else: cfg[key] = value
    out = Path(output_dir).resolve() / datetime.now().strftime('run_%Y%m%d_%H%M%S_%f')
    out.mkdir(parents=True)
    status = {'status': 'RUNNING', 'demo': demo, 'config': cfg, 'python': sys.version,
              'source_sha256': file_sha256(Path(__file__)), 'libraries': {x: importlib.metadata.version(x) for x in ['numpy', 'pandas', 'scipy', 'matplotlib']}}
    save_json(out / 'run_status.json', status)
    try:
        df, audit = prepare_data(review, product, cfg, out)
        df = df[df.category.isin(CATEGORIES) & df['_merge'].eq('both')].reset_index(drop=True)
        df = extract_features(df, out)
        df = extract_discordance_text(prepare_discordance_fields(df), cfg, out)
        configure_discordance_font(cfg, out)
        run_discordance_tables(df, cfg, out)
        status['models'] = run_discordance_models(df, cfg, out)
        plot_results(df, out)
        status['status'] = 'COMPLETED' if all(x['status'] == 'estimated' for x in status['models'].values()) else 'PARTIAL'
    except Exception as exc:
        status.update(status='FAILED', reason=f'{type(exc).__name__}: {exc}')
        raise
    finally: save_json(out / 'run_status.json', status)
    print(out)
    return {'output_dir': str(out), **status}


def demo_frames(n=1200, seed=42):
    rng = np.random.default_rng(seed)
    product = pd.DataFrame({'商品ID': [f'DEMO{i:03}' for i in range(48)],
        '親カテゴリ名': np.resize(['医療ダイエット', '', 'ダイエット食品', 'ダイエットドリンク'], 48),
        '子カテゴリ名': np.resize(['漢方薬', 'ダイエットサプリ', 'ゼリー', '茶'], 48)})
    before = np.round(rng.uniform(45, 110, n), 2)
    after = np.round(before-rng.choice([-3, 0, 1, 2, 3, 5, 8, 10], n), 2)
    phrases = ['解約しました。', '解約できませんでした。', '副作用はありませんでした。', '腹痛が出ました。',
        '価格が高い。', '効果がありました。', '効果がありませんでした。', '美味しい。', '続けやすい。']
    review = pd.DataFrame({'口コミID': [f'DEMO_R{i:06}' for i in range(n)],
        '商品ID': product['商品ID'].iloc[np.arange(n)%48].to_numpy(), '評価': rng.integers(1, 6, n).astype(str),
        '口コミ本文': ['人工の動作確認用例文です。'+''.join(p for p in phrases if rng.random()<.2) for i in range(n)],
        '投稿日時': [f'{y}/01/01' for y in rng.integers(2017, 2024, n)], '使用前体重': before.astype(str),
        '使用後体重': after.astype(str), '使用期間': rng.choice(['1週間', '1ヶ月', '3ヶ月', '6ヶ月', ''], n)})
    review.loc[rng.random(n)<.1, '使用後体重'] = ''
    review.loc[rng.random(n)<.1, '評価'] = '評価なし'
    return review, product


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=HERE / 'Maindata',
                        help='CSVの保存先（既定: Analysis.py と同じ場所の Maindata）')
    parser.add_argument('--output-dir', type=Path, default=HERE / 'outputs_private')
    parser.add_argument('--encoding', default='utf-8-sig')
    parser.add_argument('--config', type=Path)
    parser.add_argument('--demo', action='store_true', help='Run with fictional data')
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding='utf-8-sig')) if args.config else None
    if args.demo: review, product = demo_frames()
    else:
        missing = [name for name in ['review.csv', 'product.csv']
                   if not (args.data_dir / name).is_file()]
        if missing:
            parser.error(f"CSVが見つかりません: {', '.join(missing)}\n"
                         f"次のフォルダーに review.csv と product.csv を入れてください:\n"
                         f"{args.data_dir.resolve()}")
        review, product = [pd.read_csv(args.data_dir / name, dtype='string', encoding=args.encoding)
                           for name in ['review.csv', 'product.csv']]
    result = run_all(review, product, args.output_dir, config, args.demo)
    return 0 if result['status'] == 'COMPLETED' else 2


if __name__ == '__main__':
    raise SystemExit(main())
