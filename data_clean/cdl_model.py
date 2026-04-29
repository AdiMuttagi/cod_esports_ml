import pandas as pd
import numpy as np
from itertools import combinations
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# ── Import Elo lookups ────────────────────────────────────────────────────────
from elo import (
    get_series_elo, get_mode_elo, get_series_streak,
    get_h2h_series_wr, get_h2h_mode_wr
)

def parse_time(val):
    if pd.isna(val):
        return np.nan
    val = str(val).strip()
    if ':' in val:
        parts = val.split(':')
        return int(parts[0]) * 60 + int(parts[1])
    try:
        return float(val)
    except:
        return np.nan

ml   = pd.read_csv('map_level.csv')
hp   = pd.read_csv('player_stats_hp.csv')
ovld = pd.read_csv('player_stats_ovld.csv')
snd  = pd.read_csv('player_stats_snd.csv')

ml.loc[ml['mode'] == 'SCAR', 'mode'] = 'OVLD'

for col in ['hill_time_sec', 'avg_hill_time_sec', 'contested_hill_time_sec']:
    hp[col] = hp[col].apply(parse_time)

hp['hill_time_per_death'] = hp['hill_time_sec'] / hp['deaths'].replace(0, np.nan)
hp['hill_time_per_kill']  = hp['hill_time_sec'] / hp['kills'].replace(0, np.nan)
hp['contested_ratio']     = hp['contested_hill_time_sec'] / hp['hill_time_sec'].replace(0, np.nan)

common_cols = ['match_id', 'map_number', 'mode', 'map_name', 'team', 'opponent',
               'player_slot', 'player_name', 'map_winner', 'kills', 'deaths', 'damage']
snd = pd.concat([snd, hp[hp['mode'] == 'SND'][common_cols].copy()], ignore_index=True)
hp  = hp[hp['mode'] == 'HP'].copy()

def get_map_order(match_id):
    import re
    m = re.match(r'2026_M(\d+)([QT])(\d+)S(\d+)', match_id)
    if not m:
        return 0
    major, stage, stage_num, series = int(m.group(1)), m.group(2), int(m.group(3)), int(m.group(4))
    stage_offset = 0 if stage == 'Q' else 10000
    return major * 100000 + stage_offset + series

ml['map_order'] = ml['match_id'].apply(get_map_order) * 10 + ml['map_number']
for df in [hp, ovld, snd]:
    df['map_order'] = df['match_id'].apply(get_map_order) * 10 + df['map_number']
    df['is_lan']    = df['match_id'].apply(lambda x: 1 if '_T' in x else 0)

DECAY    = 0.99
LAN_BOOST = 1.5

# ── Chemistry score ───────────────────────────────────────────────────────────
def build_chemistry_scores(player_df, decay=0.99):
    rosters = (
        player_df.drop_duplicates(subset=['match_id', 'map_number', 'team', 'player_name'])
        [['team', 'map_order', 'player_name']]
        .groupby(['team', 'map_order'])['player_name']
        .apply(set).reset_index()
    )
    rosters = rosters.sort_values('map_order')
    results = []
    for team, group in rosters.groupby('team'):
        group = group.sort_values('map_order').reset_index(drop=True)
        pair_history = {}
        for idx, (_, row) in enumerate(group.iterrows()):
            current_roster = row['player_name']
            current_order  = row['map_order']
            current_pairs  = list(combinations(sorted(current_roster), 2))
            if idx == 0:
                chemistry = 0.0
            else:
                pair_scores = []
                for pair in current_pairs:
                    key = frozenset(pair)
                    if key in pair_history:
                        total = sum(w * (decay ** (current_order - mo))
                                    for mo, w in pair_history[key])
                        pair_scores.append(min(total, 1.0))
                    else:
                        pair_scores.append(0.0)
                chemistry = np.mean(pair_scores)
            results.append({'team': team, 'map_order': current_order,
                             'chemistry_score': chemistry})
            for pair in current_pairs:
                key = frozenset(pair)
                if key not in pair_history:
                    pair_history[key] = []
                pair_history[key].append((current_order, 1.0))
    return pd.DataFrame(results)

all_players = pd.concat([
    hp[['match_id', 'map_number', 'team', 'map_order', 'player_name']],
    ovld[['match_id', 'map_number', 'team', 'map_order', 'player_name']],
    snd[['match_id', 'map_number', 'team', 'map_order', 'player_name']],
]).drop_duplicates()

chemistry_df = build_chemistry_scores(all_players, decay=DECAY)
latest_chemistry = (
    chemistry_df.sort_values('map_order').groupby('team').last()
    .reset_index()[['team', 'chemistry_score']]
)

# ── Weighted aggregation ──────────────────────────────────────────────────────
def weighted_mean(df, stat_cols, group_col='team'):
    results = []
    for team, group in df.groupby(group_col):
        group = group.sort_values('map_order')
        n = len(group)
        weights = np.array([DECAY ** (n - 1 - i) for i in range(n)])
        weights *= np.where(group['is_lan'] == 1, LAN_BOOST, 1.0)
        weights /= weights.sum()
        row = {'team': team}
        for col in stat_cols:
            vals = pd.to_numeric(group[col], errors='coerce')
            row[col] = np.nansum(vals * weights)
        results.append(row)
    return pd.DataFrame(results)

def get_score_margins(ml):
    rows = []
    for _, r in ml.iterrows():
        rows.append({'team': r['away_team'], 'mode': r['mode'], 'map_name': r['map_name'],
                     'score_margin': (r['away_score'] - r['home_score']) / (250 if r['mode'] == 'HP' else 1),
                     'map_order': r['map_order'], 'is_lan': r['is_lan']})
        rows.append({'team': r['home_team'], 'mode': r['mode'], 'map_name': r['map_name'],
                     'score_margin': (r['home_score'] - r['away_score']) / (250 if r['mode'] == 'HP' else 1),
                     'map_order': r['map_order'], 'is_lan': r['is_lan']})
    return pd.DataFrame(rows)

score_margins = get_score_margins(ml)

def get_weighted_score_margin(score_margins, mode):
    results = []
    df = score_margins[score_margins['mode'] == mode].copy()
    for team, group in df.groupby('team'):
        group = group.sort_values('map_order')
        n = len(group)
        weights = np.array([DECAY ** (n - 1 - i) for i in range(n)])
        weights *= np.where(group['is_lan'] == 1, LAN_BOOST, 1.0)
        weights /= weights.sum()
        results.append({'team': team, 'score_margin': np.sum(group['score_margin'] * weights)})
    return pd.DataFrame(results)

def get_map_score_margins(score_margins):
    results = []
    for (team, mode, map_name), group in score_margins.groupby(['team', 'mode', 'map_name']):
        group = group.sort_values('map_order')
        n = len(group)
        weights = np.array([DECAY ** (n - 1 - i) for i in range(n)])
        weights *= np.where(group['is_lan'] == 1, LAN_BOOST, 1.0)
        weights /= weights.sum()
        results.append({'team': team, 'col': f'margin_{mode}_{map_name}',
                        'score_margin': np.sum(group['score_margin'] * weights)})
    df = pd.DataFrame(results)
    return df.pivot_table(index='team', columns='col', values='score_margin').reset_index()

map_score_margins = get_map_score_margins(score_margins)

mode_stat_cols = {
    'HP':   ['kills', 'deaths', 'damage', 'hill_time_sec', 'obj_kills',
             'hill_time_per_death', 'hill_time_per_kill', 'contested_ratio'],
    'OVLD': ['kills', 'deaths', 'damage', 'overloads', 'obj_kills'],
    'SND':  ['kills', 'deaths', 'damage', 'first_bloods', 'bombs_planted'],
}

hp_agg   = weighted_mean(hp,   mode_stat_cols['HP'])
ovld_agg = weighted_mean(ovld, mode_stat_cols['OVLD'])
snd_agg  = weighted_mean(snd,  mode_stat_cols['SND'])

for mode, agg in [('HP', hp_agg), ('OVLD', ovld_agg), ('SND', snd_agg)]:
    margin_df = get_weighted_score_margin(score_margins, mode)
    agg['score_margin'] = agg['team'].map(margin_df.set_index('team')['score_margin'])

for agg in [hp_agg, ovld_agg, snd_agg]:
    agg['chemistry_score'] = agg['team'].map(latest_chemistry.set_index('team')['chemistry_score'])

# ── Add Elo and H2H features ──────────────────────────────────────────────────
def add_elo_features(agg, mode):
    agg['series_elo']  = agg['team'].apply(get_series_elo)
    agg['mode_elo']    = agg['team'].apply(lambda t: get_mode_elo(t, mode))
    agg['streak']      = agg['team'].apply(get_series_streak)
    return agg

hp_agg   = add_elo_features(hp_agg,   'HP')
ovld_agg = add_elo_features(ovld_agg, 'OVLD')
snd_agg  = add_elo_features(snd_agg,  'SND')

def prefix_cols(df, prefix):
    return df.rename(columns={c: f'{prefix}_{c}' for c in df.columns if c != 'team'})

hp_agg   = prefix_cols(hp_agg,   'hp')
ovld_agg = prefix_cols(ovld_agg, 'ovld')
snd_agg  = prefix_cols(snd_agg,  'snd')

def get_mode_win_rate_weighted(df):
    results = []
    df = df.drop_duplicates(subset=['match_id', 'map_number', 'team']).copy()
    df['won'] = (df['map_winner'] == df['team']).astype(int)
    for (team, mode), group in df.groupby(['team', 'mode']):
        group = group.sort_values('map_order')
        n = len(group)
        weights = np.array([DECAY ** (n - 1 - i) for i in range(n)])
        weights *= np.where(group['is_lan'] == 1, LAN_BOOST, 1.0)
        weights /= weights.sum()
        wr = np.sum(group['won'] * weights)
        results.append({'team': team, 'mode': mode, 'win_rate': wr})
    return pd.DataFrame(results)

mode_win_rates = pd.concat([
    get_mode_win_rate_weighted(hp[['match_id','map_number','team','mode','map_winner','map_order','is_lan']]),
    get_mode_win_rate_weighted(ovld[['match_id','map_number','team','mode','map_winner','map_order','is_lan']]),
    get_mode_win_rate_weighted(snd[['match_id','map_number','team','mode','map_winner','map_order','is_lan']])
])
mode_win_rates = mode_win_rates.pivot(index='team', columns='mode', values='win_rate').reset_index()
mode_win_rates.columns = ['team', 'hp_win_rate', 'ovld_win_rate', 'snd_win_rate']

def get_map_win_rates_weighted(ml):
    ml = ml.copy()
    rows = []
    for _, row in ml.iterrows():
        for team in [row['home_team'], row['away_team']]:
            rows.append({'team': team, 'mode': row['mode'], 'map_name': row['map_name'],
                         'won': int(row['map_winner'] == team),
                         'map_order': row['map_order'], 'is_lan': row['is_lan']})
    df = pd.DataFrame(rows)
    results = []
    for (team, mode, map_name), group in df.groupby(['team', 'mode', 'map_name']):
        group = group.sort_values('map_order')
        n = len(group)
        weights = np.array([DECAY ** (n - 1 - i) for i in range(n)])
        weights *= np.where(group['is_lan'] == 1, LAN_BOOST, 1.0)
        weights /= weights.sum()
        wr = np.sum(group['won'] * weights)
        results.append({'team': team, 'col': f'wr_{mode}_{map_name}', 'win_rate': wr})
    wr_df = pd.DataFrame(results)
    return wr_df.pivot_table(index='team', columns='col', values='win_rate').reset_index()

map_win_rates = get_map_win_rates_weighted(ml)

def build_mode_features(ml, agg_df, mode, prefix, mode_win_rates,
                        map_win_rates, map_score_margins):
    maps = ml[ml['mode'] == mode].copy()
    ts   = agg_df.copy()
    ts   = ts.merge(mode_win_rates[['team', f'{prefix}_win_rate']], on='team')

    mode_map_wr_cols     = ['team'] + [c for c in map_win_rates.columns    if c.startswith(f'wr_{mode}_')]
    mode_map_margin_cols = ['team'] + [c for c in map_score_margins.columns if c.startswith(f'margin_{mode}_')]
    ts = ts.merge(map_win_rates[mode_map_wr_cols],         on='team', how='left')
    ts = ts.merge(map_score_margins[mode_map_margin_cols], on='team', how='left')

    teamA_stats = ts.copy()
    teamA_stats.columns = ['away_team'] + [f'A_{c}' for c in ts.columns if c != 'team']
    teamB_stats = ts.copy()
    teamB_stats.columns = ['home_team'] + [f'B_{c}' for c in ts.columns if c != 'team']

    maps = maps.merge(teamA_stats, on='away_team').merge(teamB_stats, on='home_team')

    stat_cols = [c for c in ts.columns if c != 'team']
    rows = []
    rng  = np.random.default_rng(42)

    for _, row in maps.iterrows():
        flip = rng.integers(0, 2)
        feat_row = {}
        map_wr_col     = f'wr_{mode}_{row["map_name"]}'
        map_margin_col = f'margin_{mode}_{row["map_name"]}'

        # H2H features — computed from elo module directly
        away, home = row['away_team'], row['home_team']
        h2h_series = get_h2h_series_wr(away, home)
        h2h_mode   = get_h2h_mode_wr(away, home, mode)

        for c in stat_cols:
            if c.startswith('wr_') and c != map_wr_col:
                continue
            if c.startswith('margin_') and c != map_margin_col:
                continue
            a_val = row[f'A_{c}']
            b_val = row[f'B_{c}']
            if flip:
                a_val, b_val = b_val, a_val
            feat_row[f'A_{c}'] = a_val
            feat_row[f'B_{c}'] = b_val
            feat_row[f'diff_{c}'] = a_val - b_val

        # Add H2H as features (flip-aware)
        feat_row['A_h2h_series_wr'] = h2h_series if not flip else (1 - h2h_series)
        feat_row['B_h2h_series_wr'] = (1 - h2h_series) if not flip else h2h_series
        feat_row['diff_h2h_series_wr'] = feat_row['A_h2h_series_wr'] - feat_row['B_h2h_series_wr']
        feat_row['A_h2h_mode_wr'] = h2h_mode if not flip else (1 - h2h_mode)
        feat_row['B_h2h_mode_wr'] = (1 - h2h_mode) if not flip else h2h_mode
        feat_row['diff_h2h_mode_wr'] = feat_row['A_h2h_mode_wr'] - feat_row['B_h2h_mode_wr']

        away_won = row['map_winner'] == row['away_team']
        feat_row['target'] = int(away_won) if not flip else int(not away_won)
        rows.append(feat_row)

    df = pd.DataFrame(rows).fillna(0.5)
    return df.dropna(), ts, map_win_rates

hp_df,   hp_ts,   _ = build_mode_features(ml, hp_agg,   'HP',   'hp',   mode_win_rates, map_win_rates, map_score_margins)
ovld_df, ovld_ts, _ = build_mode_features(ml, ovld_agg, 'OVLD', 'ovld', mode_win_rates, map_win_rates, map_score_margins)
snd_df,  snd_ts,  _ = build_mode_features(ml, snd_agg,  'SND',  'snd',  mode_win_rates, map_win_rates, map_score_margins)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

candidates = {
    'XGBoost': lambda: XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.1,
                                      eval_metric='logloss', random_state=42),
    'LogReg':  lambda: Pipeline([('scaler', StandardScaler()),
                                  ('model', LogisticRegression(max_iter=2000, random_state=42))]),
    'RF':      lambda: RandomForestClassifier(n_estimators=100, max_depth=3, random_state=42),
}

print("=== Model Comparison ===")
best_model_name = {}
for mode, df in [('HP', hp_df), ('OVLD', ovld_df), ('SND', snd_df)]:
    X = df.drop('target', axis=1)
    y = df['target']
    best_score = -1
    for name, factory in candidates.items():
        scores = cross_val_score(factory(), X, y, cv=cv, scoring='accuracy')
        print(f"{mode} {name}: {scores.mean():.3f} (+/- {scores.std():.3f})")
        if scores.mean() > best_score:
            best_score = scores.mean()
            best_model_name[mode] = name

print("\n=== Training Final Models (best per mode) ===")
models = {}
for mode, df, ts in [('HP', hp_df, hp_ts), ('OVLD', ovld_df, ovld_ts), ('SND', snd_df, snd_ts)]:
    X = df.drop('target', axis=1)
    y = df['target']
    name = best_model_name[mode]
    m    = candidates[name]()
    scores = cross_val_score(m, X, y, cv=cv, scoring='accuracy')
    print(f"{mode} → {name} | {len(df)} maps | CV: {scores.mean():.3f} (+/- {scores.std():.3f})")
    m.fit(X, y)
    models[mode] = (m, X.columns.tolist())

# ── Prediction ────────────────────────────────────────────────────────────────
def predict_series(team_a, team_b, map_rotation, best_of=5):
    wins_needed = (best_of // 2) + 1
    print(f"\n=== {team_a} vs {team_b} (BO{best_of}) ===")

    # Print Elo context
    print(f"Series Elo  : {team_a}={get_series_elo(team_a):.0f}  {team_b}={get_series_elo(team_b):.0f}")
    print(f"H2H series  : {team_a} {get_h2h_series_wr(team_a, team_b):.0%} vs {team_b} {get_h2h_series_wr(team_b, team_a):.0%}")
    print(f"Streak      : {team_a}={get_series_streak(team_a)}W  {team_b}={get_series_streak(team_b)}W")
    print()

    mode_ts = {'HP': hp_ts, 'OVLD': ovld_ts, 'SND': snd_ts}
    a_wins, b_wins = 0, 0

    for map_num, mode, map_name in map_rotation:
        if a_wins == wins_needed or b_wins == wins_needed:
            break
        ts = mode_ts[mode]
        model, feat_cols = models[mode]

        a_row = ts[ts['team'] == team_a].iloc[0]
        b_row = ts[ts['team'] == team_b].iloc[0]

        map_wr_col     = f'wr_{mode}_{map_name}'
        map_margin_col = f'margin_{mode}_{map_name}'
        feat_row = {}

        for c in ts.columns:
            if c == 'team':
                continue
            if c.startswith('wr_') and c != map_wr_col:
                continue
            if c.startswith('margin_') and c != map_margin_col:
                continue
            feat_row[f'A_{c}'] = a_row[c]
            feat_row[f'B_{c}'] = b_row[c]
            feat_row[f'diff_{c}'] = a_row[c] - b_row[c]

        # H2H features
        h2h_s = get_h2h_series_wr(team_a, team_b)
        h2h_m = get_h2h_mode_wr(team_a, team_b, mode)
        feat_row['A_h2h_series_wr']   = h2h_s
        feat_row['B_h2h_series_wr']   = 1 - h2h_s
        feat_row['diff_h2h_series_wr'] = h2h_s - (1 - h2h_s)
        feat_row['A_h2h_mode_wr']     = h2h_m
        feat_row['B_h2h_mode_wr']     = 1 - h2h_m
        feat_row['diff_h2h_mode_wr']  = h2h_m - (1 - h2h_m)

        feat_df = pd.DataFrame([feat_row]).reindex(columns=feat_cols, fill_value=0.5)
        prob    = model.predict_proba(feat_df)[0]
        pred    = model.predict(feat_df)[0]
        winner  = team_a if pred == 1 else team_b

        if pred == 1:
            a_wins += 1
        else:
            b_wins += 1

        h2h_mode_str = f"H2H {mode}: {team_a} {get_h2h_mode_wr(team_a, team_b, mode):.0%}"
        print(f"Map {map_num} ({mode} - {map_name}): {team_a} {prob[1]:.0%} vs {team_b} {prob[0]:.0%} → {winner}  [{team_a} {a_wins}-{b_wins} {team_b}]  |  {h2h_mode_str}")

    series_winner = team_a if a_wins == wins_needed else team_b
    print(f"\nPredicted series winner: {series_winner} ({a_wins}-{b_wins})")

predict_series(
    team_a='PAR',
    team_b='MIA',
    best_of=5,
    map_rotation=[
        (1, 'HP',   'Colossus'),
        (2, 'SND',  'Raid'),
        (3, 'OVLD', 'Exposure'),
        (4, 'HP',   'Scar'),
        (5, 'SND',  'Den'),
    ]
)