import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold

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

ml = pd.read_csv('map_level.csv')
hp = pd.read_csv('player_stats_hp.csv')
ovld = pd.read_csv('player_stats_ovld.csv')
snd = pd.read_csv('player_stats_snd.csv')

ml.loc[ml['mode'] == 'SCAR', 'mode'] = 'OVLD'

for col in ['hill_time_sec', 'avg_hill_time_sec', 'contested_hill_time_sec']:
    hp[col] = hp[col].apply(parse_time)

common_cols = ['match_id', 'map_number', 'mode', 'map_name', 'team', 'opponent',
               'player_slot', 'player_name', 'map_winner', 'kills', 'deaths', 'damage']
snd = pd.concat([snd, hp[hp['mode'] == 'SND'][common_cols].copy()], ignore_index=True)
hp = hp[hp['mode'] == 'HP'].copy()

# Assign a global chronological order to each map
# M1Q1 < M1T1, series number within each stage preserves order
def get_map_order(match_id):
    import re
    m = re.match(r'2026_M(\d+)([QT])(\d+)S(\d+)', match_id)
    if not m:
        return 0
    major, stage, stage_num, series = int(m.group(1)), m.group(2), int(m.group(3)), int(m.group(4))
    stage_offset = 0 if stage == 'Q' else 10000
    return major * 100000 + stage_offset + series

ml['map_order'] = ml['match_id'].apply(get_map_order) * 10 + ml['map_number']

# Add map_order to player dfs for weighting
for df in [hp, ovld, snd]:
    df['map_order'] = df['match_id'].apply(get_map_order) * 10 + df['map_number']
    df['is_lan'] = df['match_id'].apply(lambda x: 1 if '_T' in x else 0)

# Recency + LAN weighted team averages
DECAY = 0.95
LAN_BOOST = 1.5

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

mode_stat_cols = {
    'HP':   ['kills', 'deaths', 'damage', 'hill_time_sec', 'obj_kills'],
    'OVLD': ['kills', 'deaths', 'damage', 'overloads', 'obj_kills'],
    'SND':  ['kills', 'deaths', 'damage', 'first_bloods', 'bombs_planted'],
}

hp_agg   = weighted_mean(hp,   mode_stat_cols['HP'])
ovld_agg = weighted_mean(ovld, mode_stat_cols['OVLD'])
snd_agg  = weighted_mean(snd,  mode_stat_cols['SND'])

def prefix_cols(df, prefix):
    return df.rename(columns={c: f'{prefix}_{c}' for c in df.columns if c != 'team'})

hp_agg   = prefix_cols(hp_agg,   'hp')
ovld_agg = prefix_cols(ovld_agg, 'ovld')
snd_agg  = prefix_cols(snd_agg,  'snd')

# Recency weighted win rates per mode
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
    get_mode_win_rate_weighted(hp[['match_id', 'map_number', 'team', 'mode', 'map_winner', 'map_order', 'is_lan']]),
    get_mode_win_rate_weighted(ovld[['match_id', 'map_number', 'team', 'mode', 'map_winner', 'map_order', 'is_lan']]),
    get_mode_win_rate_weighted(snd[['match_id', 'map_number', 'team', 'mode', 'map_winner', 'map_order', 'is_lan']])
])
mode_win_rates = mode_win_rates.pivot(index='team', columns='mode', values='win_rate').reset_index()
mode_win_rates.columns = ['team', 'hp_win_rate', 'ovld_win_rate', 'snd_win_rate']

# Recency weighted map-specific win rates
def get_map_win_rates_weighted(ml):
    ml = ml.copy()
    rows = []
    for _, row in ml.iterrows():
        for team in [row['home_team'], row['away_team']]:
            rows.append({
                'team': team,
                'mode': row['mode'],
                'map_name': row['map_name'],
                'won': int(row['map_winner'] == team),
                'map_order': row['map_order'],
                'is_lan': row['is_lan']
            })
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
print(f"Map-specific win rate columns: {[c for c in map_win_rates.columns if c != 'team']}")

def build_mode_features(ml, agg_df, mode, prefix, mode_win_rates, map_win_rates):
    maps = ml[ml['mode'] == mode].copy()
    ts = agg_df.copy()
    ts = ts.merge(mode_win_rates[['team', f'{prefix}_win_rate']], on='team')
    mode_map_cols = ['team'] + [c for c in map_win_rates.columns if c.startswith(f'wr_{mode}_')]
    ts = ts.merge(map_win_rates[mode_map_cols], on='team', how='left')

    teamA_stats = ts.copy()
    teamA_stats.columns = ['away_team'] + [f'A_{c}' for c in ts.columns if c != 'team']
    teamB_stats = ts.copy()
    teamB_stats.columns = ['home_team'] + [f'B_{c}' for c in ts.columns if c != 'team']

    maps = maps.merge(teamA_stats, on='away_team').merge(teamB_stats, on='home_team')

    stat_cols = [c for c in ts.columns if c != 'team']
    rows = []
    rng = np.random.default_rng(42)

    for _, row in maps.iterrows():
        flip = rng.integers(0, 2)
        feat_row = {}
        map_col = f'wr_{mode}_{row["map_name"]}'
        for c in stat_cols:
            if c.startswith('wr_') and c != map_col:
                continue
            a_val = row[f'A_{c}']
            b_val = row[f'B_{c}']
            if flip:
                a_val, b_val = b_val, a_val
            feat_row[f'A_{c}'] = a_val
            feat_row[f'B_{c}'] = b_val
            feat_row[f'diff_{c}'] = a_val - b_val
        away_won = row['map_winner'] == row['away_team']
        feat_row['target'] = int(away_won) if not flip else int(not away_won)
        rows.append(feat_row)

    df = pd.DataFrame(rows).fillna(0.5)
    return df.dropna(), ts, map_win_rates

hp_df,   hp_ts,   _ = build_mode_features(ml, hp_agg,   'HP',   'hp',   mode_win_rates, map_win_rates)
ovld_df, ovld_ts, _ = build_mode_features(ml, ovld_agg, 'OVLD', 'ovld', mode_win_rates, map_win_rates)
snd_df,  snd_ts,  _ = build_mode_features(ml, snd_agg,  'SND',  'snd',  mode_win_rates, map_win_rates)

models = {}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for mode, df in [('HP', hp_df), ('OVLD', ovld_df), ('SND', snd_df)]:
    X = df.drop('target', axis=1)
    y = df['target']
    m = XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.1,
                      eval_metric='logloss', random_state=42)
    scores = cross_val_score(m, X, y, cv=cv, scoring='accuracy')
    print(f"{mode} model — {len(df)} maps | CV accuracy: {scores.mean():.3f} (+/- {scores.std():.3f})")
    m.fit(X, y)
    models[mode] = (m, X.columns.tolist())

def predict_series(team_a, team_b, map_rotation, best_of=5):
    wins_needed = (best_of // 2) + 1
    print(f"\n=== {team_a} vs {team_b} Series Prediction (BO{best_of}) ===")
    mode_ts = {'HP': hp_ts, 'OVLD': ovld_ts, 'SND': snd_ts}
    a_wins = 0
    b_wins = 0

    for map_num, mode, map_name in map_rotation:
        if a_wins == wins_needed or b_wins == wins_needed:
            break
        ts = mode_ts[mode]
        model, feat_cols = models[mode]
        a_row = ts[ts['team'] == team_a].iloc[0]
        b_row = ts[ts['team'] == team_b].iloc[0]
        map_col = f'wr_{mode}_{map_name}'
        feat_row = {}
        for c in ts.columns:
            if c == 'team':
                continue
            if c.startswith('wr_') and c != map_col:
                continue
            feat_row[f'A_{c}'] = a_row[c]
            feat_row[f'B_{c}'] = b_row[c]
            feat_row[f'diff_{c}'] = a_row[c] - b_row[c]

        feat_df = pd.DataFrame([feat_row]).reindex(columns=feat_cols, fill_value=0.5)
        prob = model.predict_proba(feat_df)[0]
        pred = model.predict(feat_df)[0]
        winner = team_a if pred == 1 else team_b

        if pred == 1:
            a_wins += 1
        else:
            b_wins += 1

        print(f"Map {map_num} ({mode} - {map_name}): {team_a} {prob[1]:.0%} vs {team_b} {prob[0]:.0%} → {winner} wins  [{team_a} {a_wins}-{b_wins} {team_b}]")

    series_winner = team_a if a_wins == wins_needed else team_b
    print(f"\nPredicted series winner: {series_winner} ({a_wins}-{b_wins})")

predict_series(
    team_a='PAR',
    team_b='TX',
    best_of=7,
    map_rotation=[
        (1, 'HP',   'Colossus'),
        (2, 'SND',  'Raid'),
        (3, 'OVLD', 'Scar'),
        (4, 'HP',   'Exposure'),
        (5, 'SND',  'Exposure'),
        (6, 'OVLD', 'Exposure'),
        (7, 'HP',   'Scar'),
    ]
)