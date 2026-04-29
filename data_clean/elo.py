import pandas as pd
import numpy as np

# ── Config ────────────────────────────────────────────────────────────────────
STARTING_ELO = 1500
K_SERIES     = 32   # series level
K_MODE       = 20   # map level (noisier than series)
LAN_MULT     = 1.5  # LAN results count more

# ── Core math ─────────────────────────────────────────────────────────────────
def expected_score(elo_a, elo_b):
    return 1 / (1 + 10 ** ((elo_b - elo_a) / 400))

def elo_update(winner_elo, loser_elo, k):
    exp = expected_score(winner_elo, loser_elo)
    return (
        winner_elo + k * (1 - exp),
        loser_elo  + k * (0 - (1 - exp))
    )

# ── Load data ─────────────────────────────────────────────────────────────────
ml = pd.read_csv('map_level.csv')
ml.loc[ml['mode'] == 'SCAR', 'mode'] = 'OVLD'

def get_map_order(match_id):
    import re
    m = re.match(r'2026_M(\d+)([QT])(\d+)S(\d+)', match_id)
    if not m:
        return 0
    major, stage, stage_num, series = int(m.group(1)), m.group(2), int(m.group(3)), int(m.group(4))
    stage_offset = 0 if stage == 'Q' else 10000
    return major * 100000 + stage_offset + series

ml['map_order']    = ml['match_id'].apply(get_map_order) * 10 + ml['map_number']
ml['series_order'] = ml['match_id'].apply(get_map_order)

# ── Series results ────────────────────────────────────────────────────────────
def get_series_results(ml):
    rows = []
    for match_id, group in ml.groupby('match_id'):
        away       = group['away_team'].iloc[0]
        home       = group['home_team'].iloc[0]
        is_lan     = group['is_lan'].iloc[0]
        series_ord = group['series_order'].iloc[0]
        away_maps  = (group['map_winner'] == away).sum()
        home_maps  = (group['map_winner'] == home).sum()
        winner, loser = (away, home) if away_maps > home_maps else (home, away)
        rows.append({
            'match_id':     match_id,
            'series_order': series_ord,
            'away':         away,
            'home':         home,
            'winner':       winner,
            'loser':        loser,
            'is_lan':       int(is_lan)
        })
    return pd.DataFrame(rows).sort_values('series_order').reset_index(drop=True)

series_results = get_series_results(ml)
teams = sorted(set(ml['away_team'].unique()) | set(ml['home_team'].unique()))

# ── Series Elo ────────────────────────────────────────────────────────────────
series_elo    = {t: STARTING_ELO for t in teams}
series_wins   = {t: 0 for t in teams}
series_losses = {t: 0 for t in teams}
series_streak = {t: 0 for t in teams}
h2h_wins      = {}

for _, row in series_results.iterrows():
    w, l   = row['winner'], row['loser']
    k      = K_SERIES * LAN_MULT if row['is_lan'] else K_SERIES

    series_elo[w], series_elo[l] = elo_update(series_elo[w], series_elo[l], k)
    series_wins[w]   += 1
    series_losses[l] += 1
    series_streak[w] += 1
    series_streak[l]  = 0
    h2h_wins[(w, l)]  = h2h_wins.get((w, l), 0) + 1

# ── Mode Elo ──────────────────────────────────────────────────────────────────
mode_elo      = {mode: {t: STARTING_ELO for t in teams} for mode in ['HP', 'SND', 'OVLD']}
mode_wins     = {mode: {t: 0 for t in teams} for mode in ['HP', 'SND', 'OVLD']}
mode_losses   = {mode: {t: 0 for t in teams} for mode in ['HP', 'SND', 'OVLD']}
h2h_mode_wins = {mode: {} for mode in ['HP', 'SND', 'OVLD']}

for _, row in ml.sort_values('map_order').iterrows():
    mode   = row['mode']
    away   = row['away_team']
    home   = row['home_team']
    winner = row['map_winner']
    loser  = home if winner == away else away
    k      = K_MODE * LAN_MULT if row['is_lan'] else K_MODE

    mode_elo[mode][winner], mode_elo[mode][loser] = elo_update(
        mode_elo[mode][winner], mode_elo[mode][loser], k
    )
    mode_wins[mode][winner]  += 1
    mode_losses[mode][loser] += 1
    key = (winner, loser)
    h2h_mode_wins[mode][key] = h2h_mode_wins[mode].get(key, 0) + 1

# ── Summary tables ────────────────────────────────────────────────────────────
def build_summary():
    rows = []
    for team in teams:
        total = series_wins[team] + series_losses[team]
        rows.append({
            'team':          team,
            'series_elo':    round(series_elo[team], 1),
            'streak':        series_streak[team],
            'series_wins':   series_wins[team],
            'series_losses': series_losses[team],
            'series_wr':     round(series_wins[team] / max(total, 1), 3),
            'hp_elo':        round(mode_elo['HP'][team], 1),
            'hp_wins':       mode_wins['HP'][team],
            'hp_losses':     mode_losses['HP'][team],
            'snd_elo':       round(mode_elo['SND'][team], 1),
            'snd_wins':      mode_wins['SND'][team],
            'snd_losses':    mode_losses['SND'][team],
            'ovld_elo':      round(mode_elo['OVLD'][team], 1),
            'ovld_wins':     mode_wins['OVLD'][team],
            'ovld_losses':   mode_losses['OVLD'][team],
        })
    return pd.DataFrame(rows).sort_values('series_elo', ascending=False).reset_index(drop=True)

def build_h2h_table():
    seen, rows = set(), []
    for (w, l), wins in h2h_wins.items():
        key = frozenset([w, l])
        if key not in seen:
            seen.add(key)
            rows.append({
                'team_a': w,
                'team_b': l,
                'a_wins': wins,
                'b_wins': h2h_wins.get((l, w), 0),
                'total':  wins + h2h_wins.get((l, w), 0),
            })
    return pd.DataFrame(rows).sort_values(['team_a', 'team_b']).reset_index(drop=True)

def build_h2h_mode_table():
    seen, rows = set(), []
    for mode in ['HP', 'SND', 'OVLD']:
        for (w, l), wins in h2h_mode_wins[mode].items():
            key = (mode, frozenset([w, l]))
            if key not in seen:
                seen.add(key)
                rows.append({
                    'mode':   mode,
                    'team_a': w,
                    'team_b': l,
                    'a_wins': wins,
                    'b_wins': h2h_mode_wins[mode].get((l, w), 0),
                    'total':  wins + h2h_mode_wins[mode].get((l, w), 0),
                })
    return pd.DataFrame(rows).sort_values(['mode', 'team_a', 'team_b']).reset_index(drop=True)

# ── Lookup helpers (imported by cdl_model.py) ─────────────────────────────────
def get_series_elo(team):
    return series_elo.get(team, STARTING_ELO)

def get_mode_elo(team, mode):
    return mode_elo[mode].get(team, STARTING_ELO)

def get_series_streak(team):
    return series_streak.get(team, 0)

def get_h2h_series_wr(team_a, team_b):
    """team_a series win rate vs team_b. Returns 0.5 if no history."""
    a = h2h_wins.get((team_a, team_b), 0)
    b = h2h_wins.get((team_b, team_a), 0)
    return a / (a + b) if (a + b) > 0 else 0.5

def get_h2h_mode_wr(team_a, team_b, mode):
    """team_a map win rate vs team_b in given mode. Returns 0.5 if no history."""
    a = h2h_mode_wins[mode].get((team_a, team_b), 0)
    b = h2h_mode_wins[mode].get((team_b, team_a), 0)
    return a / (a + b) if (a + b) > 0 else 0.5

# ── Print on run ──────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print(f"Config: K_series={K_SERIES} | K_mode={K_MODE} | LAN_mult={LAN_MULT}x\n")

    summary  = build_summary()
    h2h      = build_h2h_table()
    h2h_mode = build_h2h_mode_table()

    print("=== Team Elo & Record ===")
    print(summary.to_string(index=False))
    print("\n=== Head-to-Head Series Results ===")
    print(h2h.to_string(index=False))
    print("\n=== Head-to-Head Map Results by Mode ===")
    print(h2h_mode.to_string(index=False))