import pandas as pd
import numpy as np
import datetime

import sys
sys.path.append("/Users/derekdewald/Documents/Python/Github_Repo/d_py_functions")

from data_d_dicts import links

def data_preparation_checklist(df=pd.DataFrame(),word_list=[]):
    if len(df)==0:
        df = pd.read_csv(links['d_learning_notes_url'])

    final_df = pd.DataFrame()
    
    for word in word_list:
        temp_df = df[(df['Definition'].fillna("").str.contains(word,case=False))|(df['Categorization']==word)]
        final_df = pd.concat([final_df,temp_df])

    final_df = final_df.drop(['Source','Process','Categorization'],axis=1)
    display(final_df)

    return final_df

def create_baseball_stats(
    years=(1990, 1991, 1992, 1993, 1994),
    games_per_year=162,
    teams=None,
    players=None,
    seed=42,
):
    """
    Toy dataset for explanatory/EDA/examples.

    Generates one row per player per game, with basic batting outcomes and derived rates.

    Parameters
    ----------
    years : iterable[int]
        Seasons to generate.
    games_per_year : int
        Games per year.
    teams : list[str] | None
        Team/opponent list. If None, uses defaults.
    players : list[dict] | None
        Player definitions (name + profile). If None, uses built-in fictional roster.
        Each dict: {"name": str, "contact": float, "power": float, "discipline": float, "speed": float}
    seed : int | None
        Random seed for reproducibility. Use None for non-deterministic output.

    Returns
    -------
    pd.DataFrame
    """
    rng = np.random.default_rng(seed)

    if teams is None:
        teams = [
            "Montreal Expos",
            "Boston Red Sox",
            "New York Yankees",
            "Toronto Blue Jays",
            "Oakland Athletics",
            "Los Angeles Dodgers",
            "Chicago Cubs",
            "San Francisco Giants",
        ]

    # 10+ iconic fictional players (mix of movies/books/games)
    if players is None:
        players = [
            {"name": "Rick 'Wild Thing' Vaughn", "contact": 0.42, "power": 0.35, "discipline": 0.25, "speed": 0.40},
            {"name": "Pedro Cerrano",            "contact": 0.48, "power": 0.60, "discipline": 0.30, "speed": 0.35},
            {"name": "Roy Hobbs",                "contact": 0.62, "power": 0.58, "discipline": 0.45, "speed": 0.40},
            {"name": "Benny 'The Jet' Rodriguez", "contact": 0.55, "power": 0.25, "discipline": 0.35, "speed": 0.85},
            {"name": "Kenny Powers",             "contact": 0.40, "power": 0.45, "discipline": 0.20, "speed": 0.30},
            {"name": "Crash Davis",              "contact": 0.58, "power": 0.38, "discipline": 0.55, "speed": 0.30},
            {"name": "Jake Taylor",              "contact": 0.50, "power": 0.30, "discipline": 0.50, "speed": 0.30},
            {"name": "Roger Dorn",               "contact": 0.52, "power": 0.33, "discipline": 0.48, "speed": 0.25},
            {"name": "Henry Rowengartner",       "contact": 0.35, "power": 0.20, "discipline": 0.15, "speed": 0.55},
            {"name": "Willie Mays Hays",          "contact": 0.46, "power": 0.22, "discipline": 0.28, "speed": 0.92},
            {"name": "Casey 'Back Back Back' Jones","contact":0.45,"power": 0.50, "discipline": 0.30, "speed": 0.40},
            {"name": "Homer Simpson",             "contact": 0.38, "power": 0.55, "discipline": 0.10, "speed": 0.15},
        ]

    # Helper: clamp to [0,1]
    def clamp01(x: float) -> float:
        return float(max(0.0, min(1.0, x)))

    rows = []
    for year in years:
        for game in range(1, games_per_year + 1):
            opponent = rng.choice(teams)
            for p in players:
                name = p["name"]
                contact = clamp01(p.get("contact", 0.50))
                power = clamp01(p.get("power", 0.35))
                discipline = clamp01(p.get("discipline", 0.35))
                speed = clamp01(p.get("speed", 0.35))

                # Plate appearances (PA) per game: 3–6-ish
                pa = int(rng.integers(3, 7))

                # Walk/HBP probability increases with discipline
                p_bb = 0.05 + 0.12 * discipline
                p_hbp = 0.003 + 0.01 * (1 - discipline)  # a little arbitrary/fun

                # Strikeout probability decreases with contact
                p_so = 0.12 + 0.20 * (1 - contact)

                # Hit-in-play probability: driven by contact, slightly reduced by K/BB events
                # We'll model "AB" as PA - BB - HBP - SF (simplified)
                # First decide BB/HBP/SO counts using multinomial-ish draws.
                bb = int(rng.binomial(pa, p_bb))
                remaining = pa - bb
                hbp = int(rng.binomial(remaining, p_hbp))
                remaining -= hbp
                so = int(rng.binomial(remaining, p_so))
                balls_in_play = remaining - so

                # On balls in play: chance of hit based on contact
                p_hit = 0.18 + 0.22 * contact
                hits_in_play = int(rng.binomial(balls_in_play, p_hit))

                # Allocate hit types: more power => more HR/2B/3B
                # We assign hit types among hits_in_play (and allow HR as subset of all hits).
                # Keep probabilities modest but responsive.
                p_hr = 0.03 + 0.10 * power
                p_3b = 0.01 + 0.03 * speed
                p_2b = 0.06 + 0.06 * power

                # Draw hit type per hit
                hr = three_b = two_b = one_b = 0
                for _ in range(hits_in_play):
                    r = rng.random()
                    if r < p_hr:
                        hr += 1
                    elif r < p_hr + p_3b:
                        three_b += 1
                    elif r < p_hr + p_3b + p_2b:
                        two_b += 1
                    else:
                        one_b += 1

                hits = one_b + two_b + three_b + hr

                # Sac flies (tiny chance): simplified
                sf = int(rng.binomial(max(pa - bb - hbp, 0), 0.02))

                # AB excludes BB/HBP/SF (simplified)
                ab = max(pa - bb - hbp - sf, 0)

                # If AB is 0, force hits 0 for consistency
                if ab == 0:
                    hits = one_b = two_b = three_b = hr = 0
                    so = 0  # optional

                # Runs/RBI: noisy but correlated with extra-base hits and hits
                rbi = int(min(hits + hr + two_b, rng.poisson(0.6 + 0.25 * hr + 0.10 * hits)))
                runs = int(min(hits + bb + hbp, rng.poisson(0.7 + 0.15 * (two_b + three_b + hr))))

                # Steals: driven by speed and times-on-base
                tob = hits + bb + hbp
                sb = int(rng.binomial(tob, 0.02 + 0.10 * speed))

                # Derived stats (avoid div-by-zero)
                ba = (hits / ab) if ab > 0 else np.nan
                obp = ((hits + bb + hbp) / (pa)) if pa > 0 else np.nan
                tb = one_b + 2 * two_b + 3 * three_b + 4 * hr
                slg = (tb / ab) if ab > 0 else np.nan
                ops = (obp + slg) if (pd.notna(obp) and pd.notna(slg)) else np.nan

                rows.append(
                    {
                        "PLAYER": name,
                        "OPPONENT": opponent,
                        "YEAR": int(year),
                        "GAME_NO": int(game),
                        "PA": int(pa),
                        "AB": int(ab),
                        "H": int(hits),
                        "1B": int(one_b),
                        "2B": int(two_b),
                        "3B": int(three_b),
                        "HR": int(hr),
                        "BB": int(bb),
                        "HBP": int(hbp),
                        "SO": int(so),
                        "SF": int(sf),
                        "RBI": int(rbi),
                        "R": int(runs),
                        "SB": int(sb),
                        "BA": ba,
                        "OBP": obp,
                        "SLG": slg,
                        "OPS": ops,
                    }
                )

    df = pd.DataFrame(rows)

    # Helpful ordering for EDA
    col_order = [
        "YEAR", "GAME_NO", "PLAYER", "OPPONENT",
        "PA", "AB", "H", "1B", "2B", "3B", "HR", "BB", "HBP", "SO", "SF", "R", "RBI", "SB",
        "BA", "OBP", "SLG", "OPS",
    ]
    return df[col_order]