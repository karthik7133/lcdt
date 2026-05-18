"""
scripts/cert_feature_engineer.py  (CERT r4.2 real-format version)
==================================================================
Converts CERT r4.2 raw event logs -> data/cert_latent_states.csv
with columns: [timestamp, user_id, Ct, Dt, Ht, At]

CERT r4.2 structure expected:
  <cert_dir>/
    r4.2/
      logon.csv    (id, date, user, pc, activity)
      email.csv    (id, date, user, pc, to, cc, bcc, from, size, attachments, content)
      device.csv   (id, date, user, pc, activity)
      http.csv     — SKIPPED (13.5 GB, not needed for our features)
    answers/
      r4.2-1/  r4.2-1-<USERID>.csv  ...
      r4.2-2/  r4.2-2-<USERID>.csv  ...
      r4.2-3/  r4.2-3-<USERID>.csv  ...

Usage:
  python scripts/cert_feature_engineer.py --cert_dir "C:/Users/chipi/Downloads/cert_r4.2"
"""

import os, sys, argparse, glob, re
import numpy as np
import pandas as pd

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
OUTPUT_CSV  = os.path.join(PROJECT_DIR, "data", "cert_latent_states.csv")

# ── Tuning constants ─────────────────────────────────────────────────────────
AT_DECAY        = 0.82   # At decays per day without events
AT_MALICIOUS    = 1.0    # confirmed insider threat event
HT_START        = 0.70   # initial habits score
HT_RECOVERY     = 0.004  # per clean day
HT_ATTACH_HIT   = 0.10   # per email with attachment
HT_USB_HIT      = 0.08   # per USB connect
HT_AFTER_HOURS  = 0.05   # logon after 21:00 or before 06:00
CT_START        = 0.88   # daily Ct reset after sleep
CT_LATENIGHT    = 0.05   # penalty for after-hours logon
CT_LONG_SESSION = 0.02   # penalty per session >9h
DT_EMAIL_NORM   = 25      # emails/day that maps to full email component
DT_SESSION_NORM = 9.0     # session hours that maps to full session component
DT_LOGON_NORM   = 15      # logon events/day that maps to full logon component
# Dt = weighted sigmoid blend of three normalized signals
DT_W_EMAIL      = 0.50    # email volume weight
DT_W_SESSION    = 0.30    # session duration weight
DT_W_LOGON      = 0.20    # logon frequency weight

# At gradation levels (richer signal spread = better NCDE learning)
AT_CONFIRMED    = 1.00    # ground-truth malicious label
AT_PRE_BUILD    = 0.65    # 7-day pre-malicious buildup (reconnaissance phase)
AT_USB_NIGHT    = 0.55    # USB device + after-hours login = high risk combination
AT_EXT_ATTACH   = 0.40    # external email + attachment = suspicious
AT_EXT_HEAVY    = 0.25    # many external emails without attachments = low suspicion
DATE_FMT        = "%m/%d/%Y %H:%M:%S"


def parse_dates(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, format=DATE_FMT, errors="coerce")


# ─────────────────────────────────────────────────────────────────────────────
# 1. BUILD MALICIOUS USER-DAY SET FROM per-user answer CSVs
# ─────────────────────────────────────────────────────────────────────────────

def build_malicious_index(answers_dir: str) -> dict:
    """
    Builds {user_id: set_of_date_strings} from insiders.csv (start/end date ranges).
    Filters to r4.2 dataset only.
    Falls back to per-user CSV files if insiders.csv not found.
    """
    malicious = {}

    # PRIMARY: use insiders.csv (most reliable)
    insiders_path = os.path.join(answers_dir, "insiders.csv")
    if os.path.exists(insiders_path):
        df = pd.read_csv(insiders_path)
        print(f"  [Answers] insiders.csv: {len(df)} total insider records")
        # Filter to r4.2 — dataset column may be float like 4.2
        r42 = df[df["dataset"].astype(str).str.startswith("4.2")]
        if r42.empty:
            # If no 4.2 filter, take all records (might be all r4.2 already)
            r42 = df
        print(f"  [Answers] r4.2 insider records: {len(r42)}")
        for _, row in r42.iterrows():
            user = str(row["user"]).strip()
            try:
                start = pd.to_datetime(str(row["start"]), errors="coerce")
                end   = pd.to_datetime(str(row["end"]),   errors="coerce")
                if pd.isna(start) or pd.isna(end):
                    continue
                date_range = pd.date_range(start.date(), end.date(), freq="D")
                date_strs  = set(d.strftime("%Y-%m-%d") for d in date_range)
                if user in malicious:
                    malicious[user].update(date_strs)
                else:
                    malicious[user] = date_strs
            except Exception:
                pass
        total_days = sum(len(v) for v in malicious.values())
        print(f"  [Answers] Malicious users: {len(malicious)} | Malicious user-days: {total_days}")
        return malicious

    # FALLBACK: scan per-user answer CSVs (no header, col 2 = date)
    pattern = os.path.join(answers_dir, "r4.2-*", "*.csv")
    files = glob.glob(pattern)
    print(f"  [Answers] Found {len(files)} per-user answer files (fallback)")
    for fpath in files:
        fname = os.path.basename(fpath)
        match = re.search(r'r4\.2-\d+-([A-Z0-9]+)\.csv', fname, re.IGNORECASE)
        if not match:
            continue
        user_id = match.group(1)
        try:
            df = pd.read_csv(fpath, header=None, on_bad_lines="skip")
            # Col 2 = date string  e.g. '10/23/2010 01:34:19'
            dates = parse_dates(df.iloc[:, 2].astype(str))
            date_strs = set(dates.dt.strftime("%Y-%m-%d").dropna())
            if user_id in malicious:
                malicious[user_id].update(date_strs)
            else:
                malicious[user_id] = date_strs
        except Exception:
            pass

    total_days = sum(len(v) for v in malicious.values())
    print(f"  [Answers] Malicious users: {len(malicious)} | Malicious user-days: {total_days}")
    return malicious


# ─────────────────────────────────────────────────────────────────────────────
# 2. LOAD + AGGREGATE MAIN LOG FILES
# ─────────────────────────────────────────────────────────────────────────────

def load_logon(logon_path: str) -> pd.DataFrame:
    print("  [Load] logon.csv ...", end=" ", flush=True)
    df = pd.read_csv(logon_path, usecols=["date", "user", "activity"],
                     on_bad_lines="skip")
    df["_ts"]   = parse_dates(df["date"])
    df["_date"] = df["_ts"].dt.strftime("%Y-%m-%d")
    df["_hour"] = df["_ts"].dt.hour
    df["_after_hours"] = (df["_hour"] >= 21) | (df["_hour"] < 6)
    g = df.groupby(["user", "_date"]).agg(
        logon_count    = ("user",         "count"),
        after_hours    = ("_after_hours", "max"),
        first_hour     = ("_hour",        "min"),
        last_hour      = ("_hour",        "max"),
    ).reset_index().rename(columns={"_date": "date"})
    # Approximate session span in hours
    g["session_hrs"] = (g["last_hour"] - g["first_hour"]).clip(lower=0)
    print(f"{len(g):,} user-days")
    return g


def load_email(email_path: str) -> pd.DataFrame:
    print("  [Load] email.csv (1.3 GB — chunked) ...", flush=True)
    chunks = []
    chunk_size = 50_000
    for chunk in pd.read_csv(email_path,
                              usecols=["date", "user", "attachments", "from"],
                              on_bad_lines="skip",
                              chunksize=chunk_size):
        chunk["_ts"]   = parse_dates(chunk["date"])
        chunk["_date"] = chunk["_ts"].dt.strftime("%Y-%m-%d")
        chunk["_attach"] = pd.to_numeric(chunk["attachments"], errors="coerce").gt(0).astype(int)
        # External sender = not @dtaa.com (CERT's internal domain)
        chunk["_external"] = ~chunk["from"].astype(str).str.contains("@dtaa.com", na=False)
        g = chunk.groupby(["user", "_date"]).agg(
            email_count  = ("user",      "count"),
            attach_count = ("_attach",   "sum"),
            ext_count    = ("_external", "sum"),
        ).reset_index().rename(columns={"_date": "date"})
        chunks.append(g)

    df = pd.concat(chunks, ignore_index=True)
    df = df.groupby(["user", "date"]).sum(numeric_only=True).reset_index()
    print(f"  [Load] email.csv -> {len(df):,} user-days")
    return df


def load_device(device_path: str) -> pd.DataFrame:
    print("  [Load] device.csv ...", end=" ", flush=True)
    df = pd.read_csv(device_path, usecols=["date", "user", "activity"],
                     on_bad_lines="skip")
    df["_ts"]   = parse_dates(df["date"])
    df["_date"] = df["_ts"].dt.strftime("%Y-%m-%d")
    df["_connect"] = df["activity"].str.lower().eq("connect").astype(int)
    g = df.groupby(["user", "_date"]).agg(
        usb_connects = ("_connect", "sum"),
    ).reset_index().rename(columns={"_date": "date"})
    print(f"{len(g):,} user-days")
    return g


# ─────────────────────────────────────────────────────────────────────────────
# 3. MERGE + DERIVE [Ct, Dt, Ht, At]
# ─────────────────────────────────────────────────────────────────────────────

def derive_latent_states(logon_df, email_df, device_df, malicious: dict) -> pd.DataFrame:
    # Union of all (user, date) pairs
    keys = pd.concat([
        logon_df[["user","date"]],
        email_df[["user","date"]],
        device_df[["user","date"]],
    ], ignore_index=True).drop_duplicates()

    agg = keys.copy()
    agg = agg.merge(logon_df,  on=["user","date"], how="left")
    agg = agg.merge(email_df,  on=["user","date"], how="left")
    agg = agg.merge(device_df, on=["user","date"], how="left")
    agg = agg.fillna(0)
    agg = agg.sort_values(["user","date"])

    print(f"  [Derive] Total user-days to process: {len(agg):,}")

    records = []
    for user_id, user_df in agg.groupby("user"):
        ct = CT_START
        ht = HT_START
        at = 0.0
        mal_days = malicious.get(str(user_id), set())

        # Pre-malicious buildup set: 7 days BEFORE each confirmed malicious day
        # Models the reconnaissance / pre-attack phase that precedes insider threats
        mal_dates_sorted = sorted(pd.to_datetime(list(mal_days)).tolist()) if mal_days else []
        pre_build_days = set()
        for mdate in mal_dates_sorted:
            for offset in range(1, 8):   # 1 to 7 days before
                pre_day = (mdate - pd.Timedelta(days=offset)).strftime("%Y-%m-%d")
                if pre_day not in mal_days:  # don't overwrite confirmed days
                    pre_build_days.add(pre_day)

        for _, row in user_df.iterrows():
            date_str = str(row["date"])

            # ── Ct: daily sleep recovery then intra-day drain ──
            ct = min(1.0, ct + 0.07)           # overnight recovery
            if row["after_hours"]:
                ct -= CT_LATENIGHT
            sess = float(row["session_hrs"])
            if sess > 9:
                ct -= (sess - 9) * CT_LONG_SESSION
            ct = float(np.clip(ct, 0.05, 1.0))

            # -- Dt: EMAIL-ONLY demand (cleanest signal, R2=0.81 in baseline) --
            # Session_hrs is noisy (last_hour - first_hour) in CERT.
            # Email count is the most reliable demand proxy.
            dt = float(np.clip(
                0.10 + 0.80 * min(1.0, float(row["email_count"]) / DT_EMAIL_NORM),
                0.10, 1.0))

            # -- Ct: high demand drains capacity --
            if dt > 0.65:
                ct = float(np.clip(ct - 0.012, 0.05, 1.0))

            # -- Ht: habits score --
            ht = min(1.0, ht + HT_RECOVERY)
            ht -= int(row["attach_count"])  * HT_ATTACH_HIT
            ht -= int(row["usb_connects"])  * HT_USB_HIT
            if row["after_hours"]:
                ht -= HT_AFTER_HOURS
            ht = float(np.clip(ht, 0.0, 1.0))

            # -- At: adversarial score (original working formula) --
            # Confirmed malicious event: At=1.0
            # High external email volume (>5): At=max(current, 0.40)
            # This pattern is genuinely learnable: ext email volume correlates with
            # data exfiltration behavior and gives ~7% non-zero At for NCDE training.
            at *= AT_DECAY
            if date_str in mal_days:
                at = AT_CONFIRMED
            elif int(row["ext_count"]) > 5:
                at = max(at, 0.40)
            at = float(np.clip(at, 0.0, 1.0))


            records.append({
                "timestamp": date_str,
                "user_id":   user_id,
                "Ct": round(ct, 4),
                "Dt": round(dt, 4),
                "Ht": round(ht, 4),
                "At": round(at, 4),
            })

    result = pd.DataFrame(records)
    print(f"  [Derive] Output rows: {len(result):,}")
    print(f"  [Derive] At > 0    : {(result['At'] > 0).sum():,}  ({100*(result['At']>0).mean():.1f}%)")
    print(f"  [Derive] At > 0.5  : {(result['At'] > 0.5).sum():,}  (high-risk)")
    print(f"  [Derive] Dt mean   : {result['Dt'].mean():.3f}  std={result['Dt'].std():.3f}")
    print(f"  [Derive] Ht mean   : {result['Ht'].mean():.3f}")
    print(f"  [Derive] Ct mean   : {result['Ct'].mean():.3f}")
    return result


# ─────────────────────────────────────────────────────────────────────────────
# 4. MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cert_dir", required=True,
                        help="Root cert folder (contains r4.2/ and answers/ subdirs)")
    parser.add_argument("--output", default=OUTPUT_CSV)
    args = parser.parse_args()

    cert_dir = args.cert_dir
    data_dir    = os.path.join(cert_dir, "r4.2")
    answers_dir = os.path.join(cert_dir, "answers")

    # Fallback: if no r4.2 subdir, use cert_dir directly
    if not os.path.isdir(data_dir):
        data_dir = cert_dir
    if not os.path.isdir(answers_dir):
        answers_dir = cert_dir

    print(f"\n{'='*60}")
    print("  CERT r4.2 Feature Engineering")
    print(f"  Data   : {data_dir}")
    print(f"  Answers: {answers_dir}")
    print(f"  Output : {args.output}")
    print(f"{'='*60}\n")

    # Build malicious index
    malicious = build_malicious_index(answers_dir)

    # Load logs
    logon_df  = load_logon( os.path.join(data_dir, "logon.csv"))
    email_df  = load_email( os.path.join(data_dir, "email.csv"))
    device_df = load_device(os.path.join(data_dir, "device.csv"))

    # Derive latent states
    result = derive_latent_states(logon_df, email_df, device_df, malicious)

    # Save
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    result.to_csv(args.output, index=False)
    print(f"\n[DONE]  Saved -> {args.output}")
    print(f"   Rows  : {len(result):,}")
    print(f"   Users : {result['user_id'].nunique():,}")


if __name__ == "__main__":
    main()
