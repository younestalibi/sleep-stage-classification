"""
shared_data.py
Load Sleep-EDF directly from MNE, extract Welch PSD features,
balance the training set, then apply StandardScaler + PCA.

Class distribution (25 subjects, 46,688 epochs):
  W   25,389  54.4%   -- majority, 13.9× more than N1
  N1   1,830   3.9%   -- rarest
  N2  11,655  25.0%
  N3   3,152   6.8%
  REM  4,662  10.0%

Feature vector: 6 channels × 21 features = 126-dim random vector per epoch.
Balancing: cap each class at N1_count × CAP_RATIO in TRAIN only.
TEST set is NEVER resampled — evaluation metrics are honest.

References
[1] Kemp B. et al. (2000) IEEE Trans. Biomed. Eng. 47(9)   -- Sleep-EDF dataset
[2] Berry R. et al. (2012) AASM Scoring Manual v2.4        -- staging rules
[3] Welch P.D. (1967) IEEE Trans. Audio Electroacoust.     -- PSD estimation
[4] Hotelling H. (1933) J. Educ. Psychol. 24(6)           -- PCA
[5] He H. & Garcia E. (2009) IEEE TKDE 21(9)              -- imbalanced learning
"""
import os, warnings
import numpy as np
from scipy import signal as sig
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import mne
from mne.datasets.sleep_physionet.age import fetch_data

warnings.filterwarnings("ignore")
mne.set_log_level("ERROR")

# ── constants ──────────────────────────────────────────────────────────────────
DATA_PATH   = os.path.join(os.path.expanduser("~"), "mne_data", "sleep_edf_demo")
STAGE_MAP   = {"Sleep stage W":0, "Sleep stage 1":1, "Sleep stage 2":2,
                "Sleep stage 3":3, "Sleep stage 4":3, "Sleep stage R":4}
STAGE_NAMES = ["W", "N1", "N2", "N3", "REM"]
N_STAGES    = 5
BANDS       = {"delta":(0.5,4.), "theta":(4.,8.), "alpha":(8.,12.),
                "sigma":(12.,16.), "beta":(16.,30.), "total":(0.5,30.)}
N_PCA       = 30       # captures ~92% variance
CAP_RATIO   = 2.0      # train cap = N1_count × CAP_RATIO


# ── feature extraction ─────────────────────────────────────────────────────────
def _bandpower(f, psd, lo, hi):
    m = (f >= lo) & (f < hi)
    return float(np.trapz(psd[m], f[m])) if m.any() else 0.0

def _sp_entropy(psd, eps=1e-12):
    p = np.clip(psd / (psd.sum() + eps), eps, 1.0)
    return float(-np.sum(p * np.log(p)))

def _ch_features(x, sfreq):
    """21 scalar random variables from one EEG/EOG/EMG channel of one 30-s epoch."""
    x = np.nan_to_num(x.astype(np.float64)) - x.mean()
    f, psd = sig.welch(x, fs=sfreq,
                        nperseg=min(len(x), int(4 * sfreq)),
                        scaling="density")
    bp  = {k: _bandpower(f, psd, lo, hi) for k, (lo, hi) in BANDS.items()}
    rel = [bp[b] / max(bp["total"], 1e-12)
           for b in ("delta","theta","alpha","sigma","beta")]
    return np.nan_to_num(np.array([
        x.mean(), x.std(), x.min(), x.max(), np.median(x),
        np.percentile(x, 25), np.percentile(x, 75),
        float(np.sqrt(np.mean(x**2))),
        float(np.sum(np.abs(np.diff(x)))),
        bp["delta"], bp["theta"], bp["alpha"], bp["sigma"],
        bp["beta"],  bp["total"],
        *rel, _sp_entropy(psd)
    ]))   # 21-dim

def epoch_to_features(ep, sfreq):
    """(n_ch, n_times) → 126-dim random vector  (6 channels × 21 features)."""
    return np.concatenate([_ch_features(ch, sfreq) for ch in ep])


# ── MNE loading ────────────────────────────────────────────────────────────────
def _pick_channels(raw):
    return [ch for ch in raw.ch_names
            if "EVENT" not in ch.upper() and "MARKER" not in ch.upper()]

def load_all_subjects(subjects=range(25), recordings=(1,)):
    """Load Sleep-EDF EDF files via MNE. Returns list of subject dicts."""
    os.makedirs(DATA_PATH, exist_ok=True)
    rows = []
    pairs = fetch_data(subjects=list(subjects), recording=list(recordings),
                       path=DATA_PATH, on_missing="warn")
    for idx, (psg, hyp) in enumerate(pairs):
        subj = list(subjects)[idx // len(recordings)]
        try:
            raw   = mne.io.read_raw_edf(psg, stim_channel="Event marker",
                                         infer_types=True, preload=True, verbose="error")
            annot = mne.read_annotations(hyp)
            onsets = [a["onset"] for a in annot if a["description"] in STAGE_MAP]
            if not onsets:
                continue
            annot.crop(max(0., onsets[0] - 1800), onsets[-1] + 1800)
            raw.set_annotations(annot, emit_warning=False)
            events, _ = mne.events_from_annotations(raw, event_id=STAGE_MAP,
                                                      chunk_duration=30.)
            picks  = _pick_channels(raw)
            epochs = mne.Epochs(raw, events, event_id=STAGE_MAP,
                                 tmin=0., tmax=30. - 1./raw.info["sfreq"],
                                 baseline=None, preload=True,
                                 picks=picks, verbose="error")
            sfreq  = float(epochs.info["sfreq"])
            data   = epochs.get_data()
            y      = epochs.events[:, 2].astype(np.int64)
            mask   = np.isin(y, range(N_STAGES))
            X      = np.vstack([epoch_to_features(ep, sfreq) for ep in data[mask]])
            rows.append({"subject": subj, "X": X, "y": y[mask],
                          "n_epochs": int(mask.sum())})
            print(f"  subj {subj:2d}: {mask.sum():4d} epochs, {len(picks)} ch")
        except Exception as e:
            print(f"  subj {subj:2d}: skipped ({e})")
    return rows


# ── stratified undersampling ───────────────────────────────────────────────────
def _balance(X, y, rng, cap_ratio=CAP_RATIO):
    """
    Undersample majority classes so max_class ≤ min_class × cap_ratio.
    Only applied to TRAIN. TEST is never touched.  [He & Garcia 2009]
    """
    counts = np.bincount(y, minlength=N_STAGES)
    min_c  = counts[counts > 0].min()
    cap    = int(min_c * cap_ratio)
    Xp, yp = [], []
    for cls in range(N_STAGES):
        idx = np.where(y == cls)[0]
        if len(idx) == 0:
            continue
        if len(idx) > cap:
            idx = rng.choice(idx, size=cap, replace=False)
        Xp.append(X[idx]); yp.append(y[idx])
    Xb = np.vstack(Xp); yb = np.concatenate(yp)
    perm = rng.permutation(len(yb))
    return Xb[perm], yb[perm]


# ── main pipeline ──────────────────────────────────────────────────────────────
def build_train_test(rows, n_test_subj=6, cap_ratio=CAP_RATIO, n_pca=N_PCA):
    """
    Split by subject → balance TRAIN → StandardScale → PCA.

    Returns
    -------
    X_tr, X_te   : float32 (n, n_pca)
    y_tr, y_te   : int64
    tr_lens, te_lens : list[int] — per-subject lengths for HMM
    scaler, pca  : fitted sklearn objects
    train_s, test_s : subject ID lists
    class_w      : (N_STAGES,) inverse-frequency weights
    """
    rng     = np.random.RandomState(42)
    subjs   = sorted({r["subject"] for r in rows})
    train_s = subjs[:-n_test_subj]
    test_s  = subjs[-n_test_subj:]

    def concat(sl):
        Xs, ys, ls = [], [], []
        for r in rows:
            if r["subject"] in sl:
                Xs.append(r["X"]); ys.append(r["y"]); ls.append(len(r["y"]))
        return np.vstack(Xs), np.concatenate(ys), ls

    X_tr_raw, y_tr_raw, _ = concat(train_s)
    X_te_raw, y_te, te_lens = concat(test_s)

    print("\nTrain distribution BEFORE balancing:")
    cnt = np.bincount(y_tr_raw, minlength=N_STAGES)
    for i, s in enumerate(STAGE_NAMES):
        print(f"  {s:>4}: {cnt[i]:6d}  ({cnt[i]/len(y_tr_raw)*100:5.1f}%)")

    X_tr_bal, y_tr = _balance(X_tr_raw, y_tr_raw, rng, cap_ratio)

    print(f"\nTrain distribution AFTER balancing (cap_ratio={cap_ratio}):")
    cnt2 = np.bincount(y_tr, minlength=N_STAGES)
    for i, s in enumerate(STAGE_NAMES):
        print(f"  {s:>4}: {cnt2[i]:6d}  ({cnt2[i]/len(y_tr)*100:5.1f}%)")
    print(f"  total: {len(y_tr):,}  (was {len(y_tr_raw):,})")

    cnt2f   = np.where(cnt2.astype(float) == 0, 1, cnt2.astype(float))
    class_w = len(y_tr) / (N_STAGES * cnt2f)
    print("\nClass weights  w_i = N / (K · N_i):")
    for i, s in enumerate(STAGE_NAMES):
        print(f"  {s:>4}: {class_w[i]:.3f}")

    scaler   = StandardScaler()
    X_tr_sc  = scaler.fit_transform(X_tr_bal.astype(np.float32))
    X_te_sc  = scaler.transform(X_te_raw.astype(np.float32))

    n_comp = min(n_pca, X_tr_sc.shape[1])
    pca    = PCA(n_components=n_comp, random_state=42)
    X_tr   = pca.fit_transform(X_tr_sc).astype(np.float32)
    X_te   = pca.transform(X_te_sc).astype(np.float32)

    cum = pca.explained_variance_ratio_.cumsum()[-1]
    print(f"\nPCA: {n_comp} components → {cum*100:.1f}% variance retained")
    print(f"Train: {len(y_tr):,}   Test: {len(y_te):,}   dim: {X_tr.shape[1]}")

    n_s   = len(train_s)
    base  = len(y_tr) // n_s
    tr_lens = [base] * n_s
    tr_lens[-1] += len(y_tr) - sum(tr_lens)

    return (X_tr, X_te, y_tr, y_te,
            tr_lens, te_lens,
            scaler, pca,
            train_s, test_s,
            class_w)


# ── metric helper ──────────────────────────────────────────────────────────────
def print_metrics(y_true, y_pred, label=""):
    """Print Accuracy + Macro-F1 + Cohen kappa together."""
    from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score
    acc   = accuracy_score(y_true, y_pred)
    mf1   = f1_score(y_true, y_pred, average="macro", zero_division=0)
    kappa = cohen_kappa_score(y_true, y_pred)
    tag   = f"[{label}]  " if label else ""
    print(f"{tag}Acc={acc:.4f}  Macro-F1={mf1:.4f}  Kappa={kappa:.4f}")
    return acc, mf1, kappa
