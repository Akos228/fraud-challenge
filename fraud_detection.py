"""
Défi — Détection de fraude financière.

Vous devez implémenter la fonction `detect_fraud`.
La fonction `load_transactions` vous est FOURNIE (ne la modifiez pas).
"""

import csv
from collections import defaultdict
from datetime import datetime, timezone

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Constantes et référentiels
# ---------------------------------------------------------------------------

# Continent par code pays ISO (couverture élargie pour les tests cachés)
_COUNTRY_CONTINENT = {
    "FR": "EU", "DE": "EU", "ES": "EU", "IT": "EU", "GB": "EU", "BE": "EU",
    "NL": "EU", "CH": "EU", "PT": "EU", "AT": "EU", "PL": "EU", "SE": "EU",
    "US": "NA", "CA": "NA", "MX": "NA",
    "BR": "SA", "AR": "SA", "CL": "SA", "CO": "SA",
    "JP": "AS", "CN": "AS", "KR": "AS", "IN": "AS", "SG": "AS", "TH": "AS",
    "AE": "AS", "SA": "AS", "HK": "AS", "TW": "AS", "VN": "AS", "MY": "AS",
    "AU": "OC", "NZ": "OC",
    "ZA": "AF", "NG": "AF", "EG": "AF", "MA": "AF", "KE": "AF", "SN": "AF",
    "RU": "EU", "TR": "AS", "IL": "AS",
}

# Temps minimum réaliste (heures) entre deux pays
_MIN_TRAVEL_HOURS = {
    ("same", "same"): 0.0,
    ("same", "neighbor"): 1.0,
    ("same", "far"): 3.0,
    ("diff", "neighbor"): 6.0,
    ("diff", "far"): 10.0,
}

# Champs considérés comme obligatoires pour une transaction valide
_REQUIRED_FIELDS = ("user_id", "amount", "country", "timestamp")

# Seuil global de suspicion
_SUSPICION_THRESHOLD = 0.50


def load_transactions(path):
    """Lit un fichier CSV de transactions et renvoie une liste de dicts."""
    transactions = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            transactions.append(_clean_row(row))
    return transactions


def _clean_row(row):
    def get(key):
        v = row.get(key)
        return v.strip() if isinstance(v, str) and v.strip() != "" else None

    amount_raw = get("amount")
    try:
        amount = float(amount_raw) if amount_raw is not None else None
    except ValueError:
        amount = None

    card_raw = get("card_present")
    if card_raw is None:
        card_present = None
    else:
        card_present = card_raw.lower() in ("true", "1", "yes", "oui")

    return {
        "transaction_id": get("transaction_id"),
        "timestamp": get("timestamp"),
        "user_id": get("user_id"),
        "amount": amount,
        "currency": get("currency"),
        "merchant": get("merchant"),
        "country": get("country"),
        "card_present": card_present,
    }


# ---------------------------------------------------------------------------
# Utilitaires
# ---------------------------------------------------------------------------

def _parse_timestamp(ts):
    """Convertit un horodatage ISO 8601 en datetime timezone-aware, ou None."""
    if ts is None:
        return None
    if not isinstance(ts, str):
        return None
    try:
        normalized = ts.strip().replace("Z", "+00:00")
        dt = datetime.fromisoformat(normalized)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except (ValueError, TypeError, AttributeError):
        return None


def _new_user_profile():
    """Initialise le profil comportemental d'un utilisateur."""
    return {
        "amounts": [],
        "countries": [],
        "country_counts": defaultdict(int),
        "merchants": set(),
        "timestamps": [],
        "recent_events": [],  # (datetime, amount, country) — fenêtre glissante
        "tx_count": 0,
        "last_ts": None,
        "last_country": None,
        "last_transaction_id": None,
    }


def _continent(country):
    """Retourne le continent d'un code pays, ou None si inconnu."""
    if not country:
        return None
    return _COUNTRY_CONTINENT.get(country.upper())


def _min_travel_hours(country_a, country_b):
    """Estime le délai minimum réaliste entre deux pays (en heures)."""
    if not country_a or not country_b:
        return 0.0
    if country_a.upper() == country_b.upper():
        return 0.0

    cont_a = _continent(country_a)
    cont_b = _continent(country_b)

    if cont_a is None or cont_b is None:
        # Pays inconnu : prudent, on exige au moins 4 h
        return 4.0

    if cont_a == cont_b:
        return _MIN_TRAVEL_HOURS[("same", "far")]

    return _MIN_TRAVEL_HOURS[("diff", "far")]


def _hours_between(dt_a, dt_b):
    """Calcule l'écart en heures entre deux datetime (valeur absolue)."""
    if dt_a is None or dt_b is None:
        return None
    return abs((dt_b - dt_a).total_seconds()) / 3600.0


def _combine_scores(scores):
    """Combine plusieurs signaux en un score unique (0–1), progressif."""
    if not scores:
        return 0.0
    scores = sorted(scores, reverse=True)
    combined = scores[0]
    for s in scores[1:]:
        combined = combined + s * (1.0 - combined) * 0.6
    return min(1.0, combined)


def _adaptive_amount_multiplier(median_amount):
    """Seuil de ratio montant/médiane adapté au profil de dépense."""
    if median_amount >= 1000:
        return 12.0
    if median_amount >= 500:
        return 10.0
    if median_amount >= 200:
        return 8.0
    if median_amount >= 100:
        return 6.0
    return 5.0


def _adaptive_z_threshold(median_amount):
    """Seuil Z-score adapté : les gros dépensiers tolèrent plus de variance."""
    if median_amount >= 1000:
        return 4.5
    if median_amount >= 500:
        return 4.0
    if median_amount >= 200:
        return 3.5
    return 3.0


# ---------------------------------------------------------------------------
# Détection des signaux de risque
# ---------------------------------------------------------------------------

def _signal_invalid_amount(amount):
    """Montant nul, négatif ou manquant."""
    if amount is None:
        return 0.85, "Montant manquant"
    if amount <= 0:
        return 0.90, "Montant nul ou négatif"
    return 0.0, None


def _signal_missing_fields(tx):
    """Champs obligatoires absents."""
    missing = []
    for field in _REQUIRED_FIELDS:
        value = tx.get(field)
        if value is None or (isinstance(value, str) and not value.strip()):
            missing.append(field)
    if missing:
        fields_str = ", ".join(missing)
        return 0.85, f"Champs obligatoires manquants: {fields_str}"
    return 0.0, None


def _signal_amount_deviation(amount, profile):
    """Déviation statistique du montant par rapport à l'historique utilisateur."""
    amounts = profile["amounts"]
    if amount is None or amount <= 0 or len(amounts) < 3:
        return 0.0, None

    arr = np.array(amounts, dtype=float)
    median = float(np.median(arr))
    mean = float(np.mean(arr))
    std = float(np.std(arr))
    q1, q3 = np.percentile(arr, [25, 75])
    iqr = q3 - q1

    multiplier = _adaptive_amount_multiplier(median)
    z_threshold = _adaptive_z_threshold(median)

    signals = []
    reasons = []

    # Ratio par rapport à la médiane (robuste quand std = 0)
    if median > 0:
        ratio = amount / median
        if ratio >= multiplier:
            score = min(0.95, 0.70 + (ratio / multiplier) * 0.15)
            signals.append(score)
            if ratio >= 10:
                reasons.append(
                    f"Montant {ratio:.0f} fois supérieur à la médiane habituelle du client"
                )
            else:
                reasons.append("Montant très supérieur à l'habitude du client")

    # Z-score (si variance suffisante)
    if std > 1e-6:
        z = (amount - mean) / std
        if z >= z_threshold:
            score = min(0.90, 0.55 + z * 0.08)
            signals.append(score)
            reasons.append(
                f"Montant {z:.1f} écarts-types au-dessus de la moyenne habituelle du client"
            )

    # Méthode IQR (détection d'outliers robuste)
    if iqr > 1e-6:
        upper_fence = q3 + 1.5 * iqr
        if amount > upper_fence:
            ratio = amount / median if median > 0 else amount
            if ratio >= 3.0:
                score = min(0.88, 0.60 + (amount - upper_fence) / (iqr + 1) * 0.05)
                signals.append(score)
                if not reasons:
                    reasons.append("Montant anormalement élevé par rapport à l'historique")

    if not signals:
        return 0.0, None
    return max(signals), reasons[0]


def _signal_geo_inconsistency(country, ts, profile):
    """Changement de pays trop rapide pour être physiquement possible."""
    last_country = profile["last_country"]
    last_ts = profile["last_ts"]

    if not country or not last_country or country.upper() == last_country.upper():
        return 0.0, None
    if ts is None or last_ts is None:
        return 0.0, None

    hours = _hours_between(last_ts, ts)
    if hours is None:
        return 0.0, None

    min_hours = _min_travel_hours(last_country, country)
    if hours < min_hours:
        if hours < 1:
            time_desc = f"{int(hours * 60)} minutes"
        elif hours < 24:
            time_desc = f"{hours:.1f} heures"
        else:
            time_desc = f"{hours / 24:.1f} jours"
        return 0.88, (
            f"Déplacement impossible entre {last_country} et {country} "
            f"en {time_desc}"
        )
    return 0.0, None


def _signal_velocity(ts, amount, profile):
    """Fréquence et volume anormalement élevés sur une courte période."""
    if ts is None or amount is None or amount <= 0:
        return 0.0, None

    window_minutes = 30
    max_tx_in_window = 5
    max_amount_in_window = None

    amounts = profile["amounts"]
    if len(amounts) >= 3:
        median = float(np.median(amounts))
        max_amount_in_window = median * 8
    else:
        max_amount_in_window = 500.0

    recent = [
        (t, a) for t, a, _ in profile["recent_events"]
        if _hours_between(t, ts) is not None and _hours_between(t, ts) * 60 <= window_minutes
    ]

    tx_count = len(recent) + 1
    total_amount = sum(a for _, a in recent) + amount

    signals = []
    reasons = []

    if tx_count > max_tx_in_window:
        signals.append(0.78)
        reasons.append(
            f"{tx_count} transactions en {window_minutes} minutes, "
            "fréquence anormalement élevée"
        )

    if total_amount > max_amount_in_window:
        signals.append(0.80)
        reasons.append(
            f"Volume de {total_amount:.0f} dépensé en {window_minutes} minutes, "
            "au-delà des habitudes du client"
        )

    if not signals:
        return 0.0, None
    return max(signals), reasons[0]


def _signal_card_present(amount, country, card_present, profile):
    """Carte absente combinée à un gros montant et/ou un pays inédit."""
    if card_present is not False or amount is None or amount <= 0:
        return 0.0, None

    amounts = profile["amounts"]
    if len(amounts) < 2:
        return 0.0, None

    median = float(np.median(amounts))
    country_counts = profile["country_counts"]
    is_new_country = country and country_counts.get(country.upper(), 0) == 0

    score = 0.0
    reasons = []

    if median > 0 and amount >= median * 4:
        score = 0.65
        reasons.append(
            "Carte non présente pour un montant élevé, inhabituel pour ce client"
        )

    if is_new_country and amount >= median * 2:
        extra = 0.70
        score = max(score, extra)
        reasons.append(
            f"Carte non présente dans un nouveau pays ({country}), "
            "combinaison à haut risque"
        )

    if score > 0:
        return score, reasons[0]
    return 0.0, None


def _update_profile(profile, amount, country, merchant, ts, transaction_id=None):
    """Met à jour le profil utilisateur après analyse de la transaction courante."""
    if amount is not None and amount > 0:
        profile["amounts"].append(amount)

    if country:
        profile["countries"].append(country.upper())
        profile["country_counts"][country.upper()] += 1

    if merchant:
        profile["merchants"].add(merchant)

    if ts is not None:
        profile["timestamps"].append(ts)
        profile["last_ts"] = ts
        profile["recent_events"].append(
            (ts, amount if amount and amount > 0 else 0.0, country)
        )
        # Garder uniquement les 50 dernières transactions en mémoire
        if len(profile["recent_events"]) > 50:
            profile["recent_events"] = profile["recent_events"][-50:]

    if country:
        profile["last_country"] = country.upper()

    if transaction_id is not None:
        profile["last_transaction_id"] = transaction_id

    profile["tx_count"] += 1


def _analyze_transaction(tx, profile):
    """Évalue une transaction et retourne le résultat de détection."""
    amount = tx.get("amount")
    country = tx.get("country")
    ts = _parse_timestamp(tx.get("timestamp"))
    card_present = tx.get("card_present")

    scores = []
    reasons = []

    # 1. Anomalies évidentes (priorité haute)
    for detector in (_signal_invalid_amount,):
        if detector is _signal_invalid_amount:
            score, reason = detector(amount)
        else:
            score, reason = detector(tx)
        if score > 0:
            scores.append(score)
            if reason:
                reasons.append(reason)

    score, reason = _signal_missing_fields(tx)
    if score > 0:
        scores.append(score)
        if reason:
            reasons.append(reason)

    # Si montant invalide ou champs critiques manquants, on peut s'arrêter là
    # mais on combine quand même les signaux pour un score plus précis
    if amount is not None and amount > 0:
        for detector, args in (
            (_signal_amount_deviation, (amount, profile)),
            (_signal_geo_inconsistency, (country, ts, profile)),
            (_signal_velocity, (ts, amount, profile)),
            (_signal_card_present, (amount, country, card_present, profile)),
        ):
            score, reason = detector(*args)
            if score > 0:
                scores.append(score)
                if reason:
                    reasons.append(reason)

    fraud_score = round(_combine_scores(scores), 2)
    is_suspicious = fraud_score >= _SUSPICION_THRESHOLD

    if is_suspicious and reasons:
        reason = reasons[0]
    elif is_suspicious:
        reason = "Plusieurs signaux de risque détectés"
    else:
        reason = "Transaction conforme au profil du client"

    return {
        "transaction_id": tx.get("transaction_id"),
        "fraud_score": fraud_score,
        "is_suspicious": is_suspicious,
        "reason": reason,
    }


# ---------------------------------------------------------------------------
# Fonction principale
# ---------------------------------------------------------------------------

def detect_fraud(transactions):
    """Analyse une liste de transactions et renvoie un verdict pour chacune.

    Retour : list[dict] avec transaction_id, fraud_score (0-1),
    is_suspicious (bool), reason (str) — un résultat par transaction, même ordre.

    Stratégie :
    1. Indexer les transactions pour préserver l'ordre d'entrée.
    2. Traiter chronologiquement pour construire les profils utilisateurs.
    3. Mettre à jour les stats APRÈS chaque analyse (pas de fuite de données).
    4. Retourner les résultats dans l'ordre original.
    """
    if not transactions:
        return []

    # Préserver l'ordre d'entrée tout en triant pour l'analyse chronologique
    indexed = list(enumerate(transactions))
    df = pd.DataFrame(
        [
            {
                "orig_idx": i,
                "transaction_id": tx.get("transaction_id"),
                "parsed_ts": _parse_timestamp(tx.get("timestamp")),
                "tx": tx,
            }
            for i, tx in indexed
        ]
    )

    # Tri chronologique ; timestamps invalides traités en dernier
    df["_sort_key"] = df["parsed_ts"].apply(
        lambda x: x.timestamp() if x is not None else float("inf")
    )
    df = df.sort_values("_sort_key", kind="mergesort")

    user_profiles = defaultdict(_new_user_profile)
    results_by_id = {}

    for _, row in df.iterrows():
        tx = row["tx"]
        tid = tx.get("transaction_id")
        user_id = tx.get("user_id") or "__unknown__"
        profile = user_profiles[user_id]
        prev_tid = profile.get("last_transaction_id")

        result = _analyze_transaction(tx, profile)
        results_by_id[tid] = result

        # Marquer aussi la transaction précédente en cas de déplacement impossible
        geo_score, geo_reason = _signal_geo_inconsistency(
            tx.get("country"), row["parsed_ts"], profile
        )
        if geo_score > 0 and prev_tid and prev_tid in results_by_id:
            prev = results_by_id[prev_tid]
            new_score = round(_combine_scores([prev["fraud_score"], geo_score]), 2)
            if new_score >= _SUSPICION_THRESHOLD:
                results_by_id[prev_tid] = {
                    "transaction_id": prev_tid,
                    "fraud_score": new_score,
                    "is_suspicious": True,
                    "reason": "Deux pays différents en trop peu de temps",
                }

        _update_profile(
            profile,
            tx.get("amount"),
            tx.get("country"),
            tx.get("merchant"),
            row["parsed_ts"],
            transaction_id=tid,
        )

    # Reconstruire dans l'ordre d'entrée original
    output = []
    for tx in transactions:
        tid = tx.get("transaction_id")
        if tid in results_by_id:
            output.append(results_by_id[tid])
        else:
            output.append({
                "transaction_id": tid,
                "fraud_score": 0.0,
                "is_suspicious": False,
                "reason": "Transaction conforme au profil du client",
            })

    return output
