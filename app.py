"""
Interface Streamlit — À CRÉER PAR VOUS pour le jury.

Le jury lancera :  streamlit run app.py

Règles :
  - Ne modifiez pas l'appel à detect_fraud / load_transactions (contrat technique).
  - Personnalisez render_interface() : clarté, intuitivité, compréhension pour un public non technique.
  - L'interface n'est PAS notée par la CI ; elle sert au jury pour repêcher et comparer les candidats.
"""

from __future__ import annotations

from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

from fraud_detection import detect_fraud, load_transactions

SAMPLE_CSV = Path(__file__).parent / "data" / "sample_transactions.csv"

SUSPICION_THRESHOLD = 0.50

# Palette fintech sombre
_COLOR_BG = "#000000"
_COLOR_SURFACE = "#0A1628"
_COLOR_SURFACE_ALT = "#0D1B2E"
_COLOR_BLUE = "#0066FF"
_COLOR_BLUE_LIGHT = "#1E90FF"
_COLOR_TEXT = "#E8EDF5"
_COLOR_TEXT_MUTED = "#8BA3C7"
_COLOR_SAFE = "#00C896"
_COLOR_ALERT = "#FF4D6A"
_COLOR_WARN = "#FFB020"

# ---------------------------------------------------------------------------
# Styles & helpers
# ---------------------------------------------------------------------------

_CUSTOM_CSS = f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    .stApp {{
        background-color: {_COLOR_BG};
        color: {_COLOR_TEXT};
        font-family: 'Inter', sans-serif;
    }}

    .block-container {{
        padding-top: 1.25rem;
        max-width: 1200px;
    }}

    [data-testid="stSidebar"] {{
        background-color: {_COLOR_SURFACE};
        border-right: 1px solid rgba(0, 102, 255, 0.25);
    }}
    [data-testid="stSidebar"] * {{
        color: {_COLOR_TEXT} !important;
    }}

    h1, h2, h3, h4, p, label, span {{
        color: {_COLOR_TEXT};
    }}

    .hero-banner {{
        background: linear-gradient(145deg, {_COLOR_SURFACE} 0%, #001433 55%, {_COLOR_BG} 100%);
        border: 1px solid rgba(0, 102, 255, 0.35);
        border-radius: 16px;
        padding: 2rem 2.2rem;
        margin-bottom: 1.25rem;
        color: {_COLOR_TEXT};
        box-shadow: 0 12px 40px rgba(0, 102, 255, 0.12);
    }}
    .hero-banner h1 {{
        font-family: 'Inter', sans-serif;
        font-size: 2rem;
        font-weight: 700;
        margin: 0 0 0.6rem 0;
        letter-spacing: -0.02em;
        color: #FFFFFF;
    }}
    .hero-banner p {{
        font-family: 'Inter', sans-serif;
        font-size: 1.02rem;
        margin: 0;
        color: {_COLOR_TEXT_MUTED};
        line-height: 1.6;
    }}
    .hero-badge {{
        display: inline-block;
        background: rgba(0, 102, 255, 0.18);
        border: 1px solid rgba(30, 144, 255, 0.45);
        border-radius: 999px;
        padding: 0.3rem 0.9rem;
        font-size: 0.78rem;
        margin-bottom: 0.9rem;
        font-weight: 600;
        color: {_COLOR_BLUE_LIGHT};
    }}

    .data-source-section {{
        background: {_COLOR_SURFACE};
        border: 1px solid rgba(0, 102, 255, 0.3);
        border-radius: 16px;
        padding: 1.75rem 2rem 2rem 2rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.45);
    }}
    .data-source-title {{
        font-size: 1.35rem;
        font-weight: 700;
        color: #FFFFFF;
        margin: 0 0 0.35rem 0;
        text-align: center;
    }}
    .data-source-subtitle {{
        font-size: 0.95rem;
        color: {_COLOR_TEXT_MUTED};
        text-align: center;
        margin: 0 0 1.5rem 0;
    }}

    .upload-zone-label {{
        text-align: center;
        font-size: 1.1rem;
        font-weight: 600;
        color: #FFFFFF;
        margin: 0.5rem 0 1rem 0;
        padding: 0.75rem 1rem;
        background: rgba(0, 102, 255, 0.1);
        border: 2px dashed rgba(30, 144, 255, 0.55);
        border-radius: 12px;
    }}

    div[data-testid="stFileUploader"] {{
        background: {_COLOR_SURFACE_ALT};
        border: 2px dashed {_COLOR_BLUE};
        border-radius: 14px;
        padding: 2.5rem 1.5rem;
        min-height: 180px;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        transition: border-color 0.2s, box-shadow 0.2s;
    }}
    div[data-testid="stFileUploader"]:hover {{
        border-color: {_COLOR_BLUE_LIGHT};
        box-shadow: 0 0 24px rgba(0, 102, 255, 0.2);
    }}
    div[data-testid="stFileUploader"] section {{
        padding: 0 !important;
    }}
    div[data-testid="stFileUploader"] button {{
        background: linear-gradient(135deg, {_COLOR_BLUE} 0%, {_COLOR_BLUE_LIGHT} 100%) !important;
        color: #FFFFFF !important;
        border: none !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        padding: 0.55rem 1.4rem !important;
    }}
    div[data-testid="stFileUploader"] small {{
        color: {_COLOR_TEXT_MUTED} !important;
    }}

    .sample-card {{
        background: {_COLOR_SURFACE_ALT};
        border: 1px solid rgba(0, 102, 255, 0.25);
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
    }}
    .sample-card p {{
        color: {_COLOR_TEXT_MUTED};
        margin: 0 0 0.5rem 0;
        line-height: 1.55;
    }}
    .sample-card strong {{
        color: {_COLOR_BLUE_LIGHT};
    }}

    .cta-row {{
        display: flex;
        justify-content: center;
        margin-top: 1.25rem;
    }}

    .kpi-card {{
        background: {_COLOR_SURFACE};
        border-radius: 12px;
        padding: 1.1rem 1.25rem;
        border: 1px solid rgba(0, 102, 255, 0.2);
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.35);
        height: 100%;
    }}
    .kpi-card.safe {{ border-left: 4px solid {_COLOR_SAFE}; }}
    .kpi-card.alert {{ border-left: 4px solid {_COLOR_ALERT}; }}
    .kpi-card.neutral {{ border-left: 4px solid {_COLOR_BLUE}; }}
    .kpi-card.rate {{ border-left: 4px solid {_COLOR_WARN}; }}
    .kpi-label {{
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: {_COLOR_TEXT_MUTED};
        font-weight: 600;
        margin-bottom: 0.35rem;
    }}
    .kpi-value {{
        font-size: 2rem;
        font-weight: 700;
        color: #FFFFFF;
        line-height: 1.1;
    }}
    .kpi-sub {{
        font-size: 0.85rem;
        color: {_COLOR_TEXT_MUTED};
        margin-top: 0.25rem;
    }}

    .explain-box {{
        background: {_COLOR_SURFACE_ALT};
        border-radius: 12px;
        padding: 1rem 1.2rem;
        border: 1px solid rgba(0, 102, 255, 0.2);
        margin: 0.5rem 0 1rem 0;
        color: {_COLOR_TEXT};
    }}
    .explain-box strong {{ color: #FFFFFF; }}

    .alert-chip {{
        display: inline-block;
        background: rgba(255, 77, 106, 0.15);
        color: #FF8FA3;
        border: 1px solid rgba(255, 77, 106, 0.4);
        border-radius: 8px;
        padding: 0.15rem 0.55rem;
        font-size: 0.78rem;
        font-weight: 600;
    }}

    div[data-testid="stExpander"] {{
        border: 1px solid rgba(0, 102, 255, 0.25) !important;
        border-radius: 10px !important;
        background: {_COLOR_SURFACE_ALT} !important;
    }}
    div[data-testid="stExpander"] summary {{
        color: {_COLOR_TEXT} !important;
    }}

    div[data-testid="stRadio"] > div {{
        gap: 0.75rem;
    }}
    div[data-testid="stRadio"] label {{
        background: {_COLOR_SURFACE_ALT};
        border: 1px solid rgba(0, 102, 255, 0.25);
        border-radius: 10px;
        padding: 0.6rem 1rem;
    }}

    .stButton > button[kind="primary"] {{
        background: linear-gradient(135deg, {_COLOR_BLUE} 0%, {_COLOR_BLUE_LIGHT} 100%) !important;
        color: #FFFFFF !important;
        border: none !important;
        font-weight: 700 !important;
        font-size: 1.05rem !important;
        padding: 0.65rem 2.5rem !important;
        border-radius: 10px !important;
        box-shadow: 0 4px 20px rgba(0, 102, 255, 0.35) !important;
    }}
    .stButton > button[kind="primary"]:hover {{
        box-shadow: 0 6px 28px rgba(30, 144, 255, 0.45) !important;
    }}

    hr {{
        border-color: rgba(0, 102, 255, 0.2) !important;
    }}

    [data-testid="stMetricValue"] {{
        color: #FFFFFF !important;
    }}
    [data-testid="stMetricLabel"] {{
        color: {_COLOR_TEXT_MUTED} !important;
    }}
</style>
"""


def _build_dataframe(transactions: list[dict], results: list[dict]) -> pd.DataFrame:
    """Fusionne transactions et résultats de détection."""
    tx_df = pd.DataFrame(transactions)
    res_df = pd.DataFrame(results)
    if "transaction_id" in tx_df.columns and "transaction_id" in res_df.columns:
        df = tx_df.merge(res_df, on="transaction_id", how="left")
    else:
        df = pd.concat([tx_df, res_df], axis=1)
    df["is_suspicious"] = df["is_suspicious"].fillna(False).astype(bool)
    df["fraud_score"] = pd.to_numeric(df["fraud_score"], errors="coerce").fillna(0.0)
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
    df["parsed_ts"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
    df["statut"] = df["is_suspicious"].map({True: "🚨 Suspecte", False: "✅ Conforme"})
    df["statut_label"] = df["is_suspicious"].map({True: "Suspecte", False: "Conforme"})
    return df


def _risk_level(score: float) -> str:
    if score >= 0.75:
        return "Critique"
    if score >= SUSPICION_THRESHOLD:
        return "Élevé"
    if score >= 0.25:
        return "Modéré"
    return "Faible"


def _chart_theme() -> alt.theme.ThemeConfig:
    return {
        "background": _COLOR_SURFACE,
        "view": {"stroke": "transparent"},
        "title": {"color": _COLOR_TEXT, "font": "Inter"},
        "axis": {
            "labelColor": _COLOR_TEXT_MUTED,
            "titleColor": _COLOR_TEXT,
            "gridColor": "rgba(0, 102, 255, 0.15)",
            "domainColor": "rgba(0, 102, 255, 0.3)",
        },
        "legend": {
            "labelColor": _COLOR_TEXT,
            "titleColor": _COLOR_TEXT,
        },
    }


def _style_dataframe(df: pd.DataFrame) -> pd.io.formats.style.Styler:
    display_cols = [
        c
        for c in [
            "transaction_id",
            "timestamp",
            "user_id",
            "amount",
            "currency",
            "merchant",
            "country",
            "card_present",
            "fraud_score",
            "statut",
            "reason",
        ]
        if c in df.columns
    ]

    def _row_style(row: pd.Series) -> list[str]:
        if row.get("is_suspicious"):
            bg, fg = "rgba(255, 77, 106, 0.18)", "#FFE0E6"
        else:
            bg, fg = "rgba(0, 200, 150, 0.12)", "#D4FFF2"
        return [f"background-color: {bg}; color: {fg};" for _ in display_cols]

    styled = df[display_cols].style.apply(_row_style, axis=1)
    styled = styled.format({"fraud_score": "{:.2f}", "amount": "{:.2f}"}, na_rep="—")
    return styled


def _render_kpi_cards(df: pd.DataFrame) -> None:
    total = len(df)
    suspicious = int(df["is_suspicious"].sum())
    safe = total - suspicious
    rate = (suspicious / total * 100) if total else 0.0
    avg_score = df["fraud_score"].mean() if total else 0.0

    c1, c2, c3, c4 = st.columns(4)
    cards = [
        (c1, "neutral", "Transactions analysées", str(total), "Lot complet importé"),
        (c2, "alert", "Alertes détectées", str(suspicious), "À vérifier en priorité"),
        (c3, "safe", "Transactions conformes", str(safe), "Profil client cohérent"),
        (c4, "rate", "Taux de fraude", f"{rate:.1f} %", f"Score moyen : {avg_score:.2f}"),
    ]
    for col, css_class, label, value, sub in cards:
        with col:
            st.markdown(
                f"""
                <div class="kpi-card {css_class}">
                    <div class="kpi-label">{label}</div>
                    <div class="kpi-value">{value}</div>
                    <div class="kpi-sub">{sub}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def _render_charts(df: pd.DataFrame) -> None:
    st.subheader("📊 Visualisations interactives")

    col_pie, col_hist = st.columns(2)

    with col_pie:
        st.markdown("**Répartition conforme / suspecte**")
        pie_df = (
            df["statut_label"]
            .value_counts()
            .reset_index()
            .rename(columns={"statut_label": "Statut", "count": "Nombre"})
        )
        if pie_df.empty:
            pie_df = pd.DataFrame({"Statut": ["Conforme", "Suspecte"], "Nombre": [0, 0]})
        pie_chart = (
            alt.Chart(pie_df)
            .mark_arc(innerRadius=50)
            .encode(
                theta=alt.Theta("Nombre:Q"),
                color=alt.Color(
                    "Statut:N",
                    scale=alt.Scale(
                        domain=["Conforme", "Suspecte"],
                        range=[_COLOR_SAFE, _COLOR_ALERT],
                    ),
                    legend=alt.Legend(title="Statut"),
                ),
                tooltip=["Statut", "Nombre"],
            )
            .properties(height=280)
            .configure_view(strokeWidth=0)
            .configure(**_chart_theme())
        )
        st.altair_chart(pie_chart, use_container_width=True)

    with col_hist:
        st.markdown("**Distribution des scores de risque**")
        hist_df = df.copy()
        hist_df["tranche"] = pd.cut(
            hist_df["fraud_score"],
            bins=[0, 0.25, 0.5, 0.75, 1.0],
            labels=["0–25 %", "25–50 %", "50–75 %", "75–100 %"],
            include_lowest=True,
        )
        hist_counts = (
            hist_df["tranche"]
            .value_counts()
            .sort_index()
            .reset_index()
            .rename(columns={"tranche": "Tranche", "count": "Transactions"})
        )
        hist_chart = (
            alt.Chart(hist_counts)
            .mark_bar(color=_COLOR_BLUE, cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
            .encode(
                x=alt.X("Tranche:N", title="Score de fraude"),
                y=alt.Y("Transactions:Q", title="Nombre"),
                tooltip=["Tranche", "Transactions"],
            )
            .properties(height=280)
            .configure(**_chart_theme())
        )
        rule = (
            alt.Chart(pd.DataFrame({"x": ["25–50 %"]}))
            .mark_rule(color=_COLOR_ALERT, strokeDash=[6, 4], strokeWidth=2)
            .encode(x="x:N")
        )
        st.altair_chart(hist_chart + rule, use_container_width=True)
        st.caption(f"Ligne rouge : zone du seuil d'alerte ({SUSPICION_THRESHOLD:.0%}).")

    col_amount, col_country = st.columns(2)

    with col_amount:
        st.markdown("**Montants par statut**")
        amt_df = df.dropna(subset=["amount"]).copy()
        if not amt_df.empty:
            amt_chart = (
                alt.Chart(amt_df)
                .mark_circle(size=90, opacity=0.75)
                .encode(
                    x=alt.X("amount:Q", title="Montant", scale=alt.Scale(zero=False)),
                    y=alt.Y("fraud_score:Q", title="Score de fraude"),
                    color=alt.Color(
                        "statut_label:N",
                        scale=alt.Scale(
                            domain=["Conforme", "Suspecte"],
                            range=[_COLOR_SAFE, _COLOR_ALERT],
                        ),
                        legend=alt.Legend(title="Statut"),
                    ),
                    tooltip=[
                        "transaction_id",
                        "amount",
                        "fraud_score",
                        "statut_label",
                        "reason",
                    ],
                )
                .properties(height=260)
                .configure(**_chart_theme())
            )
            st.altair_chart(amt_chart, use_container_width=True)
        else:
            st.caption("Montants non disponibles.")

    with col_country:
        st.markdown("**Transactions par pays**")
        if "country" in df.columns:
            country_df = (
                df.assign(country=df["country"].fillna("Inconnu"))
                .groupby(["country", "statut_label"])
                .size()
                .reset_index(name="nombre")
            )
            if not country_df.empty:
                country_chart = (
                    alt.Chart(country_df)
                    .mark_bar(cornerRadiusEnd=3)
                    .encode(
                        y=alt.Y("country:N", sort="-x", title="Pays"),
                        x=alt.X("nombre:Q", title="Transactions"),
                        color=alt.Color(
                            "statut_label:N",
                            scale=alt.Scale(
                                domain=["Conforme", "Suspecte"],
                                range=[_COLOR_SAFE, _COLOR_ALERT],
                            ),
                            legend=alt.Legend(title="Statut"),
                        ),
                        tooltip=["country", "statut_label", "nombre"],
                    )
                    .properties(height=260)
                    .configure(**_chart_theme())
                )
                st.altair_chart(country_chart, use_container_width=True)
            else:
                st.caption("Aucun pays renseigné.")


def _render_timeline(df: pd.DataFrame) -> None:
    timeline_df = df.dropna(subset=["parsed_ts"]).sort_values("parsed_ts")
    if timeline_df.empty:
        return

    st.subheader("🕐 Chronologie des transactions")
    st.caption(
        "Chaque point représente une transaction dans le temps. "
        "Les alertes apparaissent en rouge pour repérer les pics de risque."
    )

    timeline_chart = (
        alt.Chart(timeline_df)
        .mark_circle(size=100)
        .encode(
            x=alt.X("parsed_ts:T", title="Date"),
            y=alt.Y("amount:Q", title="Montant"),
            color=alt.Color(
                "statut_label:N",
                scale=alt.Scale(
                    domain=["Conforme", "Suspecte"],
                    range=[_COLOR_SAFE, _COLOR_ALERT],
                ),
                legend=alt.Legend(title="Statut"),
            ),
            shape=alt.Shape(
                "statut_label:N",
                scale=alt.Scale(
                    domain=["Conforme", "Suspecte"],
                    range=["circle", "diamond"],
                ),
            ),
            tooltip=[
                "transaction_id",
                "parsed_ts:T",
                "amount",
                "country",
                "fraud_score",
                "reason",
            ],
        )
        .properties(height=300)
        .configure(**_chart_theme())
    )
    st.altair_chart(timeline_chart, use_container_width=True)


def _render_suspicious_details(df: pd.DataFrame) -> None:
    suspicious_df = df[df["is_suspicious"]].sort_values("fraud_score", ascending=False)
    if suspicious_df.empty:
        st.success("Aucune transaction suspecte détectée sur ce lot — excellent signal de confiance.")
        return

    st.subheader(f"🔍 Détail des alertes ({len(suspicious_df)})")
    st.markdown(
        '<div class="explain-box">'
        "<strong>Comment lire une alerte ?</strong> Chaque transaction suspecte cumule des "
        "<em>signaux de risque</em> : montant inhabituel, déplacement géographique impossible, "
        "fréquence anormale, carte absente, ou données manquantes. Le score (0 à 100 %) "
        f"combine ces signaux ; au-delà de <strong>{int(SUSPICION_THRESHOLD * 100)} %</strong>, "
        "la transaction est signalée."
        "</div>",
        unsafe_allow_html=True,
    )

    for _, row in suspicious_df.iterrows():
        tid = row.get("transaction_id", "—")
        score = float(row.get("fraud_score", 0))
        reason = row.get("reason", "Raison non précisée")
        amount = row.get("amount")
        country = row.get("country") or "—"
        user = row.get("user_id") or "—"
        ts = row.get("timestamp") or "—"
        merchant = row.get("merchant") or "—"
        level = _risk_level(score)

        amount_str = f"{amount:,.2f}" if pd.notna(amount) else "—"
        title = f"{tid} · Score {score:.0%} · Niveau {level}"

        with st.expander(title, expanded=score >= 0.75):
            c1, c2, c3 = st.columns(3)
            c1.metric("Client", user)
            c2.metric("Montant", amount_str)
            c3.metric("Pays", country)

            st.markdown(f"**Commerçant :** {merchant}")
            st.markdown(f"**Horodatage :** {ts}")
            st.markdown(
                f'<span class="alert-chip">ALERTE</span> &nbsp; **Pourquoi ?** {reason}',
                unsafe_allow_html=True,
            )

            st.progress(min(max(score, 0.0), 1.0), text=f"Indice de risque : {score:.0%}")

            tips = []
            reason_lower = reason.lower()
            if "montant" in reason_lower or "médiane" in reason_lower:
                tips.append("Vérifiez si le client a réellement autorisé ce débit inhabituel.")
            if "pays" in reason_lower or "déplacement" in reason_lower:
                tips.append("Contactez le client : un voyage ou une utilisation à l'étranger est-elle plausible ?")
            if "fréquence" in reason_lower or "minutes" in reason_lower:
                tips.append("Bloquez temporairement la carte et demandez une confirmation SMS.")
            if "carte" in reason_lower or "présente" in reason_lower:
                tips.append("Transaction à distance à haut risque : exiger une authentification forte (3-D Secure).")
            if "manquant" in reason_lower or "négatif" in reason_lower:
                tips.append("Données incohérentes : rejeter la transaction et alerter l'équipe conformité.")

            if tips:
                st.markdown("**Actions recommandées :**")
                for tip in tips:
                    st.markdown(f"- {tip}")


def render_interface(transactions: list[dict], results: list[dict]) -> None:
    """
    Interface intuitive pour le jury / le public — visualisation complète des résultats.
    """
    st.markdown(_CUSTOM_CSS, unsafe_allow_html=True)

    df = _build_dataframe(transactions, results)

    st.markdown(
        """
        <div class="hero-banner">
            <div class="hero-badge">🛡️ INTELO2026 · Sécurité financière par l'IA</div>
            <h1>Détecteur de fraude en temps réel</h1>
            <p>
                Cet outil analyse chaque transaction bancaire et compare le comportement du client
                à son historique : montants, pays, fréquence et mode de paiement. Les opérations
                atypiques sont signalées <strong>avant</strong> qu'elles ne causent un préjudice —
                sans bloquer inutilement les clients honnêtes.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.divider()
        st.header("⚙️ Paramètres d'analyse")
        st.markdown(
            f"**Seuil d'alerte :** `{SUSPICION_THRESHOLD:.0%}`\n\n"
            "Une transaction est marquée suspecte lorsque son score combiné "
            "dépasse ce seuil. Plus le score est élevé, plus le risque est critique."
        )
        st.markdown("**Signaux surveillés :**")
        st.markdown(
            "- Montant inhabituel vs historique client\n"
            "- Déplacement géographique impossible\n"
            "- Rafale de transactions (vélocité)\n"
            "- Carte absente + montant/pays à risque\n"
            "- Données manquantes ou montant invalide"
        )
        st.divider()
        st.markdown(
            "**Jury :** évaluez l'ergonomie et la clarté de l'écran principal, "
            "pas seulement le score des tests."
        )

    _render_kpi_cards(df)

    st.divider()

    filter_mode = st.radio(
        "Filtrer les transactions",
        options=["Toutes", "Suspectes uniquement", "Conformes uniquement"],
        horizontal=True,
        help="Affinez la liste pour vous concentrer sur les alertes ou les cas sains.",
    )

    if filter_mode == "Suspectes uniquement":
        view_df = df[df["is_suspicious"]].copy()
    elif filter_mode == "Conformes uniquement":
        view_df = df[~df["is_suspicious"]].copy()
    else:
        view_df = df.copy()

    st.subheader("📋 Tableau des transactions")
    st.caption(
        f"{len(view_df)} transaction(s) affichée(s) · "
        "fond vert = conforme, fond rose = suspecte"
    )

    if view_df.empty:
        st.info("Aucune transaction ne correspond à ce filtre.")
    else:
        st.dataframe(_style_dataframe(view_df), use_container_width=True, hide_index=True)

    st.divider()
    _render_charts(df)
    _render_timeline(df)
    st.divider()
    _render_suspicious_details(df)


def main() -> None:
    st.set_page_config(
        page_title="Détection de fraude — Hackathon INTELO2026",
        page_icon="🛡️",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.markdown(_CUSTOM_CSS, unsafe_allow_html=True)

    st.markdown(
        """
        <div class="hero-banner">
            <div class="hero-badge">🛡️ INTELO2026 · Sécurité financière par l'IA</div>
            <h1>Détection de fraude financière</h1>
            <p>
                Analysez vos transactions bancaires en quelques clics.
                Importez un fichier CSV ou utilisez les données d'exemple, puis lancez l'analyse.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="data-source-section">
            <p class="data-source-title">📂 Étape 1 — Choisir vos données</p>
            <p class="data-source-subtitle">
                Importez votre fichier CSV ou utilisez le jeu de démonstration
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    tab_import, tab_sample = st.tabs(
        ["📁 Importer un fichier CSV", "📊 Données d'exemple"]
    )

    transactions: list[dict] = []
    data_ready = False

    with tab_import:
        st.markdown(
            """
            <p class="upload-zone-label">
                Glissez votre fichier CSV ici ou cliquez pour parcourir votre ordinateur
            </p>
            """,
            unsafe_allow_html=True,
        )
        uploaded = st.file_uploader(
            "Sélectionner un fichier CSV",
            type=["csv"],
            label_visibility="collapsed",
            key="main_csv_uploader",
        )
        if uploaded:
            tmp = Path(".streamlit_upload.csv")
            tmp.write_bytes(uploaded.getvalue())
            transactions = load_transactions(str(tmp))
            tmp.unlink(missing_ok=True)
            st.session_state["transactions"] = transactions
            st.session_state["data_source"] = "import"
            data_ready = True
            st.success(f"✅ **{len(transactions)}** transactions importées — **{uploaded.name}**")
        elif st.session_state.get("data_source") == "import" and st.session_state.get("transactions"):
            transactions = st.session_state["transactions"]
            data_ready = True
        else:
            st.warning("📎 Sélectionnez un fichier CSV pour lancer l'analyse.")

    with tab_sample:
        st.markdown(
            f"""
            <div class="sample-card">
                <p style="font-size: 1.05rem; color: #FFFFFF; margin-bottom: 0.75rem;">
                    <strong>📊 Données de démonstration</strong>
                </p>
                <p>
                    Transactions fictives prêtes à l'emploi — idéal pour découvrir
                    le détecteur sans préparer de fichier.
                </p>
                <p>Fichier source : <strong>{SAMPLE_CSV.name}</strong></p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button("Charger les données d'exemple", type="secondary", key="load_sample"):
            sample_tx = load_transactions(str(SAMPLE_CSV))
            st.session_state["transactions"] = sample_tx
            st.session_state["data_source"] = "sample"

        if st.session_state.get("data_source") == "sample" and st.session_state.get("transactions"):
            transactions = st.session_state["transactions"]
            data_ready = True
            st.success(f"✅ **{len(transactions)}** transactions d'exemple chargées")

    st.markdown("---")

    col_spacer, col_btn, col_spacer2 = st.columns([2, 1, 2])
    with col_btn:
        analyze_clicked = st.button("🔍 Analyser les transactions", type="primary", use_container_width=True)

    if not data_ready or not transactions:
        st.info("👆 Importez un fichier CSV ou activez les données d'exemple, puis cliquez sur **Analyser**.")
        return

    if analyze_clicked:
        try:
            results = detect_fraud(transactions)
        except NotImplementedError:
            st.error("Implémentez d'abord `detect_fraud` dans `fraud_detection.py`.")
            return
        except Exception as exc:
            st.error(f"Erreur : {exc}")
            return

        render_interface(transactions, results)


if __name__ == "__main__":
    main()
