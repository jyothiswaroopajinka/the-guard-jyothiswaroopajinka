"""
Test Case Coverage & Diversity Analysis

Shows how well the eval dataset covers different scenarios, channels,
categories, and edge cases across all 3 tasks.

Run: python analyze_coverage.py
"""

import json
from pathlib import Path
from collections import Counter

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box

console = Console()

DATA_DIR = Path(__file__).parent / "data"


def analyze_deal_copy():
    cases = json.loads((DATA_DIR / "deal_copy_cases.json").read_text())

    channels     = Counter(c["channel"] for c in cases)
    merchants    = Counter(c["merchant"] for c in cases)
    disc_types   = Counter(c["source_data"].get("type", "?") for c in cases)
    categories   = Counter(c.get("category", "?") for c in cases)
    edge_cases   = [c for c in cases if c.get("edge_case_type")]
    no_coupon    = [c for c in cases if not c["source_data"].get("coupon_code")]
    zero_min     = [c for c in cases if c["source_data"].get("min_order", 1) == 0]

    console.print(Panel(
        f"[bold]Total cases:[/] {len(cases)}\n"
        f"[bold]Unique merchants:[/] {len(merchants)}\n"
        f"[bold]Edge cases:[/] {len(edge_cases)}\n"
        f"[bold]No-coupon cases:[/] {len(no_coupon)}  [dim](tests model doesn't hallucinate a code)[/]\n"
        f"[bold]Zero min-order:[/] {len(zero_min)}  [dim](tests model doesn't invent a minimum)[/]",
        title="[bold cyan]Deal Copy — Coverage Summary[/]",
        border_style="cyan",
    ))

    # Channel distribution
    t = Table(title="Channel Distribution", box=box.SIMPLE)
    t.add_column("Channel"); t.add_column("Count", justify="right"); t.add_column("Char Limit")
    limits = {"email": "none", "whatsapp": "160", "push": "50 title + 100 body", "glance": "80"}
    for ch, count in sorted(channels.items(), key=lambda x: -x[1]):
        t.add_row(ch, str(count), limits.get(ch, "?"))
    console.print(t)

    # Discount type distribution
    t2 = Table(title="Discount Type Distribution", box=box.SIMPLE)
    t2.add_column("Type"); t2.add_column("Count", justify="right"); t2.add_column("What it tests")
    type_desc = {
        "flat":       "exact rupee/% amount grounding",
        "percentage": "percentage grounding",
        "cashback":   "cashback vs discount distinction",
        "bogo":       "buy-one-get-one logic",
    }
    for dtype, count in sorted(disc_types.items(), key=lambda x: -x[1]):
        t2.add_row(dtype, str(count), type_desc.get(dtype, ""))
    console.print(t2)

    # Edge cases
    if edge_cases:
        t3 = Table(title="Edge Cases", box=box.SIMPLE)
        t3.add_column("Case ID"); t3.add_column("Merchant"); t3.add_column("Channel"); t3.add_column("Edge Type")
        for c in edge_cases:
            t3.add_row(c["id"], c["merchant"], c["channel"], c.get("edge_case_type", ""))
        console.print(t3)

    console.print()


def analyze_insurance():
    cases = json.loads((DATA_DIR / "insurance_cases.json").read_text())

    labels      = Counter(c["ground_truth_label"] for c in cases)
    edge_cases  = [c for c in cases if c.get("edge_case")]
    categories  = Counter(c["deal_object"].get("category", "?") for c in cases)
    amounts     = [c["deal_object"].get("amount", 0) for c in cases]

    label_desc = {
        "travel_insurance":  "flight/hotel bookings above threshold",
        "health_insurance":  "prescription medicine, chronic conditions",
        "device_protection": "electronics above Rs.5000",
        "loan_protection":   "personal/business loans above threshold",
        "no_insurance":      "low value, grocery, fashion, borderline cases",
    }

    console.print(Panel(
        f"[bold]Total cases:[/] {len(cases)}\n"
        f"[bold]Labels covered:[/] {len(labels)} / 5\n"
        f"[bold]Edge cases:[/] {len(edge_cases)}  [dim](borderline decisions that test model reasoning)[/]\n"
        f"[bold]Amount range:[/] Rs.{min(amounts):,} – Rs.{max(amounts):,}",
        title="[bold cyan]Insurance — Coverage Summary[/]",
        border_style="cyan",
    ))

    # Label distribution
    t = Table(title="Label Distribution", box=box.SIMPLE)
    t.add_column("Label"); t.add_column("Count", justify="right"); t.add_column("What it represents")
    for label, count in sorted(labels.items(), key=lambda x: -x[1]):
        t.add_row(label, str(count), label_desc.get(label, ""))
    console.print(t)

    # Category distribution
    t2 = Table(title="Deal Category Distribution", box=box.SIMPLE)
    t2.add_column("Category"); t2.add_column("Count", justify="right")
    for cat, count in sorted(categories.items(), key=lambda x: -x[1]):
        t2.add_row(cat, str(count))
    console.print(t2)

    # Edge cases
    t3 = Table(title="Edge Cases — Borderline Decisions", box=box.SIMPLE)
    t3.add_column("Case ID"); t3.add_column("Label"); t3.add_column("Edge Type"); t3.add_column("Why it's hard")
    edge_why = {
        "high_value_grocery_still_no_insurance": "High spend but category never insured",
        "cheap_domestic_flight_no_insurance":    "Travel but below value threshold",
        "otc_medicine_no_chronic":               "Health category but OTC, no chronic flag",
        "refurbished_device":                    "Electronics but refurbished — still protectable?",
        "travel_electronics_combo":              "Both travel and device signals present",
        "micro_loan_no_protection":              "Loan but too small for protection",
    }
    for c in edge_cases:
        etype = c.get("edge_case_type", "")
        t3.add_row(c["id"], c["ground_truth_label"], etype, edge_why.get(etype, ""))
    console.print(t3)
    console.print()


def analyze_credit():
    cases = json.loads((DATA_DIR / "credit_cases.json").read_text())

    eligible    = Counter(c["label"]["eligible"] for c in cases)
    limits      = [c["label"]["credit_limit_inr"] for c in cases]
    gmv_growth  = [c["persona"]["gmv_growth_pct"] for c in cases]
    returns     = [c["persona"]["returns_rate_pct"] for c in cases]
    payments    = Counter(c["persona"]["payment_method"] for c in cases)
    edge_cases  = [c for c in cases if c.get("edge_case_type")]

    neg_growth   = [c for c in cases if c["persona"]["gmv_growth_pct"] < 0]
    high_returns = [c for c in cases if c["persona"]["returns_rate_pct"] > 20]
    new_users    = [c for c in cases if c["persona"]["account_age_months"] < 6]

    console.print(Panel(
        f"[bold]Total cases:[/] {len(cases)}\n"
        f"[bold]Eligible / Ineligible:[/] {eligible[True]} / {eligible[False]}\n"
        f"[bold]Credit limit range:[/] Rs.{min(limits):,} – Rs.{max(limits):,}\n"
        f"[bold]GMV growth range:[/] {min(gmv_growth):.1f}% – {max(gmv_growth):.1f}%\n"
        f"[bold]Negative growth cases:[/] {len(neg_growth)}  [dim](tests: high absolute GMV can override negative growth)[/]\n"
        f"[bold]High returns rate (>20%):[/] {len(high_returns)}  [dim](tests: penalise return abuse)[/]\n"
        f"[bold]New users (<6 months):[/] {len(new_users)}  [dim](tests: insufficient history → ineligible)[/]\n"
        f"[bold]Edge cases:[/] {len(edge_cases)}",
        title="[bold cyan]Credit Narrative — Coverage Summary[/]",
        border_style="cyan",
    ))

    # Eligibility
    t = Table(title="Eligibility Distribution", box=box.SIMPLE)
    t.add_column("Eligible"); t.add_column("Count", justify="right"); t.add_column("Credit Limit Range")
    elig_limits  = [c["label"]["credit_limit_inr"] for c in cases if c["label"]["eligible"]]
    inelig_cases = [c for c in cases if not c["label"]["eligible"]]
    t.add_row("Yes", str(eligible[True]),  f"Rs.{min(elig_limits):,} – Rs.{max(elig_limits):,}")
    t.add_row("No",  str(eligible[False]), "Rs.0 (blocked)")
    console.print(t)

    # Payment methods
    t2 = Table(title="Payment Method Distribution", box=box.SIMPLE)
    t2.add_column("Method"); t2.add_column("Count", justify="right")
    for method, count in sorted(payments.items(), key=lambda x: -x[1]):
        t2.add_row(method, str(count))
    console.print(t2)

    # Edge cases
    if edge_cases:
        t3 = Table(title="Edge Cases — Complex Credit Decisions", box=box.SIMPLE)
        t3.add_column("Case ID"); t3.add_column("Eligible"); t3.add_column("Edge Type"); t3.add_column("Why it's hard")
        edge_why = {
            "high_gmv_but_high_returns":    "Strong spend but 45% returns — return abuse risk",
            "zero_history_new_user":        "No purchase history at all — can't assess",
            "old_account_low_gmv":          "Loyal but low-spend — modest limit justified",
            "negative_growth_still_eligible": "GMV fell 21% but absolute spend still Rs.3L",
        }
        for c in edge_cases:
            etype = c.get("edge_case_type", "")
            t3.add_row(
                c["id"],
                "Yes" if c["label"]["eligible"] else "No",
                etype,
                edge_why.get(etype, ""),
            )
        console.print(t3)
    console.print()


def print_overall_summary():
    dc  = json.loads((DATA_DIR / "deal_copy_cases.json").read_text())
    ins = json.loads((DATA_DIR / "insurance_cases.json").read_text())
    cr  = json.loads((DATA_DIR / "credit_cases.json").read_text())

    total = len(dc) + len(ins) + len(cr)
    dc_edge  = len([c for c in dc  if c.get("edge_case_type")])
    ins_edge = len([c for c in ins if c.get("edge_case")])
    cr_edge  = len([c for c in cr  if c.get("edge_case_type")])
    total_edge = dc_edge + ins_edge + cr_edge

    console.print(Panel(
        f"[bold]Total test cases:[/] {total}  (deal_copy={len(dc)}, insurance={len(ins)}, credit={len(cr)})\n"
        f"[bold]Total edge cases:[/] {total_edge}  ({dc_edge} deal_copy + {ins_edge} insurance + {cr_edge} credit)\n\n"
        f"[bold]Coverage dimensions:[/]\n"
        f"  Deal Copy  — 4 channels × 4 discount types × 26 merchants × edge cases\n"
        f"  Insurance  — 5 labels × 8 categories × borderline-value cases\n"
        f"  Credit     — eligible/ineligible × growth/decline × return abuse × new users\n\n"
        f"[bold]Why this coverage matters:[/]\n"
        f"  • Edge cases reveal if a model understands the RULE, not just common patterns\n"
        f"  • Channel diversity catches format regressions specific to one channel\n"
        f"  • Negative/borderline cases test model reasoning, not just recall",
        title="[bold green]Overall Dataset Coverage[/]",
        border_style="green",
    ))


if __name__ == "__main__":
    console.print("\n[bold]The Guard — Test Case Coverage Analysis[/]\n")
    print_overall_summary()
    analyze_deal_copy()
    analyze_insurance()
    analyze_credit()
