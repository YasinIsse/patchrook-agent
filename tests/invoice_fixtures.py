"""Cross-module invoice fixture for bounded context and repair regressions."""

import json
from pathlib import Path
import shutil
from textwrap import dedent

from coding_fixtures import BenchmarkTask


FIXTURE_NAMES = ("ranking",)
RELEVANT_PATHS = ("adjustments.py", "promotions.py")
CORRECTIONS = {
    "adjustments.py": ("return subtotal + returned_total", "return subtotal - returned_total"),
    "promotions.py": ("(merchandise_cents * discount_percent + 99) // 100", "merchandise_cents * discount_percent // 100"),
}
def _source(value):
    return dedent(value).lstrip("\n")


_FILES = {
    "checkout.py": _source('''
        from adjustments import calculate_adjusted_merchandise
        from promotions import apply_percentage_discount
        from taxes import calculate_tax
        from shipping import shipping_charge
        from loyalty import calculate_points
        from receipts import receipt_lines
        from audit import invoice_audit

        # Checkout composes the domain helpers; each amount is integer cents.
        def build_invoice(unit_cents, quantity, returned_quantity=0, discount_percent=0,
                          tax_percent=0, destination="domestic", customer="guest"):
            """Calculate one settled invoice, including returns and a percentage offer.

            Domain helpers validate their own amounts and policy choices. The
            invoice retains the undiscounted amount for audit while shipping
            and loyalty eligibility use the discounted merchandise amount.
            Receipt generation is a presentation step and cannot alter money.
            """
            if not isinstance(customer, str) or not customer.strip():
                raise ValueError("customer must be a nonempty account label")
            if len(customer) > 80:
                raise ValueError("customer account labels must fit the receipt header")
            if customer != customer.strip():
                raise ValueError("customer must not include leading or trailing whitespace")
            if "\\n" in customer or "\\r" in customer:
                raise ValueError("customer account labels must occupy a single receipt line")
            if destination not in {"domestic", "international", "pickup"}:
                raise ValueError("destination must be domestic, international, or pickup")
            adjusted = calculate_adjusted_merchandise(unit_cents, quantity, returned_quantity)
            merchandise = apply_percentage_discount(adjusted, discount_percent)
            tax = calculate_tax(merchandise, tax_percent)
            delivery = shipping_charge(merchandise, destination)
            points = calculate_points(merchandise, account=customer)
            total = merchandise + tax + delivery
            invoice = {
                "customer": customer,
                "destination": destination,
                "unit_cents": unit_cents,
                "quantity": quantity,
                "returned_quantity": returned_quantity,
                "discount_percent": discount_percent,
                "tax_percent": tax_percent,
                "adjusted_cents": adjusted,
                "merchandise_cents": merchandise,
                "tax_cents": tax,
                "shipping_cents": delivery,
                "total_cents": total,
                "loyalty_points": points,
            }
            if any(type(invoice[key]) is not int for key in (
                    "adjusted_cents", "merchandise_cents", "tax_cents",
                    "shipping_cents", "total_cents", "loyalty_points")):
                raise TypeError("invoice helpers must preserve exact integer monetary accounting")
            if total < 0:
                raise ValueError("a settled invoice cannot have a negative amount due")
            invoice["receipt"] = receipt_lines(invoice)
            invoice["audit"] = invoice_audit(invoice)
            return invoice

        def checkout_reference(number):
            return f"CHECKOUT-{number:06d}"
    '''),
    "adjustments.py": _source('''
        # Returned units reduce merchandise before percentage offers are applied.
        def calculate_adjusted_merchandise(unit_cents, quantity, returned_quantity=0,
                                           return_unit_cents=None, reason="customer_return",
                                           return_reference=None):
            """Settle the line-item amount against the value of returned units.

            Returns normally use the original unit price. An explicit historical
            return price can be supplied for exchanges without changing the sale
            unit price. Neither adjustment may exceed the purchased merchandise.
            A return reference is metadata, never a multiplier or a currency value.
            """
            if type(unit_cents) is not int:
                raise TypeError("unit_cents must be an integer number of cents")
            if type(quantity) is not int:
                raise TypeError("quantity must be an integer number of purchased units")
            if type(returned_quantity) is not int:
                raise TypeError("returned_quantity must be an integer number of returned units")
            if unit_cents < 0:
                raise ValueError("unit_cents cannot be negative for a merchandise sale")
            if quantity < 0:
                raise ValueError("quantity cannot be negative; returns use returned_quantity")
            if returned_quantity < 0:
                raise ValueError("returned_quantity cannot be negative")
            if returned_quantity > quantity:
                raise ValueError("returned units cannot exceed the original purchased quantity")
            if return_unit_cents is None:
                return_unit_cents = unit_cents
            if type(return_unit_cents) is not int:
                raise TypeError("return_unit_cents must be integer cents when provided")
            if return_unit_cents < 0:
                raise ValueError("the historical return unit price cannot be negative")
            if not isinstance(reason, str) or not reason.strip():
                raise ValueError("a nonempty adjustment reason is required for an auditable return")
            if reason not in {"customer_return", "exchange", "damaged_item", "cancelled_item"}:
                raise ValueError("the return reason must use a supported adjustment category")
            if return_reference is not None:
                if not isinstance(return_reference, str) or not return_reference.strip():
                    raise ValueError("an optional return reference must be a nonempty string")
                if len(return_reference) > 80:
                    raise ValueError("return references cannot exceed eighty characters")
                if "\\n" in return_reference or "\\r" in return_reference:
                    raise ValueError("return references must occupy a single audit line")
            subtotal = unit_cents * quantity
            returned_total = return_unit_cents * returned_quantity
            if returned_total > subtotal:
                raise ValueError("the returned merchandise value cannot exceed the original subtotal")
            return subtotal + returned_total

        def adjustment_reference(number):
            return f"RETURN-{number:06d}"
    '''),
    "promotions.py": _source('''
        # Percentage offers discount the adjusted merchandise in integer cents.
        def apply_percentage_discount(merchandise_cents, discount_percent=0,
                                      eligibility="standard", minimum_cents=0,
                                      maximum_discount_cents=None, campaign=None):
            """Apply an eligible percentage offer with a downward cent rounding.

            The offer does not affect tax or shipping directly. Excluded products
            retain their merchandise value, and a campaign cap limits the amount
            removed rather than changing the percentage. Campaign names are audit
            metadata and must not contain receipt line breaks.
            """
            if type(merchandise_cents) is not int:
                raise TypeError("merchandise_cents must be an exact integer-cent amount")
            if type(discount_percent) is not int:
                raise TypeError("discount_percent must be an integer percentage")
            if type(minimum_cents) is not int:
                raise TypeError("the minimum eligible amount must use integer cents")
            if merchandise_cents < 0:
                raise ValueError("percentage offers require a nonnegative adjusted merchandise amount")
            if not 0 <= discount_percent <= 100:
                raise ValueError("the percentage offer must be between zero and one hundred")
            if minimum_cents < 0:
                raise ValueError("an offer minimum cannot be a negative merchandise amount")
            if eligibility not in {"standard", "member", "excluded"}:
                raise ValueError("offer eligibility must be standard, member, or excluded")
            if maximum_discount_cents is not None:
                if type(maximum_discount_cents) is not int:
                    raise TypeError("the optional discount cap must use integer cents")
                if maximum_discount_cents < 0:
                    raise ValueError("the optional discount cap cannot be negative")
            if campaign is not None:
                if not isinstance(campaign, str) or not campaign.strip():
                    raise ValueError("a campaign identifier must be a nonempty string when supplied")
                if len(campaign) > 60:
                    raise ValueError("campaign identifiers cannot exceed sixty characters")
                if "\\n" in campaign or "\\r" in campaign:
                    raise ValueError("campaign identifiers must occupy one audit line")
            if eligibility == "excluded" or merchandise_cents < minimum_cents:
                return merchandise_cents
            discount_cents = (merchandise_cents * discount_percent + 99) // 100
            if maximum_discount_cents is not None:
                discount_cents = min(discount_cents, maximum_discount_cents)
            if discount_cents > merchandise_cents:
                raise ValueError("a merchandise discount cannot exceed its eligible amount")
            return merchandise_cents - discount_cents

        def campaign_reference(name):
            return f"CAMPAIGN:{name}"
    '''),
    "taxes.py": _source('''
        # Tax is calculated after returns and merchandise discounts.
        def calculate_tax(merchandise_cents, tax_percent=0, jurisdiction="domestic",
                          exempt=False, certificate=None, minimum_taxable_cents=0):
            """Return integer tax cents according to the invoice's selected policy.

            Tax exemption requires an explicit policy choice. Certificates are
            recorded for validation only and never loaded from the filesystem.
            The tax threshold applies to the discounted merchandise amount and
            does not include shipping, reward points, or printed receipt text.
            """
            if type(merchandise_cents) is not int:
                raise TypeError("the taxable merchandise amount must be integer cents")
            if type(tax_percent) is not int:
                raise TypeError("tax_percent must be an integer percentage")
            if type(minimum_taxable_cents) is not int:
                raise TypeError("minimum_taxable_cents must use integer cents")
            if type(exempt) is not bool:
                raise TypeError("tax exemption must be supplied as a boolean policy choice")
            if merchandise_cents < 0:
                raise ValueError("a taxable merchandise amount cannot be negative")
            if not 0 <= tax_percent <= 100:
                raise ValueError("the tax percentage must be between zero and one hundred")
            if minimum_taxable_cents < 0:
                raise ValueError("the taxable merchandise minimum cannot be negative")
            if jurisdiction not in {"domestic", "export", "special"}:
                raise ValueError("jurisdiction must use a supported invoice tax policy")
            if certificate is not None:
                if not isinstance(certificate, str) or not certificate.strip():
                    raise ValueError("an exemption certificate must be a nonempty label")
                if len(certificate) > 80:
                    raise ValueError("an exemption certificate label cannot exceed eighty characters")
                if "\\n" in certificate or "\\r" in certificate:
                    raise ValueError("certificate labels must be printable on one audit line")
            if exempt:
                return 0
            if jurisdiction == "export":
                return 0
            if merchandise_cents < minimum_taxable_cents:
                return 0
            whole_tax_cents, fractional_remainder = divmod(merchandise_cents * tax_percent, 100)
            if fractional_remainder < 0 or fractional_remainder >= 100:
                raise ArithmeticError("tax remainder escaped the expected integer-cent range")
            if whole_tax_cents < 0:
                raise ArithmeticError("validated tax inputs produced a negative tax amount")
            return whole_tax_cents

        def tax_label(rate):
            return f"Tax {rate}%"
    '''),
    "shipping.py": _source('''
        # Shipping eligibility uses discounted merchandise, before tax.
        def shipping_charge(merchandise_cents, destination="domestic", service="standard",
                            free_threshold_cents=5000, standard_cents=500,
                            international_surcharge_cents=1200, fragile=False):
            """Calculate a shipping policy without touching merchandise or taxes.

            Domestic orders at the threshold ship free with standard service.
            International surcharges and express premiums remain payable even
            when the merchandise qualifies for a waived standard charge. Store
            pickup always has zero shipping and ignores delivery-only options.
            """
            if type(merchandise_cents) is not int:
                raise TypeError("shipping eligibility must use integer merchandise cents")
            if type(free_threshold_cents) is not int:
                raise TypeError("the free shipping threshold must use integer cents")
            if type(standard_cents) is not int:
                raise TypeError("the standard shipping charge must use integer cents")
            if type(international_surcharge_cents) is not int:
                raise TypeError("the international shipping surcharge must use integer cents")
            if type(fragile) is not bool:
                raise TypeError("fragile packaging must be an explicit boolean option")
            if merchandise_cents < 0:
                raise ValueError("shipping eligibility cannot use negative merchandise cents")
            if min(free_threshold_cents, standard_cents, international_surcharge_cents) < 0:
                raise ValueError("shipping charges and thresholds cannot be negative")
            if destination not in {"domestic", "international", "pickup"}:
                raise ValueError("destination must be domestic, international, or pickup")
            if service not in {"standard", "express"}:
                raise ValueError("delivery service must be standard or express")
            if destination == "pickup":
                return 0
            base = 0 if merchandise_cents >= free_threshold_cents else standard_cents
            surcharge = 0
            if destination == "international":
                surcharge += international_surcharge_cents
            if service == "express":
                surcharge += 900
            if fragile:
                surcharge += 200
            total = base + surcharge
            if total < 0:
                raise ArithmeticError("validated delivery options produced negative shipping")
            return total

        def shipping_label(destination):
            return destination.replace("_", " ").title()
    '''),
    "loyalty.py": _source('''
        # Reward points are informational and never reduce the invoice total.
        def calculate_points(merchandise_cents, account="guest", cents_per_point=100,
                             multiplier=1, points_cap=None, excluded=False):
            """Accrue whole loyalty points on the discounted merchandise amount.

            Guest checkouts do not accrue points. Multipliers apply after whole
            base points are calculated; the optional cap applies to the final
            reward. Accruing points does not redeem them and cannot alter tax,
            delivery policy, receipt amounts, or returned merchandise balances.
            """
            if type(merchandise_cents) is not int:
                raise TypeError("loyalty merchandise amounts must use integer cents")
            if type(cents_per_point) is not int:
                raise TypeError("the loyalty earning interval must use integer cents")
            if type(multiplier) is not int:
                raise TypeError("loyalty multipliers must be integer factors")
            if type(excluded) is not bool:
                raise TypeError("reward exclusion must be an explicit boolean")
            if merchandise_cents < 0:
                raise ValueError("loyalty points cannot accrue on negative merchandise amounts")
            if cents_per_point <= 0:
                raise ValueError("a loyalty earning interval must be greater than zero")
            if not 1 <= multiplier <= 10:
                raise ValueError("loyalty multipliers must be between one and ten")
            if not isinstance(account, str) or not account.strip():
                raise ValueError("a nonempty account label is required for loyalty accounting")
            if account != account.strip():
                raise ValueError("loyalty account labels must not have surrounding whitespace")
            if len(account) > 80:
                raise ValueError("loyalty account labels cannot exceed eighty characters")
            if "\\n" in account or "\\r" in account:
                raise ValueError("loyalty account labels must occupy one audit line")
            if points_cap is not None:
                if type(points_cap) is not int:
                    raise TypeError("an optional reward cap must be an integer number of points")
                if points_cap < 0:
                    raise ValueError("the reward cap cannot be negative")
            if account == "guest" or excluded:
                return 0
            base_points = merchandise_cents // cents_per_point
            earned_points = base_points * multiplier
            if points_cap is not None:
                earned_points = min(earned_points, points_cap)
            return earned_points

        def points_label(points):
            return f"{points} reward points"
    '''),
    "receipts.py": _source('''
        # Receipt formatting consumes settled amounts and does not recalculate them.
        def receipt_lines(invoice, show_rewards=True, title="Invoice", footer=None,
                          currency="USD", include_return_count=True):
            """Render a small receipt from a completed invoice dictionary.

            All amounts originate in the checkout calculation. Presentation may
            hide reward points or return quantities but cannot modify a payable
            amount. The supported currency is a display label, not a conversion
            instruction, and this function performs no file or network writes.
            """
            if not isinstance(invoice, dict):
                raise TypeError("receipt generation requires a completed invoice dictionary")
            if type(show_rewards) is not bool or type(include_return_count) is not bool:
                raise TypeError("receipt display switches must be boolean options")
            if not isinstance(title, str) or not title.strip():
                raise ValueError("a receipt must have a nonempty printable title")
            if len(title) > 80 or "\\n" in title or "\\r" in title:
                raise ValueError("receipt titles must occupy one line of at most eighty characters")
            if currency not in {"USD", "EUR", "GBP"}:
                raise ValueError("receipt currency must be a supported display label")
            required_amounts = ("merchandise_cents", "tax_cents", "shipping_cents", "total_cents")
            for key in required_amounts:
                if key not in invoice:
                    raise ValueError(f"the completed invoice is missing its {key} amount")
                if type(invoice[key]) is not int:
                    raise TypeError(f"the completed invoice {key} amount must be integer cents")
                if invoice[key] < 0:
                    raise ValueError(f"the completed invoice {key} amount cannot be negative")
            if footer is not None:
                if not isinstance(footer, str) or len(footer) > 160:
                    raise ValueError("an optional receipt footer must be text of at most 160 characters")
                if "\\n" in footer or "\\r" in footer:
                    raise ValueError("a receipt footer must occupy one printable line")
            lines = [title, f"Account: {invoice['customer']}"]
            for label, key in (("Merchandise", "merchandise_cents"), ("Tax", "tax_cents"),
                               ("Shipping", "shipping_cents"), ("Amount due", "total_cents")):
                cents = invoice[key]
                whole, fraction = divmod(cents, 100)
                lines.append(f"{label}: {currency} {whole}.{fraction:02d}")
            if include_return_count:
                lines.append(f"Returned units: {invoice['returned_quantity']}")
            if show_rewards:
                lines.append(f"Reward points: {invoice['loyalty_points']}")
            if footer is not None:
                lines.append(footer)
            return tuple(lines)

        def receipt_heading(customer):
            return f"Receipt for {customer}"
    '''),
    "audit.py": _source('''
        # Audit metadata is returned to the caller; no external messages are sent.
        def invoice_audit(invoice, source="checkout", status="settled", operator=None,
                          include_customer=True, tags=()):
            """Describe settled invoice values without changing their calculation.

            Audit consumers may use the returned immutable tuple for diagnostics.
            A status is descriptive and does not authorize charging a customer.
            Audit tags must be short printable labels. Monetary reconciliation
            checks the already calculated total rather than deriving new prices.
            """
            if not isinstance(invoice, dict):
                raise TypeError("invoice audit requires a completed invoice dictionary")
            if type(include_customer) is not bool:
                raise TypeError("customer visibility in an audit must be a boolean")
            if not isinstance(source, str) or not source.strip():
                raise ValueError("an audit source must be a nonempty component label")
            if len(source) > 60 or "\\n" in source or "\\r" in source:
                raise ValueError("an audit source must fit one sixty-character component label")
            if status not in {"settled", "quoted", "cancelled"}:
                raise ValueError("audit status must be settled, quoted, or cancelled")
            if operator is not None:
                if not isinstance(operator, str) or not operator.strip():
                    raise ValueError("an optional audit operator must be a nonempty account label")
                if len(operator) > 80 or "\\n" in operator or "\\r" in operator:
                    raise ValueError("an audit operator must fit one eighty-character label")
            if not isinstance(tags, (list, tuple)):
                raise TypeError("audit tags must be a sequence of short text labels")
            if len(tags) > 8:
                raise ValueError("an invoice audit cannot carry more than eight tags")
            normalized_tags = []
            for tag in tags:
                if not isinstance(tag, str) or not tag.strip() or len(tag) > 32:
                    raise ValueError("each audit tag must be a nonempty text label of at most 32 characters")
                if "\\n" in tag or "\\r" in tag:
                    raise ValueError("an audit tag must occupy one printable line")
                if tag not in normalized_tags:
                    normalized_tags.append(tag)
            values = {key: invoice[key] for key in (
                "merchandise_cents", "tax_cents", "shipping_cents", "total_cents")}
            if any(type(value) is not int or value < 0 for value in values.values()):
                raise TypeError("audit amounts must be nonnegative integer-cent values")
            if values["total_cents"] != sum(values[key] for key in (
                    "merchandise_cents", "tax_cents", "shipping_cents")):
                raise ValueError("the settled invoice amounts do not reconcile with the total")
            customer = invoice["customer"] if include_customer else "redacted"
            return (source, status, operator, customer, tuple(normalized_tags),
                    tuple(sorted(values.items())))

        def audit_reference(number):
            return f"AUDIT-{number:06d}"
    '''),
    "tests/test_checkout.py": _source('''
        from checkout import build_invoice

        def test_returned_units_reduce_merchandise_before_the_offer():
            invoice = build_invoice(1001, 3, returned_quantity=1, discount_percent=15)
            assert invoice["adjusted_cents"] == 2002
            assert invoice["merchandise_cents"] == 1702
            assert invoice["total_cents"] == 2202

        def test_fractional_discount_rounds_down_and_shipping_uses_discounted_amount():
            invoice = build_invoice(2501, 2, discount_percent=1)
            assert invoice["merchandise_cents"] == 4952
            assert invoice["shipping_cents"] == 500
            assert invoice["total_cents"] == 5452

        def test_unrelated_invoice_policies_are_preserved():
            invoice = build_invoice(3000, 2, tax_percent=10, customer="member-17")
            assert invoice["shipping_cents"] == 0
            assert invoice["tax_cents"] == 600
            assert invoice["loyalty_points"] == 60
            assert invoice["total_cents"] == 6600
            assert "Amount due: USD 66.00" in invoice["receipt"]
    '''),
}


_ACCEPTANCE = _source('''
    from adjustments import calculate_adjusted_merchandise, adjustment_reference
    from promotions import apply_percentage_discount, campaign_reference
    from checkout import build_invoice, checkout_reference
    from shipping import shipping_charge
    from loyalty import calculate_points
    from taxes import calculate_tax

    assert calculate_adjusted_merchandise(1001, 3, 1) == 2002
    assert calculate_adjusted_merchandise(900, 2, 2) == 0
    assert calculate_adjusted_merchandise(1000, 2, 1, 800) == 1200
    assert calculate_adjusted_merchandise(99, 0, 0) == 0
    assert apply_percentage_discount(2002, 15) == 1702
    assert apply_percentage_discount(101, 1) == 100
    assert apply_percentage_discount(99, 1) == 99
    assert apply_percentage_discount(101, 100) == 0
    assert apply_percentage_discount(101, 0) == 101
    assert apply_percentage_discount(2002, 15, maximum_discount_cents=100) == 1902
    assert apply_percentage_discount(2002, 15, eligibility="excluded") == 2002
    assert apply_percentage_discount(2002, 15, minimum_cents=3000) == 2002
    for amount in (0, 1, 99, 100, 101, 4999, 5000, 5001):
        for rate in (0, 1, 15, 99, 100):
            expected = amount - amount * rate // 100
            actual = apply_percentage_discount(amount, rate)
            assert type(actual) is int and actual == expected
    invoice = build_invoice(1001, 3, returned_quantity=1, discount_percent=15)
    assert invoice["merchandise_cents"] == 1702 and invoice["total_cents"] == 2202
    assert build_invoice(2500, 2)["shipping_cents"] == 0
    assert build_invoice(4999, 1)["shipping_cents"] == 500
    assert build_invoice(2501, 2, discount_percent=1)["total_cents"] == 5452
    assert build_invoice(3000, 2, tax_percent=10, customer="member-17")["total_cents"] == 6600
    assert shipping_charge(5000, "international") == 1200
    assert shipping_charge(10, "pickup", "express", fragile=True) == 0
    assert calculate_points(250, account="member", multiplier=2) == 4
    assert calculate_points(250, account="guest") == 0
    assert calculate_tax(1099, 10) == 109
    assert calculate_tax(1099, 10, jurisdiction="export") == 0
    assert adjustment_reference(7) == "RETURN-000007"
    assert checkout_reference(7) == "CHECKOUT-000007"
    assert campaign_reference("spring") == "CAMPAIGN:spring"
''')


def _ranking_task():
    return BenchmarkTask(
        "ranking",
        "Investigate the checkout invoice calculation and reproduce the existing failing tests. "
        "Returned units must reduce merchandise before percentage offers, and percentage discounts "
        "must round down to whole integer cents. Fix both defects using targeted edits, rerun the "
        "tests, and report the verification. Preserve shipping, tax, loyalty, receipt and audit "
        "behavior, all other helpers and comments, and the existing tests.",
        dict(_FILES), _ACCEPTANCE,
        unchanged_files=tuple(path for path in _FILES if path not in RELEVANT_PATHS),
    )


def _write(root, task):
    root = Path(root)
    if root.is_symlink():
        raise ValueError("Evaluation workspaces must not be symlinks")
    root.mkdir(parents=True, exist_ok=True)
    marker = root / ".patchrook-decision-fixture.json"
    if any(root.iterdir()):
        try:
            saved = json.loads(marker.read_text())
        except (OSError, ValueError) as error:
            raise ValueError("Refusing to reset a directory without a decision fixture marker") from error
        if saved != {"schema_version": 1, "fixture": task.name}:
            raise ValueError("Invalid decision fixture workspace marker")
        for entry in root.iterdir():
            if entry.is_dir() and not entry.is_symlink():
                shutil.rmtree(entry)
            else:
                entry.unlink()
    for name, content in task.files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    marker.write_text(json.dumps({"schema_version": 1, "fixture": task.name}) + "\n")
    return task


def create_fixture(root: Path, case: str = "ranking") -> BenchmarkTask:
    if case not in FIXTURE_NAMES:
        raise ValueError(f"Unknown decision fixture: {case}")
    return _write(root, _ranking_task())
