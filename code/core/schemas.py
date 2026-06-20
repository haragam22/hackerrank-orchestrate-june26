"""
code/schemas.py
Single source of truth for all data models, enums, and validators.
Every other file imports from here. Never define enums or output fields elsewhere.

Run directly to self-test:
    python code/schemas.py
"""

from __future__ import annotations
from enum import Enum
from typing import Optional
from pydantic import BaseModel, field_validator, model_validator
from collections import OrderedDict


# ---------------------------------------------------------------------------
# Enums — allowed values only, no freeform strings in output
# ---------------------------------------------------------------------------

class ClaimStatus(str, Enum):
    SUPPORTED = "supported"
    CONTRADICTED = "contradicted"
    NOT_ENOUGH_INFORMATION = "not_enough_information"


class Severity(str, Enum):
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    UNKNOWN = "unknown"


class IssueType(str, Enum):
    DENT = "dent"
    SCRATCH = "scratch"
    CRACK = "crack"
    GLASS_SHATTER = "glass_shatter"
    BROKEN_PART = "broken_part"
    MISSING_PART = "missing_part"
    TORN_PACKAGING = "torn_packaging"
    CRUSHED_PACKAGING = "crushed_packaging"
    WATER_DAMAGE = "water_damage"
    STAIN = "stain"
    NONE = "none"
    UNKNOWN = "unknown"


class CarPart(str, Enum):
    FRONT_BUMPER = "front_bumper"
    REAR_BUMPER = "rear_bumper"
    DOOR = "door"
    HOOD = "hood"
    WINDSHIELD = "windshield"
    SIDE_MIRROR = "side_mirror"
    HEADLIGHT = "headlight"
    TAILLIGHT = "taillight"
    FENDER = "fender"
    QUARTER_PANEL = "quarter_panel"
    BODY = "body"
    UNKNOWN = "unknown"


class LaptopPart(str, Enum):
    SCREEN = "screen"
    KEYBOARD = "keyboard"
    TRACKPAD = "trackpad"
    HINGE = "hinge"
    LID = "lid"
    CORNER = "corner"
    PORT = "port"
    BASE = "base"
    BODY = "body"
    UNKNOWN = "unknown"


class PackagePart(str, Enum):
    BOX = "box"
    PACKAGE_CORNER = "package_corner"
    PACKAGE_SIDE = "package_side"
    SEAL = "seal"
    LABEL = "label"
    CONTENTS = "contents"
    ITEM = "item"
    UNKNOWN = "unknown"


class RiskFlag(str, Enum):
    NONE = "none"
    BLURRY_IMAGE = "blurry_image"
    CROPPED_OR_OBSTRUCTED = "cropped_or_obstructed"
    LOW_LIGHT_OR_GLARE = "low_light_or_glare"
    WRONG_ANGLE = "wrong_angle"
    WRONG_OBJECT = "wrong_object"
    WRONG_OBJECT_PART = "wrong_object_part"
    DAMAGE_NOT_VISIBLE = "damage_not_visible"
    CLAIM_MISMATCH = "claim_mismatch"
    POSSIBLE_MANIPULATION = "possible_manipulation"
    NON_ORIGINAL_IMAGE = "non_original_image"
    TEXT_INSTRUCTION_PRESENT = "text_instruction_present"
    USER_HISTORY_RISK = "user_history_risk"
    MANUAL_REVIEW_REQUIRED = "manual_review_required"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

PART_ENUM_MAP = {
    "car": CarPart,
    "laptop": LaptopPart,
    "package": PackagePart,
}

# All valid values per object type — injected into prompts
PART_VALUES = {
    "car":     [e.value for e in CarPart],
    "laptop":  [e.value for e in LaptopPart],
    "package": [e.value for e in PackagePart],
}

ISSUE_TYPE_VALUES = [e.value for e in IssueType]
CLAIM_STATUS_VALUES = [e.value for e in ClaimStatus]
SEVERITY_VALUES = [e.value for e in Severity]

# Image quality flags that Agent 2 can produce
IMAGE_QUALITY_FLAGS = [
    RiskFlag.BLURRY_IMAGE,
    RiskFlag.CROPPED_OR_OBSTRUCTED,
    RiskFlag.LOW_LIGHT_OR_GLARE,
    RiskFlag.WRONG_ANGLE,
    RiskFlag.WRONG_OBJECT,
    RiskFlag.NON_ORIGINAL_IMAGE,
    RiskFlag.TEXT_INSTRUCTION_PRESENT,
    RiskFlag.POSSIBLE_MANIPULATION,
]

# Damage flags that Agent 3 can produce
DAMAGE_FLAGS = [
    RiskFlag.WRONG_OBJECT_PART,
    RiskFlag.DAMAGE_NOT_VISIBLE,
    RiskFlag.CLAIM_MISMATCH,
]


def resolve_object_part(claim_object: str, part_str: str) -> str:
    """
    Validate that part_str is a legal value for the given claim_object.
    Returns the validated string or 'unknown' if invalid.
    """
    enum_cls = PART_ENUM_MAP.get(claim_object.lower())
    if enum_cls is None:
        return "unknown"
    try:
        return enum_cls(part_str.lower().strip()).value
    except ValueError:
        return "unknown"


def coerce_flags(values: list, allowed: list[RiskFlag] | None = None) -> list[RiskFlag]:
    """
    Convert a list of raw strings/RiskFlag values into validated RiskFlag instances.
    Silently drops unknown values. Optionally filters to only allowed flags.
    """
    result = []
    for item in values:
        try:
            flag = RiskFlag(item) if not isinstance(item, RiskFlag) else item
            if allowed is None or flag in allowed:
                result.append(flag)
        except ValueError:
            pass  # drop unknown flags
    return result


# ---------------------------------------------------------------------------
# Agent 1 output: ParsedClaim
# ---------------------------------------------------------------------------

class ParsedClaim(BaseModel):
    """
    Output of Agent 1 (Claim Parser).
    Structured extraction from the user's conversation text.
    """
    object_part: str        # validated against claim_object's enum in the agent
    issue_type: IssueType
    claim_summary: str

    @field_validator("object_part", mode="before")
    @classmethod
    def lowercase_part(cls, v):
        return v.lower().strip() if isinstance(v, str) else v

    @field_validator("issue_type", mode="before")
    @classmethod
    def coerce_issue_type(cls, v):
        if isinstance(v, str):
            try:
                return IssueType(v.lower().strip())
            except ValueError:
                return IssueType.UNKNOWN
        return v

    @classmethod
    def fallback(cls) -> "ParsedClaim":
        """Safe fallback when Agent 1 parsing fails after retry."""
        return cls(
            object_part="unknown",
            issue_type=IssueType.UNKNOWN,
            claim_summary="Automated parsing failed — content could not be extracted.",
        )


# ---------------------------------------------------------------------------
# Agent 2 output: ImageValidation
# ---------------------------------------------------------------------------

class ImageValidation(BaseModel):
    """
    Output of Agent 2 (Image Validator).
    Assesses whether submitted images are usable for damage review.
    Does NOT assess damage — that is Agent 3's job.
    """
    valid_image: bool
    image_quality_flags: list[RiskFlag]     # only flags from IMAGE_QUALITY_FLAGS
    per_image_notes: str                    # free-text notes per image

    @field_validator("image_quality_flags", mode="before")
    @classmethod
    def coerce_image_flags(cls, v):
        if isinstance(v, list):
            return coerce_flags(v, allowed=IMAGE_QUALITY_FLAGS)
        return []

    @classmethod
    def fallback(cls) -> "ImageValidation":
        """
        Safe fallback when Agent 2 fails after retry.
        Treats as invalid — triggers fast path (skip Agent 3).
        """
        return cls(
            valid_image=False,
            image_quality_flags=[],
            per_image_notes="Image validation failed — automated review could not assess images.",
        )

    @classmethod
    def fast_path_invalid(cls, reason: str = "") -> "ImageValidation":
        """Convenience constructor for explicit invalid image cases."""
        return cls(
            valid_image=False,
            image_quality_flags=[],
            per_image_notes=reason or "Images deemed invalid by validator.",
        )


# ---------------------------------------------------------------------------
# Agent 3 output: DamageAssessment
# ---------------------------------------------------------------------------

class DamageAssessment(BaseModel):
    """
    Output of Agent 3 (Damage Assessor).
    This agent OWNS claim_status — it is set here and locked for the rest of the pipeline.
    """
    claim_status: ClaimStatus
    issue_type: IssueType
    object_part: str
    severity: Severity
    supporting_image_ids: list[str]         # filenames without extension, e.g. ["img_1"]
    justification: str
    damage_flags: list[RiskFlag]            # only flags from DAMAGE_FLAGS
    evidence_standard_met: bool
    evidence_standard_met_reason: str

    @field_validator("object_part", mode="before")
    @classmethod
    def lowercase_part(cls, v):
        return v.lower().strip() if isinstance(v, str) else v

    @field_validator("issue_type", mode="before")
    @classmethod
    def coerce_issue_type(cls, v):
        if isinstance(v, str):
            try:
                return IssueType(v.lower().strip())
            except ValueError:
                return IssueType.UNKNOWN
        return v

    @field_validator("claim_status", mode="before")
    @classmethod
    def coerce_claim_status(cls, v):
        if isinstance(v, str):
            try:
                return ClaimStatus(v.lower().strip())
            except ValueError:
                return ClaimStatus.NOT_ENOUGH_INFORMATION
        return v

    @field_validator("severity", mode="before")
    @classmethod
    def coerce_severity(cls, v):
        if isinstance(v, str):
            try:
                return Severity(v.lower().strip())
            except ValueError:
                return Severity.UNKNOWN
        return v

    @field_validator("damage_flags", mode="before")
    @classmethod
    def coerce_damage_flags(cls, v):
        if isinstance(v, list):
            return coerce_flags(v, allowed=DAMAGE_FLAGS)
        return []

    @field_validator("supporting_image_ids", mode="before")
    @classmethod
    def coerce_image_ids(cls, v):
        if isinstance(v, list):
            # Strip extensions, whitespace; drop empty strings
            cleaned = []
            for item in v:
                if isinstance(item, str):
                    s = item.strip()
                    # Remove common extensions if present
                    for ext in (".jpg", ".jpeg", ".png", ".webp"):
                        if s.lower().endswith(ext):
                            s = s[: -len(ext)]
                    if s:
                        cleaned.append(s)
            return cleaned
        return []

    @classmethod
    def fallback(cls) -> "DamageAssessment":
        """Safe fallback when Agent 3 fails after retry."""
        return cls(
            claim_status=ClaimStatus.NOT_ENOUGH_INFORMATION,
            issue_type=IssueType.UNKNOWN,
            object_part="unknown",
            severity=Severity.UNKNOWN,
            supporting_image_ids=[],
            justification="Automated damage assessment failed — manual review required.",
            damage_flags=[],
            evidence_standard_met=False,
            evidence_standard_met_reason="Assessment could not be completed.",
        )

    @classmethod
    def fast_path_fallback(cls) -> "DamageAssessment":
        """
        Used when Agent 2 returns valid_image=False.
        Agent 3 is skipped entirely — returns not_enough_information.
        """
        return cls(
            claim_status=ClaimStatus.NOT_ENOUGH_INFORMATION,
            issue_type=IssueType.UNKNOWN,
            object_part="unknown",
            severity=Severity.NONE,
            supporting_image_ids=[],
            justification="Images were deemed invalid or unusable for damage assessment.",
            damage_flags=[],
            evidence_standard_met=False,
            evidence_standard_met_reason="Images failed validation — damage could not be assessed.",
        )


# ---------------------------------------------------------------------------
# Final output: ClaimOutput — the 14-column output row
# ---------------------------------------------------------------------------

OUTPUT_COLUMNS = [
    "user_id",
    "image_paths",
    "user_claim",
    "claim_object",
    "evidence_standard_met",
    "evidence_standard_met_reason",
    "risk_flags",
    "issue_type",
    "object_part",
    "claim_status",
    "claim_status_justification",
    "supporting_image_ids",
    "valid_image",
    "severity",
]


class ClaimOutput(BaseModel):
    """
    Final output row. Matches output.csv schema exactly.
    Use to_csv_row() to get an ordered dict for pandas.
    """
    user_id: str
    image_paths: str                        # original semicolon-separated string, passed through
    user_claim: str                         # original claim text, passed through
    claim_object: str
    evidence_standard_met: bool
    evidence_standard_met_reason: str
    risk_flags: list[RiskFlag]              # stored as list, serialized on output
    issue_type: IssueType
    object_part: str
    claim_status: ClaimStatus
    claim_status_justification: str
    supporting_image_ids: list[str]
    valid_image: bool
    severity: Severity

    @field_validator("risk_flags", mode="before")
    @classmethod
    def coerce_risk_flags(cls, v):
        if isinstance(v, list):
            return coerce_flags(v)
        return []

    @field_validator("issue_type", mode="before")
    @classmethod
    def coerce_issue_type(cls, v):
        if isinstance(v, str):
            try:
                return IssueType(v.lower().strip())
            except ValueError:
                return IssueType.UNKNOWN
        return v

    @field_validator("claim_status", mode="before")
    @classmethod
    def coerce_claim_status(cls, v):
        if isinstance(v, str):
            try:
                return ClaimStatus(v.lower().strip())
            except ValueError:
                return ClaimStatus.NOT_ENOUGH_INFORMATION
        return v

    @field_validator("severity", mode="before")
    @classmethod
    def coerce_severity(cls, v):
        if isinstance(v, str):
            try:
                return Severity(v.lower().strip())
            except ValueError:
                return Severity.UNKNOWN
        return v

    def _serialize_bool(self, value: bool) -> str:
        return "true" if value else "false"

    def _serialize_flags(self) -> str:
        if not self.risk_flags:
            return "none"
        # Deduplicate preserving order; exclude 'none' sentinel if other flags present
        flags = list(dict.fromkeys(f.value for f in self.risk_flags if f != RiskFlag.NONE))
        return ";".join(flags) if flags else "none"

    def _serialize_image_ids(self) -> str:
        if not self.supporting_image_ids:
            return "none"
        return ";".join(self.supporting_image_ids)

    def to_csv_row(self) -> OrderedDict:
        """
        Returns an OrderedDict with keys in exact output.csv column order.
        All values are strings ready for CSV writing.
        """
        return OrderedDict([
            ("user_id",                    self.user_id),
            ("image_paths",                self.image_paths),
            ("user_claim",                 self.user_claim),
            ("claim_object",               self.claim_object),
            ("evidence_standard_met",      self._serialize_bool(self.evidence_standard_met)),
            ("evidence_standard_met_reason", self.evidence_standard_met_reason),
            ("risk_flags",                 self._serialize_flags()),
            ("issue_type",                 self.issue_type.value),
            ("object_part",                self.object_part),
            ("claim_status",               self.claim_status.value),
            ("claim_status_justification", self.claim_status_justification),
            ("supporting_image_ids",       self._serialize_image_ids()),
            ("valid_image",                self._serialize_bool(self.valid_image)),
            ("severity",                   self.severity.value),
        ])

    @classmethod
    def from_fallback(cls, claim_row: dict) -> "ClaimOutput":
        """
        Produces a safe fallback output row from raw claim input dict.
        Used when the full pipeline fails for a claim — guarantees one row per claim.
        """
        return cls(
            user_id=claim_row.get("user_id", "unknown"),
            image_paths=claim_row.get("image_paths", ""),
            user_claim=claim_row.get("user_claim", ""),
            claim_object=claim_row.get("claim_object", "unknown"),
            evidence_standard_met=False,
            evidence_standard_met_reason="Automated review failed — manual review required.",
            risk_flags=[RiskFlag.MANUAL_REVIEW_REQUIRED],
            issue_type=IssueType.UNKNOWN,
            object_part="unknown",
            claim_status=ClaimStatus.NOT_ENOUGH_INFORMATION,
            claim_status_justification="Automated review failed — manual review required.",
            supporting_image_ids=[],
            valid_image=False,
            severity=Severity.UNKNOWN,
        )


# ---------------------------------------------------------------------------
# Self-test — run directly: python code/schemas.py
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys, io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    print("=" * 60)
    print("schemas.py self-test")
    print("=" * 60)

    # ── Test enums ────────────────────────────────────────────────
    print("\n[1] Testing enums...")
    assert ClaimStatus("supported") == ClaimStatus.SUPPORTED
    assert Severity("high") == Severity.HIGH
    assert IssueType("dent") == IssueType.DENT
    assert CarPart("rear_bumper") == CarPart.REAR_BUMPER
    assert LaptopPart("screen") == LaptopPart.SCREEN
    assert PackagePart("box") == PackagePart.BOX
    assert RiskFlag("blurry_image") == RiskFlag.BLURRY_IMAGE
    print("   ✅ All enums OK")

    # ── Test resolve_object_part ──────────────────────────────────
    print("\n[2] Testing resolve_object_part...")
    assert resolve_object_part("car", "rear_bumper") == "rear_bumper"
    assert resolve_object_part("car", "REAR_BUMPER") == "rear_bumper"
    assert resolve_object_part("car", "banana") == "unknown"
    assert resolve_object_part("laptop", "screen") == "screen"
    assert resolve_object_part("package", "box") == "box"
    assert resolve_object_part("unknown_object", "anything") == "unknown"
    print("   ✅ resolve_object_part OK")

    # ── Test ParsedClaim ──────────────────────────────────────────
    print("\n[3] Testing ParsedClaim...")
    pc = ParsedClaim(
        object_part="REAR_BUMPER",
        issue_type="dent",
        claim_summary="Customer claims a dent on the rear bumper from a collision.",
    )
    assert pc.object_part == "rear_bumper"
    assert pc.issue_type == IssueType.DENT
    assert "dent" in pc.claim_summary.lower()

    fallback_pc = ParsedClaim.fallback()
    assert fallback_pc.object_part == "unknown"
    assert fallback_pc.issue_type == IssueType.UNKNOWN
    print("   ✅ ParsedClaim OK")

    # ── Test ImageValidation ──────────────────────────────────────
    print("\n[4] Testing ImageValidation...")
    iv = ImageValidation(
        valid_image=True,
        image_quality_flags=["blurry_image", "wrong_angle"],
        per_image_notes="img_1 is blurry; img_2 has wrong angle.",
    )
    assert iv.valid_image is True
    assert RiskFlag.BLURRY_IMAGE in iv.image_quality_flags
    assert RiskFlag.WRONG_ANGLE in iv.image_quality_flags
    # Damage flags must NOT appear in image_quality_flags
    iv_with_bad_flag = ImageValidation(
        valid_image=True,
        image_quality_flags=["blurry_image", "claim_mismatch"],  # claim_mismatch is a damage flag
        per_image_notes="test",
    )
    assert RiskFlag.CLAIM_MISMATCH not in iv_with_bad_flag.image_quality_flags, \
        "Damage flag slipped into image_quality_flags"

    fallback_iv = ImageValidation.fallback()
    assert fallback_iv.valid_image is False
    fast_iv = ImageValidation.fast_path_invalid("No images found.")
    assert fast_iv.valid_image is False
    print("   ✅ ImageValidation OK")

    # ── Test DamageAssessment ─────────────────────────────────────
    print("\n[5] Testing DamageAssessment...")
    da = DamageAssessment(
        claim_status="supported",
        issue_type="DENT",
        object_part="rear_bumper",
        severity="high",
        supporting_image_ids=["img_1.jpg", "img_2"],  # one with extension, one without
        justification="Clear dent visible on rear bumper in img_1.",
        damage_flags=["damage_not_visible"],  # should be dropped if not in DAMAGE_FLAGS... wait it is
        evidence_standard_met=True,
        evidence_standard_met_reason="Rear bumper clearly visible with damage.",
    )
    assert da.claim_status == ClaimStatus.SUPPORTED
    assert da.issue_type == IssueType.DENT
    assert da.severity == Severity.HIGH
    assert da.evidence_standard_met is True
    # Extension should be stripped from img_1
    assert "img_1" in da.supporting_image_ids
    assert "img_1.jpg" not in da.supporting_image_ids
    assert "img_2" in da.supporting_image_ids

    fallback_da = DamageAssessment.fallback()
    assert fallback_da.claim_status == ClaimStatus.NOT_ENOUGH_INFORMATION
    fast_da = DamageAssessment.fast_path_fallback()
    assert fast_da.claim_status == ClaimStatus.NOT_ENOUGH_INFORMATION
    assert fast_da.severity == Severity.NONE
    print("   ✅ DamageAssessment OK")

    # ── Test ClaimOutput + to_csv_row ─────────────────────────────
    print("\n[6] Testing ClaimOutput serialization...")
    sample = ClaimOutput(
        user_id="user_001",
        image_paths="images/sample/case_001/img_1.jpg",
        user_claim="My car's rear bumper has a dent.",
        claim_object="car",
        evidence_standard_met=True,
        evidence_standard_met_reason="Rear bumper clearly visible with dent.",
        risk_flags=[RiskFlag.USER_HISTORY_RISK, RiskFlag.MANUAL_REVIEW_REQUIRED],
        issue_type=IssueType.DENT,
        object_part="rear_bumper",
        claim_status=ClaimStatus.SUPPORTED,
        claim_status_justification="Image clearly shows dent on rear bumper.",
        supporting_image_ids=["img_1"],
        valid_image=True,
        severity=Severity.MEDIUM,
    )
    row = sample.to_csv_row()
    assert list(row.keys()) == OUTPUT_COLUMNS, f"Column order mismatch: {list(row.keys())}"
    assert row["evidence_standard_met"] == "true"
    assert row["valid_image"] == "true"
    assert row["risk_flags"] == "user_history_risk;manual_review_required"
    assert row["supporting_image_ids"] == "img_1"
    assert row["claim_status"] == "supported"
    assert row["severity"] == "medium"
    print("   ✅ ClaimOutput serialization OK")

    # ── Test 'none' flag sentinel ─────────────────────────────────
    print("\n[7] Testing risk_flags 'none' sentinel...")
    no_flags = ClaimOutput(
        user_id="user_002", image_paths="", user_claim="", claim_object="car",
        evidence_standard_met=False, evidence_standard_met_reason="",
        risk_flags=[], issue_type=IssueType.UNKNOWN, object_part="unknown",
        claim_status=ClaimStatus.NOT_ENOUGH_INFORMATION,
        claim_status_justification="", supporting_image_ids=[],
        valid_image=False, severity=Severity.UNKNOWN,
    )
    assert no_flags.to_csv_row()["risk_flags"] == "none"

    none_sentinel = ClaimOutput(
        user_id="user_003", image_paths="", user_claim="", claim_object="car",
        evidence_standard_met=False, evidence_standard_met_reason="",
        risk_flags=[RiskFlag.NONE], issue_type=IssueType.UNKNOWN, object_part="unknown",
        claim_status=ClaimStatus.NOT_ENOUGH_INFORMATION,
        claim_status_justification="", supporting_image_ids=[],
        valid_image=False, severity=Severity.UNKNOWN,
    )
    assert none_sentinel.to_csv_row()["risk_flags"] == "none"
    print("   ✅ none sentinel OK")

    # ── Test from_fallback ────────────────────────────────────────
    print("\n[8] Testing ClaimOutput.from_fallback...")
    fallback = ClaimOutput.from_fallback({
        "user_id": "user_999",
        "image_paths": "img.jpg",
        "user_claim": "test claim",
        "claim_object": "car",
    })
    fb_row = fallback.to_csv_row()
    assert fb_row["claim_status"] == "not_enough_information"
    assert fb_row["valid_image"] == "false"
    assert fb_row["evidence_standard_met"] == "false"
    assert "manual_review_required" in fb_row["risk_flags"]
    assert fb_row["supporting_image_ids"] == "none"
    print("   ✅ from_fallback OK")

    # ── Test coerce_flags helper ──────────────────────────────────
    print("\n[9] Testing coerce_flags helper...")
    flags = coerce_flags(
        ["blurry_image", "invalid_flag_xyz", "wrong_angle", RiskFlag.NONE],
        allowed=IMAGE_QUALITY_FLAGS
    )
    assert RiskFlag.BLURRY_IMAGE in flags
    assert RiskFlag.WRONG_ANGLE in flags
    assert RiskFlag.NONE not in flags          # not in IMAGE_QUALITY_FLAGS
    assert len([f for f in flags if f.value == "invalid_flag_xyz"]) == 0
    print("   ✅ coerce_flags OK")

    print("\n" + "=" * 60)
    print("schemas.py self-test complete ✅  All 9 checks passed.")
    print("=" * 60)
