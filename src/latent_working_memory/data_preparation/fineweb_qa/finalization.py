"""Apply the fixed semantic review before assembling final FineWeb QA trajectories."""

from __future__ import annotations

from collections import defaultdict

from latent_working_memory.data_preparation.fineweb_qa.assembly import assemble_document


REVIEW_FIELDS = {
    "qa_id",
    "accepted",
    "reason",
    "same_fact_with",
    "evidence_prediction_correct",
}


def validate_resolved_reviews(panel: list[dict], resolved: dict) -> dict[str, dict]:
    """Require one complete reviewer decision for every QA on the frozen panel."""
    if not isinstance(panel, list):
        raise ValueError("review panel must be a list")
    panel_ids = []
    for row in panel:
        if (
            not isinstance(row, dict)
            or not isinstance(row.get("qa_id"), str)
            or not row["qa_id"].strip()
        ):
            raise ValueError("every review-panel row needs a nonempty qa_id")
        panel_ids.append(row["qa_id"])
    if len(panel_ids) != len(set(panel_ids)):
        raise ValueError("review panel has duplicate QA IDs")
    if not isinstance(resolved, dict) or set(resolved) != {"decisions"}:
        raise ValueError("resolved review must contain only decisions")
    decisions = resolved["decisions"]
    if not isinstance(decisions, list):
        raise ValueError("resolved review decisions must be a list")

    by_id = {}
    for decision in decisions:
        if not isinstance(decision, dict) or set(decision) != REVIEW_FIELDS:
            raise ValueError("each resolved review decision needs the canonical fields")
        qa_id = decision["qa_id"]
        if not isinstance(qa_id, str) or not qa_id.strip():
            raise ValueError("resolved review qa_id must be a nonempty string")
        if qa_id in by_id:
            raise ValueError(f"duplicate resolved review for {qa_id}")
        if type(decision["accepted"]) is not bool:
            raise ValueError(f"review accepted must be boolean for {qa_id}")
        reason = decision["reason"]
        if not isinstance(reason, str) or (not decision["accepted"] and not reason.strip()):
            raise ValueError(f"rejected review needs a reason for {qa_id}")
        references = decision["same_fact_with"]
        if (
            not isinstance(references, list)
            or any(
                not isinstance(reference, str) or not reference.strip() for reference in references
            )
            or len(references) != len(set(references))
        ):
            raise ValueError(f"same_fact_with must contain distinct QA IDs for {qa_id}")
        if qa_id in references:
            raise ValueError(f"review cannot merge {qa_id} with itself")
        if not decision["accepted"] and references:
            raise ValueError(f"rejected review cannot merge facts for {qa_id}")
        evidence_correct = decision["evidence_prediction_correct"]
        if evidence_correct is not None and type(evidence_correct) is not bool:
            raise ValueError(f"evidence_prediction_correct must be boolean or null for {qa_id}")
        by_id[qa_id] = dict(decision, same_fact_with=references.copy())

    if set(by_id) != set(panel_ids):
        missing = sorted(set(panel_ids) - set(by_id))
        unknown = sorted(set(by_id) - set(panel_ids))
        raise ValueError(
            f"resolved reviews must cover the panel exactly; missing={missing}, unknown={unknown}"
        )
    return by_id


def validate_document_reviews(panel: list[dict], resolved: dict, result: dict) -> dict[str, dict]:
    """Validate a document's final decisions, including externally corrected decisions."""
    by_id = validate_resolved_reviews(panel, resolved)
    available = {d["qa_id"] for d in result["review_decisions"] if d["accepted"]}
    for decision in by_id.values():
        if decision["evidence_prediction_correct"] is None:
            raise ValueError("every sampled evidence answer needs a semantic review decision")
        if decision["qa_id"] not in available or any(
            target not in available for target in decision["same_fact_with"]
        ):
            raise ValueError("adjudication links an unknown or rejected candidate")
        if any(
            target in by_id and not by_id[target]["accepted"]
            for target in decision["same_fact_with"]
        ):
            raise ValueError("adjudication links a jointly rejected panel item")
    return by_id


def apply_resolved_reviews(
    document: dict, result: dict, resolved_by_id: dict[str, dict], qa_config: dict
) -> dict:
    """Remove reviewer-rejected QAs and merge reviewed facts before role allocation."""
    candidates = result["local_candidates"]
    decisions = result["review_decisions"]
    candidate_ids = [candidate["qa_id"] for candidate in candidates]
    if len(candidate_ids) != len(set(candidate_ids)):
        raise ValueError("local candidate QA IDs must be unique")
    model_by_id = {}
    for decision in decisions:
        qa_id = decision["qa_id"]
        if qa_id in model_by_id:
            raise ValueError(f"duplicate model review decision for {qa_id}")
        model_by_id[qa_id] = decision
    if set(model_by_id) != set(candidate_ids):
        raise ValueError("model review decisions must cover local candidates exactly")

    candidate_by_id = {candidate["qa_id"]: candidate for candidate in candidates}
    local_reviews = {
        qa_id: resolved_by_id[qa_id] for qa_id in candidate_ids if qa_id in resolved_by_id
    }
    retained = {
        qa_id
        for qa_id in candidate_ids
        if model_by_id[qa_id]["accepted"]
        and (qa_id not in local_reviews or local_reviews[qa_id]["accepted"])
    }

    parent = {qa_id: qa_id for qa_id in candidate_ids}

    def root(qa_id: str) -> str:
        while parent[qa_id] != qa_id:
            parent[qa_id] = parent[parent[qa_id]]
            qa_id = parent[qa_id]
        return qa_id

    def union(first: str, second: str) -> None:
        first_root, second_root = root(first), root(second)
        if first_root != second_root:
            parent[second_root] = first_root

    first_by_group = {}
    for qa_id in candidate_ids:
        if qa_id not in retained:
            continue
        group_id = model_by_id[qa_id]["fact_group_id"]
        if group_id in first_by_group:
            union(first_by_group[group_id], qa_id)
        else:
            first_by_group[group_id] = qa_id

    for qa_id in candidate_ids:
        if qa_id not in local_reviews:
            continue
        review = local_reviews[qa_id]
        if not review["accepted"]:
            if review["same_fact_with"]:
                raise ValueError(f"rejected QA {qa_id} cannot merge facts")
            continue
        for target in review["same_fact_with"]:
            if target not in candidate_by_id:
                raise ValueError(f"same_fact_with target {target} is not in this document")
            if target in local_reviews and not local_reviews[target]["accepted"]:
                raise ValueError(f"same_fact_with target {target} is not retained")
            if target == qa_id:
                raise ValueError(f"QA {qa_id} cannot merge with itself")
            union(qa_id, target)

    components: dict[str, list[str]] = defaultdict(list)
    for qa_id in candidate_ids:
        if qa_id in retained:
            components[root(qa_id)].append(qa_id)
    candidate_order = {qa_id: index for index, qa_id in enumerate(candidate_ids)}
    group_by_id = {}
    used_group_ids = set()
    for members in components.values():
        original_groups = {model_by_id[qa_id]["fact_group_id"] for qa_id in members}
        if len(original_groups) == 1:
            group_id = next(iter(original_groups))
        else:
            group_id = min(
                members,
                key=lambda qa_id: (
                    int(candidate_by_id[qa_id]["segment_id"][3:]),
                    candidate_order[qa_id],
                ),
            )
        if group_id in used_group_ids:
            raise ValueError(f"merged fact group ID collides with another group: {group_id}")
        used_group_ids.add(group_id)
        for qa_id in members:
            group_by_id[qa_id] = group_id

    updated = []
    for decision in decisions:
        qa_id = decision["qa_id"]
        copy = decision.copy()
        if qa_id in local_reviews and not local_reviews[qa_id]["accepted"]:
            copy.update(accepted=False, reason=local_reviews[qa_id]["reason"], fact_group_id="")
        elif qa_id in retained:
            copy["fact_group_id"] = group_by_id[qa_id]
        updated.append(copy)
    return assemble_document(document, candidates, updated, qa_config)
