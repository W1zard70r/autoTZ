import logging
from typing import List, Dict, Any

import networkx as nx

from schemas.enums import NodeLabel, EdgeRelation
from schemas.graph import VoteCount, DecisionResolution

logger = logging.getLogger(__name__)


def resolve_decisions(G: nx.MultiDiGraph) -> List[DecisionResolution]:
    """Разрешает Decision-узлы на основе голосований в графе."""
    resolutions: List[DecisionResolution] = []
    decision_nodes = [
        (nid, data) for nid, data in G.nodes(data=True)
        if data.get("label") in (NodeLabel.DECISION.value, NodeLabel.DECISION)
    ]

    for decision_id, decision_data in decision_nodes:
        decision_name = decision_data.get("name", decision_id)

        option_ids = [
            v for _, v, edata in G.out_edges(decision_id, data=True)
            if edata.get("relation") in (EdgeRelation.RELATES_TO.value, EdgeRelation.RELATES_TO)
        ]

        if not option_ids:
            logger.warning(f"⚠️ Decision '{decision_id}' не имеет вариантов (RELATES_TO рёбра не найдены)")
            continue

        vote_counts: Dict[str, VoteCount] = {}
        for opt_id in option_ids:
            opt_name = G.nodes[opt_id].get("name", opt_id) if G.has_node(opt_id) else opt_id
            vote_counts[opt_id] = VoteCount(option_id=opt_id, option_name=opt_name)

        for src, tgt, edata in G.edges(data=True):
            relation = edata.get("relation", "")
            if isinstance(relation, EdgeRelation):
                relation = relation.value

            if tgt not in vote_counts:
                continue

            voter_name = G.nodes[src].get("name", src) if G.has_node(src) else src

            if relation == EdgeRelation.VOTED_FOR.value:
                vote_counts[tgt].votes_for += 1
                vote_counts[tgt].voters_for.append(voter_name)
            elif relation == EdgeRelation.VOTED_AGAINST.value:
                vote_counts[tgt].votes_against += 1
                vote_counts[tgt].voters_against.append(voter_name)

        options_list = list(vote_counts.values())

        if not any(vc.votes_for + vc.votes_against > 0 for vc in options_list):
            resolution = DecisionResolution(
                decision_id=decision_id,
                decision_name=decision_name,
                is_tie=True,
                options=options_list,
                conflict_description="Голосов не обнаружено. Решение не принято.",
            )
        else:
            sorted_options = sorted(options_list, key=lambda x: x.score, reverse=True)
            top = sorted_options[0]
            second = sorted_options[1] if len(sorted_options) > 1 else None
            is_tie = second is not None and top.score == second.score

            if is_tie:
                tied_names = [o.option_name for o in sorted_options if o.score == top.score]
                resolution = DecisionResolution(
                    decision_id=decision_id,
                    decision_name=decision_name,
                    is_tie=True,
                    options=options_list,
                    conflict_description=(
                        f"Ничья между: {', '.join(tied_names)} (счёт {top.score}). "
                        f"Требуется ручное решение."
                    ),
                )
            else:
                resolution = DecisionResolution(
                    decision_id=decision_id,
                    decision_name=decision_name,
                    winner_id=top.option_id,
                    winner_name=top.option_name,
                    is_tie=False,
                    options=options_list,
                )
                G.add_edge(
                    decision_id,
                    top.option_id,
                    relation=EdgeRelation.RESOLVED_TO.value,
                    evidence=f"Победитель голосования: {top.votes_for} за, {top.votes_against} против",
                )
                logger.info(
                    f"  🗳️  '{decision_name}': победил '{top.option_name}' "
                    f"({top.votes_for}✅ / {top.votes_against}❌)"
                )

        resolutions.append(resolution)

    return resolutions


def format_merge_report(
    resolutions: List[DecisionResolution],
    conflicts: List[Any],
) -> str:
    """Форматирует отчёт о голосованиях и конфликтах."""
    lines: List[str] = []

    if resolutions:
        lines.append("=" * 60)
        lines.append("🗳️  ИТОГИ ГОЛОСОВАНИЙ")
        lines.append("=" * 60)
        for res in resolutions:
            lines.append(f"\n📌 {res.decision_name}")
            lines.append("-" * 40)
            for opt in sorted(res.options, key=lambda x: x.score, reverse=True):
                bar_for = "✅" * opt.votes_for
                bar_against = "❌" * opt.votes_against
                score_str = f"[{opt.score:+d}]"
                voters_for_str = f"  За: {', '.join(opt.voters_for)}" if opt.voters_for else ""
                voters_against_str = f"  Против: {', '.join(opt.voters_against)}" if opt.voters_against else ""

                lines.append(
                    f"  {'👑 ' if res.winner_id == opt.option_id else '   '}"
                    f"{opt.option_name:<25} {bar_for}{bar_against}  {score_str}"
                )
                if voters_for_str:
                    lines.append(f"           {voters_for_str}")
                if voters_against_str:
                    lines.append(f"           {voters_against_str}")

            if res.is_tie:
                lines.append(f"\n  ⚠️  КОНФЛИКТ: {res.conflict_description}")
            elif res.winner_name:
                lines.append(f"\n  ✅ ПРИНЯТО: {res.winner_name}")

    if conflicts:
        lines.append("\n" + "=" * 60)
        lines.append("⚡ КОНФЛИКТЫ ГРАФА (требуют внимания)")
        lines.append("=" * 60)
        for conflict in conflicts:
            lines.append(f"\n  •[{conflict.node_id}] {conflict.description}")
            if conflict.conflicting_values:
                for val in conflict.conflicting_values:
                    lines.append(f"      ↳ {val}")

    if not resolutions and not conflicts:
        lines.append("✅ Голосований и конфликтов не обнаружено.")

    return "\n".join(lines)