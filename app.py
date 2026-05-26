from __future__ import annotations

from html import escape
from math import gcd
from typing import Any

import streamlit as st
import streamlit.components.v1 as components

from water_jug_solver import (
    Action,
    GoalSpec,
    ReachabilityAnalysis,
    SearchResult,
    SearchTree,
    State,
    analyse_reachability,
    build_search_tree,
    solve_water_jugs,
    state_space_size,
)


ALGORITHMS = {
    "BFS - menor número de passos": "bfs",
    "DFS - exploração em profundidade": "dfs",
    "Custo uniforme - menor custo acumulado": "ucs",
    "A* - busca informada": "astar",
}

COST_MODES = {
    "Número de operações": "operations",
    "Volume total movimentado": "volume",
}

GOAL_MODES = {
    "Quantidade em qualquer jarro": "any_jug",
    "Soma dos jarros seleccionados": "selected_sum",
    "Qualquer combinação de jarros": "any_combination",
    "Estado exacto de todos os jarros": "exact_state",
}


def main() -> None:
    st.set_page_config(
        page_title="Jarros de Água Generalizados",
        page_icon="💧",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    inject_theme()

    st.markdown(
        """
        <section class="app-hero">
            <div>
                <p class="eyebrow">Busca em espaço de estados</p>
                <h1>Jarros de Água Generalizados</h1>
                <p class="hero-copy">Configuração flexível, sequência explicável e visualização passo a passo.</p>
            </div>
            <div class="hero-visual" aria-hidden="true">
                <div class="hero-jug hero-jug-a"><span></span></div>
                <div class="hero-stream"></div>
                <div class="hero-jug hero-jug-b"><span></span></div>
            </div>
        </section>
        """,
        unsafe_allow_html=True,
    )

    config = render_sidebar()
    reachability = cached_reachability(
        tuple(config["capacities"]),
        tuple(config["initial_state"]),
        goal_cache_key(config["goal"]),
        min(config["max_nodes"], 200_000),
    )
    validation_messages = validate_reachability_hint(config, reachability)

    left, right = st.columns([0.86, 1.14], gap="large")
    with left:
        render_problem_panel(config, validation_messages)

    with right:
        if st.button("Resolver", type="primary", use_container_width=True):
            run_solver(config)

        result: SearchResult | None = st.session_state.get("result")
        result_config: dict[str, Any] | None = st.session_state.get("result_config")

        if result is not None and result_config != serialisable_config(config):
            st.info("Os parâmetros foram alterados. Prima Resolver para actualizar a sequência.")
        if result is not None:
            render_compact_result(result)

    if result is None:
        st.markdown("#### Estado inicial")
        render_jug_stage(tuple(config["initial_state"]), tuple(config["capacities"]))
    else:
        render_result(result, include_summary=False)


def render_sidebar() -> dict[str, Any]:
    with st.sidebar:
        st.header("Configuração")

        number_of_jugs = st.slider("Número de jarros", min_value=2, max_value=7, value=3, step=1)

        default_capacities = [8, 5, 3, 11, 7, 13, 9]
        capacities: list[int] = []
        capacity_columns = st.columns(2)
        for index in range(number_of_jugs):
            with capacity_columns[index % 2]:
                capacities.append(
                    int(
                        st.number_input(
                            f"Capacidade J{index + 1} (L)",
                            min_value=1,
                            max_value=100,
                            value=default_capacities[index],
                            step=1,
                            key=f"capacity_{index}",
                        )
                    )
                )

        goal_label = st.selectbox("Tipo de objectivo", list(GOAL_MODES.keys()), index=0)
        goal_mode = GOAL_MODES[goal_label]

        selected_indices: tuple[int, ...] = ()
        exact_state: tuple[int, ...] = ()
        target = 4

        if goal_mode == "exact_state":
            st.caption("Estado objectivo")
            target_values: list[int] = []
            target_columns = st.columns(2)
            for index in range(number_of_jugs):
                with target_columns[index % 2]:
                    target_values.append(
                        int(
                            st.number_input(
                                f"Objectivo J{index + 1} (L)",
                                min_value=0,
                                max_value=capacities[index],
                                value=min(0 if index else 4, capacities[index]),
                                step=1,
                                key=f"exact_goal_{index}",
                            )
                        )
                    )
            exact_state = tuple(target_values)
            target = sum(exact_state)
        else:
            max_target = sum(capacities) if goal_mode != "any_jug" else max(capacities)
            target = int(
                st.number_input(
                    "Quantidade objectivo (L)",
                    min_value=0,
                    max_value=max_target,
                    value=min(4, max_target),
                    step=1,
                )
            )

        if goal_mode == "selected_sum":
            selected = st.multiselect(
                "Jarros da combinação",
                options=list(range(number_of_jugs)),
                default=list(range(min(2, number_of_jugs))),
                format_func=lambda index: f"J{index + 1}",
            )
            selected_indices = tuple(sorted(selected))

        algorithm_label = st.selectbox("Algoritmo de busca", list(ALGORITHMS.keys()), index=0)
        cost_label = st.selectbox(
            "Custo a optimizar",
            list(COST_MODES.keys()),
            index=0,
            help="BFS privilegia passos; custo uniforme e A* usam este custo na fronteira.",
        )

        with st.expander("Estado inicial e limites"):
            custom_initial = st.toggle("Começar com água nos jarros", value=False)
            initial_state: list[int] = []
            initial_columns = st.columns(2)
            for index in range(number_of_jugs):
                default_value = 0 if not custom_initial else min(capacities[index], 0)
                with initial_columns[index % 2]:
                    initial_value = int(
                        st.number_input(
                            f"Inicial J{index + 1} (L)",
                            min_value=0,
                            max_value=capacities[index],
                            value=default_value,
                            step=1,
                            disabled=not custom_initial,
                            key=f"initial_{index}",
                        )
                    )
                    initial_state.append(initial_value if custom_initial else 0)
            max_nodes = int(
                st.number_input(
                    "Limite de nós explorados",
                    min_value=100,
                    max_value=2_000_000,
                    value=50_000,
                    step=1_000,
                )
            )
        with st.expander("Grafo parcial"):
            tree_max_nodes = int(
                st.slider("Nós no grafo", min_value=12, max_value=120, value=48, step=6)
            )
            tree_max_depth = int(
                st.slider("Profundidade", min_value=1, max_value=8, value=4, step=1)
            )

    goal = GoalSpec(
        mode=goal_mode,  # type: ignore[arg-type]
        target=target,
        selected_indices=selected_indices,
        exact_state=exact_state,
    )
    return {
        "capacities": capacities,
        "initial_state": initial_state,
        "goal": goal,
        "algorithm": ALGORITHMS[algorithm_label],
        "cost_mode": COST_MODES[cost_label],
        "goal_label": goal_label,
        "algorithm_label": algorithm_label,
        "cost_label": cost_label,
        "max_nodes": max_nodes,
        "tree_max_nodes": tree_max_nodes,
        "tree_max_depth": tree_max_depth,
    }


@st.cache_data(show_spinner=False)
def cached_reachability(
    capacities: tuple[int, ...],
    initial_state: tuple[int, ...],
    goal_key: tuple[Any, ...],
    max_nodes: int,
) -> ReachabilityAnalysis:
    return analyse_reachability(
        capacities,
        goal_from_cache_key(goal_key),
        initial_state=initial_state,
        max_nodes=max_nodes,
    )


def goal_cache_key(goal: GoalSpec) -> tuple[Any, ...]:
    return (goal.mode, goal.target, tuple(goal.selected_indices), tuple(goal.exact_state))


def goal_from_cache_key(goal_key: tuple[Any, ...]) -> GoalSpec:
    mode, target, selected_indices, exact_state = goal_key
    return GoalSpec(
        mode=mode,
        target=int(target),
        selected_indices=tuple(selected_indices),
        exact_state=tuple(exact_state),
    )


def compare_solutions(config: dict[str, Any]) -> dict[str, SearchResult]:
    common_arguments = {
        "initial_state": config["initial_state"],
        "max_nodes": config["max_nodes"],
    }
    return {
        "Menor número de passos": solve_water_jugs(
            config["capacities"],
            config["goal"],
            algorithm="bfs",
            cost_mode="operations",
            **common_arguments,
        ),
        "Menor volume movimentado": solve_water_jugs(
            config["capacities"],
            config["goal"],
            algorithm="ucs",
            cost_mode="volume",
            **common_arguments,
        ),
    }


def render_problem_panel(config: dict[str, Any], validation_messages: list[tuple[str, str]]) -> None:
    capacities = tuple(config["capacities"])
    initial = tuple(config["initial_state"])
    goal: GoalSpec = config["goal"]
    size = state_space_size(capacities)
    common_divisor = 0
    for capacity in capacities:
        common_divisor = gcd(common_divisor, capacity)

    st.markdown("#### Problema")
    summary_rows = [
        ("Capacidades", " · ".join(f"J{index + 1}: {capacity} L" for index, capacity in enumerate(capacities))),
        ("Estado inicial", format_state(initial)),
        ("Objectivo", describe_goal(goal)),
        ("Espaço máximo", f"{size:,} estados".replace(",", " ")),
        ("MDC das capacidades", f"{common_divisor} L"),
    ]

    st.markdown('<div class="summary-panel">', unsafe_allow_html=True)
    for label, value in summary_rows:
        st.markdown(
            f"""
            <div class="summary-row">
                <span>{escape(label)}</span>
                <strong>{escape(value)}</strong>
            </div>
            """,
            unsafe_allow_html=True,
        )
    st.markdown("</div>", unsafe_allow_html=True)

    for level, message in validation_messages:
        if level == "warning":
            st.warning(message)
        elif level == "error":
            st.error(message)
        else:
            st.info(message)


def run_solver(config: dict[str, Any]) -> None:
    reachability = cached_reachability(
        tuple(config["capacities"]),
        tuple(config["initial_state"]),
        goal_cache_key(config["goal"]),
        min(config["max_nodes"], 200_000),
    )

    tree = build_search_tree(
        config["capacities"],
        config["goal"],
        initial_state=config["initial_state"],
        max_nodes=config["tree_max_nodes"],
        max_depth=config["tree_max_depth"],
    )

    if reachability.possible is False:
        result = SearchResult(
            success=False,
            states=[tuple(config["initial_state"])],
            actions=[],
            total_cost=0,
            total_volume_moved=0,
            nodes_explored=reachability.checked_states,
            elapsed_ms=reachability.elapsed_ms,
            message=reachability.reason,
            algorithm=config["algorithm"],
            cost_mode=config["cost_mode"],
        )
        st.session_state.result = result
        st.session_state.result_config = serialisable_config(config)
        st.session_state.current_step = 0
        st.session_state.comparison = {}
        st.session_state.search_tree = tree
        return

    try:
        result = solve_water_jugs(
            config["capacities"],
            config["goal"],
            initial_state=config["initial_state"],
            algorithm=config["algorithm"],
            cost_mode=config["cost_mode"],
            max_nodes=config["max_nodes"],
        )
    except ValueError as error:
        st.error(str(error))
        return

    st.session_state.result = result
    st.session_state.result_config = serialisable_config(config)
    st.session_state.current_step = len(result.actions) if result.success else 0
    st.session_state.comparison = compare_solutions(config) if result.success else {}
    st.session_state.search_tree = tree


def render_compact_result(result: SearchResult) -> None:
    if result.success:
        st.success(result.message)
    else:
        st.error(result.message)

    metrics = [
        ("Passos", str(len(result.actions))),
        ("Volume", f"{result.total_volume_moved} L"),
        ("Nós explorados", f"{result.nodes_explored:,}".replace(",", " ")),
        ("Tempo", f"{result.elapsed_ms:.2f} ms"),
    ]
    cards = "".join(
        f'<div class="compact-card"><span>{escape(label)}</span><strong>{escape(value)}</strong></div>'
        for label, value in metrics
    )
    st.markdown(f'<div class="compact-metrics">{cards}</div>', unsafe_allow_html=True)


def render_result(result: SearchResult, *, include_summary: bool = True) -> None:
    if include_summary:
        if result.success:
            st.success(result.message)
        else:
            st.error(result.message)

        metric_columns = st.columns(4)
        metric_columns[0].metric("Número de passos", len(result.actions))
        metric_columns[1].metric("Volume total movimentado", f"{result.total_volume_moved} L")
        metric_columns[2].metric("Nós explorados", f"{result.nodes_explored:,}".replace(",", " "))
        metric_columns[3].metric("Tempo de execução", f"{result.elapsed_ms:.2f} ms")

    if result.success:
        step_count = len(result.actions)
        st.session_state.current_step = min(st.session_state.get("current_step", step_count), step_count)

        control_columns = st.columns([1, 1, 4, 1], vertical_alignment="center")
        with control_columns[0]:
            if st.button("◀", use_container_width=True, help="Passo anterior"):
                st.session_state.current_step = max(0, st.session_state.current_step - 1)
        with control_columns[1]:
            if st.button("▶", use_container_width=True, help="Passo seguinte"):
                st.session_state.current_step = min(step_count, st.session_state.current_step + 1)
        with control_columns[2]:
            selected_step = st.slider(
                "Passo visualizado",
                min_value=0,
                max_value=step_count,
                value=st.session_state.current_step,
                step=1,
            )
            st.session_state.current_step = selected_step
        with control_columns[3]:
            if st.button("⟲", use_container_width=True, help="Voltar ao início"):
                st.session_state.current_step = 0

        step = st.session_state.current_step
        action = result.actions[step - 1] if step > 0 else None
        state = result.states[step]
        before_state = result.states[step - 1] if step > 0 else None

        st.markdown(f"#### Passo {step}")
        if action is None:
            st.caption("Estado inicial")
        else:
            st.caption(action.label)
        render_jug_stage(state, capacities_from_result(result), action=action, before_state=before_state)

        render_comparison(st.session_state.get("comparison", {}))
        render_search_tree(st.session_state.get("search_tree"), result)

        rows = build_sequence_rows(result)
        st.markdown("#### Sequência de operações")
        st.dataframe(rows, use_container_width=True, hide_index=True)

        csv = sequence_to_csv(rows)
        st.download_button(
            "Exportar sequência CSV",
            data=csv,
            file_name="sequencia_jarros.csv",
            mime="text/csv",
            use_container_width=True,
        )
    else:
        render_jug_stage(result.states[0], capacities_from_result(result))
        render_search_tree(st.session_state.get("search_tree"), result)


def render_comparison(comparison: dict[str, SearchResult]) -> None:
    if not comparison:
        return

    st.markdown("#### Comparação de soluções")
    labels = list(comparison.keys())
    step_result = comparison.get("Menor número de passos")
    volume_result = comparison.get("Menor volume movimentado")

    comparison_cards = "".join(
        render_comparison_card(label, comparison[label])
        for label in labels
    )
    st.markdown(f'<div class="compare-grid">{comparison_cards}</div>', unsafe_allow_html=True)

    if step_result and volume_result and step_result.success and volume_result.success:
        step_delta = len(volume_result.actions) - len(step_result.actions)
        volume_delta = step_result.total_volume_moved - volume_result.total_volume_moved
        if step_delta or volume_delta:
            st.info(
                f"A solução por volume poupa {max(volume_delta, 0)} L"
                f" e usa {max(step_delta, 0)} passos adicionais em relação à solução mais curta."
            )

    rows = []
    for label, result in comparison.items():
        rows.append(
            {
                "Critério": label,
                "Algoritmo": "BFS" if result.algorithm == "bfs" else "Custo uniforme",
                "Passos": len(result.actions) if result.success else "sem solução",
                "Volume": f"{result.total_volume_moved} L" if result.success else "-",
                "Nós explorados": f"{result.nodes_explored:,}".replace(",", " "),
                "Tempo": f"{result.elapsed_ms:.2f} ms",
            }
        )
    st.dataframe(rows, use_container_width=True, hide_index=True)

    with st.expander("Ver sequências comparadas"):
        tabs = st.tabs(labels)
        for tab, label in zip(tabs, labels):
            with tab:
                result = comparison[label]
                if result.success:
                    st.dataframe(build_sequence_rows(result), use_container_width=True, hide_index=True)
                else:
                    st.warning(result.message)


def render_comparison_card(label: str, result: SearchResult) -> str:
    if result.success:
        steps = str(len(result.actions))
        volume = f"{result.total_volume_moved} L"
        foot = f"{result.nodes_explored:,} nós · {result.elapsed_ms:.2f} ms".replace(",", " ")
    else:
        steps = "-"
        volume = "-"
        foot = result.message

    return (
        '<div class="compare-card">'
        f'<span>{escape(label)}</span>'
        f'<strong>{escape(steps)} passos</strong>'
        f'<em>{escape(volume)}</em>'
        f'<small>{escape(foot)}</small>'
        "</div>"
    )


def render_search_tree(tree: SearchTree | None, result: SearchResult) -> None:
    if tree is None:
        return

    st.markdown("#### Árvore parcial de busca")
    if tree.truncated:
        st.caption(
            f"Mostrando {len(tree.nodes)} nós até profundidade {tree.max_depth}; "
            "alguns ramos foram omitidos para manter a leitura clara."
        )
    else:
        st.caption(f"Árvore parcial com {len(tree.nodes)} nós gerados.")

    st.graphviz_chart(build_tree_dot(tree, set(result.states)), use_container_width=True)


def build_tree_dot(tree: SearchTree, solution_states: set[State]) -> str:
    lines = [
        "digraph SearchTree {",
        "  graph [rankdir=LR, bgcolor=\"transparent\", ranksep=\"0.65\", nodesep=\"0.35\"];",
        "  node [shape=box, style=\"rounded,filled\", fontname=\"Inter\", fontsize=10, margin=\"0.09,0.07\"];",
        "  edge [fontname=\"Inter\", fontsize=9, color=\"#0f9f9a\", fontcolor=\"#475569\", arrowsize=0.7];",
    ]

    for node in tree.nodes:
        label = dot_escape(format_state_multiline(node.state))
        fill = "#ffffff"
        color = "#cbd5e1"
        penwidth = "1.2"
        if node.id == 0:
            fill = "#e0f2fe"
            color = "#0ea5e9"
            penwidth = "2"
        if node.state in solution_states:
            fill = "#ccfbf1"
            color = "#0f9f9a"
            penwidth = "2.4"
        if node.is_goal:
            fill = "#dcfce7"
            color = "#16a34a"
            penwidth = "2.6"
        lines.append(
            f'  n{node.id} [label="{label}\\nD{node.depth}", fillcolor="{fill}", color="{color}", penwidth="{penwidth}"];'
        )

    for edge in tree.edges:
        lines.append(
            f'  n{edge.source_id} -> n{edge.target_id} [label="{dot_escape(short_action_label(edge.action))}"];'
        )

    lines.append("}")
    return "\n".join(lines)


def format_state_multiline(state: State) -> str:
    return "\\n".join(f"J{index + 1}={value}L" for index, value in enumerate(state))


def short_action_label(action: Action) -> str:
    if action.kind == "fill" and action.target is not None:
        return f"Encher J{action.target + 1} (+{action.amount}L)"
    if action.kind == "empty" and action.source is not None:
        return f"Esvaziar J{action.source + 1} (-{action.amount}L)"
    if action.kind == "transfer" and action.source is not None and action.target is not None:
        return f"J{action.source + 1} -> J{action.target + 1}: {action.amount}L"
    return action.label


def dot_escape(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"')


def render_jug_stage(
    state: State,
    capacities: State,
    *,
    action: Action | None = None,
    before_state: State | None = None,
) -> None:
    number_of_jugs = len(state)
    stage_width = max(760, number_of_jugs * 128)
    flow = render_flow(action, number_of_jugs)
    jug_cards = "\n".join(render_jug(index, amount, capacities[index], action, before_state) for index, amount in enumerate(state))
    action_label = "Estado inicial" if action is None else action.label

    html = f"""
    <div class="jug-shell">
      <style>
        .jug-shell {{
          width: 100%;
          overflow-x: auto;
          padding: 6px 2px 2px;
          font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
        }}
        .jug-stage {{
          min-width: {stage_width}px;
          position: relative;
          border: 1px solid rgba(16, 24, 40, 0.10);
          border-radius: 18px;
          padding: 76px 22px 22px;
          background:
            linear-gradient(180deg, rgba(255,255,255,0.96), rgba(246,251,255,0.94)),
            url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='640' height='220' viewBox='0 0 640 220'%3E%3Cg fill='none' stroke='%230f9f9a' stroke-opacity='.12' stroke-width='2'%3E%3Cpath d='M0 56 C88 26 136 86 224 56 S384 26 480 56 584 82 640 56'/%3E%3Cpath d='M0 152 C88 122 136 182 224 152 S384 122 480 152 584 178 640 152'/%3E%3C/g%3E%3C/svg%3E");
          box-shadow: 0 20px 48px rgba(15, 23, 42, 0.10);
        }}
        .action-badge {{
          position: absolute;
          top: 18px;
          left: 22px;
          right: 22px;
          display: flex;
          align-items: center;
          justify-content: center;
          min-height: 38px;
          border-radius: 999px;
          color: #073b4c;
          background: rgba(255, 255, 255, 0.82);
          border: 1px solid rgba(14, 116, 144, 0.18);
          font-weight: 800;
          font-size: 15px;
          letter-spacing: 0;
          box-shadow: 0 10px 26px rgba(8, 47, 73, 0.10);
        }}
        .jugs-grid {{
          position: relative;
          display: grid;
          grid-template-columns: repeat({number_of_jugs}, minmax(96px, 1fr));
          gap: 18px;
          align-items: end;
          z-index: 2;
        }}
        .jug-card {{
          position: relative;
          display: grid;
          justify-items: center;
          min-height: 286px;
        }}
        .jug-name {{
          margin-bottom: 8px;
          display: inline-flex;
          align-items: center;
          justify-content: center;
          width: 46px;
          height: 30px;
          border-radius: 999px;
          color: #0f172a;
          background: #ffffff;
          border: 1px solid rgba(15, 23, 42, 0.10);
          box-shadow: 0 8px 20px rgba(15, 23, 42, 0.08);
          font-size: 13px;
          font-weight: 900;
        }}
        .jug-body {{
          position: relative;
          width: 86px;
          height: 204px;
          border: 5px solid rgba(15, 23, 42, 0.72);
          border-top: 0;
          border-radius: 8px 8px 30px 30px;
          background:
            linear-gradient(90deg, rgba(255,255,255,0.76), rgba(255,255,255,0.18) 38%, rgba(255,255,255,0.54));
          overflow: hidden;
          box-shadow:
            inset 9px 0 18px rgba(255, 255, 255, 0.48),
            inset -9px 0 18px rgba(15, 23, 42, 0.10),
            0 18px 36px rgba(15, 23, 42, 0.16);
        }}
        .jug-body:before {{
          content: "";
          position: absolute;
          right: -25px;
          top: 42px;
          width: 34px;
          height: 72px;
          border: 5px solid rgba(15, 23, 42, 0.55);
          border-left: 0;
          border-radius: 0 28px 28px 0;
        }}
        .jug-neck {{
          width: 52px;
          height: 17px;
          border: 5px solid rgba(15, 23, 42, 0.72);
          border-bottom: 0;
          border-radius: 10px 10px 0 0;
          background: rgba(255, 255, 255, 0.72);
        }}
        .water {{
          position: absolute;
          left: 0;
          right: 0;
          bottom: 0;
          height: var(--water-height);
          min-height: var(--water-min);
          background:
            linear-gradient(180deg, rgba(125, 249, 255, 0.92), rgba(14, 165, 233, 0.92) 48%, rgba(13, 148, 136, 0.96));
          transition: height 420ms ease;
          box-shadow: inset 0 12px 24px rgba(255,255,255,0.38);
        }}
        .water:before,
        .water:after {{
          content: "";
          position: absolute;
          left: -18%;
          width: 136%;
          height: 18px;
          top: -9px;
          background: rgba(240, 253, 250, 0.76);
          border-radius: 45%;
          animation: wave 2.4s ease-in-out infinite;
        }}
        .water:after {{
          opacity: 0.55;
          transform: translateX(14px);
          animation-duration: 3.1s;
        }}
        .bubbles {{
          position: absolute;
          inset: 10px 14px 8px;
          background:
            radial-gradient(circle at 18% 72%, rgba(255,255,255,0.80) 0 3px, transparent 4px),
            radial-gradient(circle at 72% 62%, rgba(255,255,255,0.64) 0 2px, transparent 3px),
            radial-gradient(circle at 48% 36%, rgba(255,255,255,0.70) 0 2px, transparent 3px);
          animation: bubbles 2.8s linear infinite;
        }}
        .jug-value {{
          margin-top: 12px;
          color: #073b4c;
          font-weight: 900;
          font-size: 15px;
        }}
        .jug-delta {{
          margin-top: 4px;
          min-height: 20px;
          color: #9a3412;
          font-weight: 800;
          font-size: 12px;
        }}
        .flow-layer {{
          position: absolute;
          inset: 0;
          z-index: 1;
          pointer-events: none;
        }}
        .transfer-flow {{
          position: absolute;
          top: 98px;
          left: var(--flow-left);
          width: var(--flow-width);
          height: 10px;
          border-radius: 999px;
          background: repeating-linear-gradient(90deg, rgba(14,165,233,0), rgba(14,165,233,0) 0 8px, rgba(14,165,233,0.92) 8px 25px, rgba(103,232,249,0.92) 25px 38px);
          filter: drop-shadow(0 6px 8px rgba(14, 116, 144, 0.22));
          animation: stream 850ms linear infinite;
        }}
        .transfer-flow.reverse {{
          animation-direction: reverse;
        }}
        .transfer-flow:after {{
          content: "";
          position: absolute;
          top: -6px;
          width: 0;
          height: 0;
          border-top: 11px solid transparent;
          border-bottom: 11px solid transparent;
        }}
        .transfer-flow.forward:after {{
          right: -13px;
          border-left: 17px solid rgba(14, 165, 233, 0.95);
        }}
        .transfer-flow.reverse:after {{
          left: -13px;
          border-right: 17px solid rgba(14, 165, 233, 0.95);
        }}
        .vertical-flow {{
          position: absolute;
          left: var(--flow-left);
          transform: translateX(-50%);
          width: 13px;
          border-radius: 999px;
          background: repeating-linear-gradient(180deg, rgba(14,165,233,0), rgba(14,165,233,0) 0 6px, rgba(14,165,233,0.92) 6px 22px, rgba(125,249,255,0.9) 22px 34px);
          filter: drop-shadow(0 6px 8px rgba(14, 116, 144, 0.18));
          animation: verticalStream 800ms linear infinite;
        }}
        .fill-flow {{
          top: 70px;
          height: 94px;
        }}
        .empty-flow {{
          top: 262px;
          height: 92px;
          background: repeating-linear-gradient(180deg, rgba(251,146,60,0), rgba(251,146,60,0) 0 6px, rgba(251,146,60,0.94) 6px 22px, rgba(253,186,116,0.9) 22px 34px);
        }}
        @keyframes wave {{
          0%, 100% {{ transform: translateX(0) scaleY(1); }}
          50% {{ transform: translateX(16px) scaleY(0.72); }}
        }}
        @keyframes bubbles {{
          0% {{ transform: translateY(12px); opacity: 0.65; }}
          100% {{ transform: translateY(-18px); opacity: 1; }}
        }}
        @keyframes stream {{
          to {{ background-position: 48px 0; }}
        }}
        @keyframes verticalStream {{
          to {{ background-position: 0 48px; }}
        }}
      </style>
      <div class="jug-stage">
        <div class="action-badge">{escape(action_label)}</div>
        <div class="flow-layer">{flow}</div>
        <div class="jugs-grid">{jug_cards}</div>
      </div>
    </div>
    """
    components.html(html, height=435, scrolling=False)


def render_jug(index: int, amount: int, capacity: int, action: Action | None, before_state: State | None) -> str:
    percent = 0 if capacity == 0 else max(0, min(100, amount / capacity * 100))
    water_min = "5px" if amount > 0 else "0px"
    delta = ""
    if action is not None and before_state is not None:
        difference = amount - before_state[index]
        if difference > 0:
            delta = f"+{difference} L"
        elif difference < 0:
            delta = f"{difference} L"

    return f"""
      <div class="jug-card">
        <div class="jug-name">J{index + 1}</div>
        <div class="jug-neck"></div>
        <div class="jug-body">
          <div class="water" style="--water-height: {percent:.2f}%; --water-min: {water_min};">
            <div class="bubbles"></div>
          </div>
        </div>
        <div class="jug-value">{amount} / {capacity} L</div>
        <div class="jug-delta">{escape(delta)}</div>
      </div>
    """


def render_flow(action: Action | None, number_of_jugs: int) -> str:
    if action is None:
        return ""

    cell = 100 / number_of_jugs

    if action.kind == "transfer" and action.source is not None and action.target is not None:
        source_center = (action.source + 0.5) * cell
        target_center = (action.target + 0.5) * cell
        left = min(source_center, target_center)
        width = abs(target_center - source_center)
        direction = "forward" if action.source < action.target else "reverse"
        return (
            f'<div class="transfer-flow {direction}" '
            f'style="--flow-left: {left:.4f}%; --flow-width: {width:.4f}%;"></div>'
        )

    if action.kind == "fill" and action.target is not None:
        target_center = (action.target + 0.5) * cell
        return f'<div class="vertical-flow fill-flow" style="--flow-left: {target_center:.4f}%;"></div>'

    if action.kind == "empty" and action.source is not None:
        source_center = (action.source + 0.5) * cell
        return f'<div class="vertical-flow empty-flow" style="--flow-left: {source_center:.4f}%;"></div>'

    return ""


def build_sequence_rows(result: SearchResult) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = [
        {
            "Passo": 0,
            "Acção": "Estado inicial",
            "Estado": format_state(result.states[0]),
            "Volume movido": "0 L",
            "Volume acumulado": "0 L",
            "Custo acumulado": "0",
        }
    ]
    cumulative_volume = 0
    cumulative_cost = 0.0

    for step, action in enumerate(result.actions, start=1):
        cumulative_volume += action.amount
        cumulative_cost += 1 if result.cost_mode == "operations" else action.amount
        rows.append(
            {
                "Passo": step,
                "Acção": action.label,
                "Estado": format_state(result.states[step]),
                "Volume movido": f"{action.amount} L",
                "Volume acumulado": f"{cumulative_volume} L",
                "Custo acumulado": f"{cumulative_cost:g}",
            }
        )

    return rows


def sequence_to_csv(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return ""
    headers = list(rows[0].keys())
    lines = [";".join(headers)]
    for row in rows:
        lines.append(";".join(str(row[header]).replace(";", ",") for header in headers))
    return "\n".join(lines)


def format_state(state: State) -> str:
    return " | ".join(f"J{index + 1}={value} L" for index, value in enumerate(state))


def describe_goal(goal: GoalSpec) -> str:
    if goal.mode == "any_jug":
        return f"{goal.target} L em pelo menos um jarro"
    if goal.mode == "selected_sum":
        indices = goal.selected_indices or ()
        selected = ", ".join(f"J{index + 1}" for index in indices) if indices else "todos os jarros"
        return f"{goal.target} L na soma de {selected}"
    if goal.mode == "any_combination":
        return f"{goal.target} L em qualquer combinação de jarros"
    if goal.mode == "exact_state":
        return f"Estado exacto: {format_state(goal.exact_state)}"
    return "Objectivo configurado"


def capacities_from_result(result: SearchResult) -> State:
    config = st.session_state.get("result_config", {})
    capacities = config.get("capacities")
    if capacities:
        return tuple(capacities)
    return tuple(max(values) for values in zip(*result.states))


def validate_reachability_hint(
    config: dict[str, Any],
    reachability: ReachabilityAnalysis,
) -> list[tuple[str, str]]:
    capacities = tuple(config["capacities"])
    goal: GoalSpec = config["goal"]
    messages: list[tuple[str, str]] = []
    size = state_space_size(capacities)

    if size > 1_000_000:
        messages.append(("warning", f"O espaço pode chegar a {size:,} estados; considere reduzir capacidades ou limite.".replace(",", " ")))

    if goal.mode == "selected_sum" and not goal.selected_indices:
        messages.append(("error", "Seleccione pelo menos um jarro para o objectivo por soma."))

    if reachability.possible is False:
        messages.append(("error", reachability.reason))
    elif reachability.possible is None:
        messages.append(("warning", reachability.reason.replace(",", " ")))
    else:
        messages.append(
            (
                "info",
                f"Detecção automática: objectivo alcançável após verificar {reachability.checked_states:,} estados.".replace(",", " "),
            )
        )

    return messages


def serialisable_config(config: dict[str, Any]) -> dict[str, Any]:
    goal: GoalSpec = config["goal"]
    return {
        "capacities": list(config["capacities"]),
        "initial_state": list(config["initial_state"]),
        "goal": {
            "mode": goal.mode,
            "target": goal.target,
            "selected_indices": list(goal.selected_indices),
            "exact_state": list(goal.exact_state),
        },
        "algorithm": config["algorithm"],
        "cost_mode": config["cost_mode"],
        "max_nodes": config["max_nodes"],
        "tree_max_nodes": config["tree_max_nodes"],
        "tree_max_depth": config["tree_max_depth"],
    }


def inject_theme() -> None:
    st.markdown(
        """
        <style>
          :root {
            --text: #172033;
            --muted: #5f6c80;
            --panel: rgba(255, 255, 255, 0.88);
            --line: rgba(23, 32, 51, 0.10);
            --teal: #0f9f9a;
            --coral: #f97316;
          }
          .stApp {
            color: var(--text);
            background:
              linear-gradient(180deg, rgba(248,251,255,0.96), rgba(238,248,246,0.96) 48%, rgba(255,250,244,0.96)),
              url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='900' height='420' viewBox='0 0 900 420'%3E%3Cg fill='none' stroke='%230f9f9a' stroke-opacity='.08' stroke-width='2'%3E%3Cpath d='M0 74 C140 22 220 126 360 74 S600 22 900 74'/%3E%3Cpath d='M0 238 C140 186 220 290 360 238 S600 186 900 238'/%3E%3C/g%3E%3Cg fill='none' stroke='%23f97316' stroke-opacity='.07' stroke-width='2'%3E%3Cpath d='M0 156 C120 116 210 190 330 156 S560 116 900 156'/%3E%3C/g%3E%3C/svg%3E");
            background-size: auto, 900px 420px;
          }
          [data-testid="stSidebar"] {
            background: rgba(255, 255, 255, 0.86);
            border-right: 1px solid var(--line);
          }
          [data-testid="stSidebar"] h1,
          [data-testid="stSidebar"] h2,
          [data-testid="stSidebar"] h3,
          [data-testid="stSidebar"] p,
          [data-testid="stSidebar"] label {
            color: #172033 !important;
          }
          [data-testid="stSidebar"] * {
            letter-spacing: 0;
          }
          .block-container {
            padding-top: 1.4rem;
            padding-bottom: 2.5rem;
          }
          .app-hero {
            min-height: 188px;
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 28px;
            padding: 30px 34px;
            margin-bottom: 22px;
            overflow: hidden;
            border: 1px solid var(--line);
            border-radius: 22px;
            background:
              linear-gradient(135deg, rgba(255,255,255,0.93), rgba(240,253,250,0.80)),
              url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='800' height='260' viewBox='0 0 800 260'%3E%3Cg fill='none' stroke='%230f9f9a' stroke-opacity='.18' stroke-width='2'%3E%3Cpath d='M-20 184 C120 110 190 230 330 156 S560 80 840 156'/%3E%3Cpath d='M-20 216 C130 132 214 252 360 182 S610 112 840 188'/%3E%3C/g%3E%3C/svg%3E");
            background-size: cover;
            box-shadow: 0 22px 54px rgba(15, 23, 42, 0.10);
          }
          .eyebrow {
            margin: 0 0 8px;
            color: var(--teal);
            font-size: 13px;
            font-weight: 900;
            text-transform: uppercase;
            letter-spacing: 0;
          }
          .app-hero h1 {
            margin: 0;
            color: #111827;
            font-size: clamp(2.1rem, 4.8vw, 4.8rem);
            line-height: 0.98;
            letter-spacing: 0;
          }
          .hero-copy {
            max-width: 680px;
            margin: 14px 0 0;
            color: var(--muted);
            font-size: 1.05rem;
          }
          .hero-visual {
            min-width: 230px;
            height: 142px;
            position: relative;
          }
          .hero-jug {
            position: absolute;
            bottom: 10px;
            width: 74px;
            height: 104px;
            border: 5px solid rgba(15, 23, 42, 0.72);
            border-top: 0;
            border-radius: 8px 8px 25px 25px;
            overflow: hidden;
            background: rgba(255,255,255,0.72);
            box-shadow: inset 8px 0 18px rgba(255,255,255,.58), 0 16px 28px rgba(15,23,42,.13);
          }
          .hero-jug:before {
            content: "";
            position: absolute;
            top: 25px;
            right: -24px;
            width: 28px;
            height: 42px;
            border: 5px solid rgba(15, 23, 42, 0.50);
            border-left: 0;
            border-radius: 0 22px 22px 0;
          }
          .hero-jug span {
            position: absolute;
            left: 0;
            right: 0;
            bottom: 0;
            background: linear-gradient(180deg, #7df9ff, #0ea5e9 54%, #0f9f9a);
            box-shadow: inset 0 10px 18px rgba(255,255,255,.48);
          }
          .hero-jug span:before {
            content: "";
            position: absolute;
            top: -8px;
            left: -16px;
            width: 110px;
            height: 16px;
            border-radius: 50%;
            background: rgba(240,253,250,.78);
            animation: heroWave 2.6s ease-in-out infinite;
          }
          .hero-jug-a {
            left: 10px;
            transform: rotate(-9deg);
          }
          .hero-jug-a span {
            height: 74%;
          }
          .hero-jug-b {
            right: 8px;
          }
          .hero-jug-b span {
            height: 42%;
          }
          .hero-stream {
            position: absolute;
            left: 88px;
            top: 50px;
            width: 82px;
            height: 10px;
            border-radius: 999px;
            transform: rotate(14deg);
            background: repeating-linear-gradient(90deg, rgba(14,165,233,0), rgba(14,165,233,0) 0 8px, rgba(14,165,233,.92) 8px 24px, rgba(125,249,255,.9) 24px 36px);
            filter: drop-shadow(0 6px 8px rgba(14, 116, 144, .18));
            animation: heroStream 900ms linear infinite;
          }
          @keyframes heroWave {
            0%, 100% { transform: translateX(0) scaleY(1); }
            50% { transform: translateX(14px) scaleY(.76); }
          }
          @keyframes heroStream {
            to { background-position: 48px 0; }
          }
          div[data-testid="stMetric"] {
            padding: 18px 18px 16px;
            border: 1px solid var(--line);
            border-radius: 16px;
            background: rgba(255, 255, 255, 0.78);
            box-shadow: 0 14px 32px rgba(15, 23, 42, 0.08);
          }
          div[data-testid="stMetricLabel"] p {
            color: var(--muted);
            font-weight: 800;
          }
          div[data-testid="stMetricValue"] {
            color: #073b4c;
          }
          .summary-panel {
            display: grid;
            gap: 10px;
            padding: 16px;
            border: 1px solid var(--line);
            border-radius: 16px;
            background: rgba(255,255,255,0.74);
            box-shadow: 0 14px 32px rgba(15, 23, 42, 0.07);
          }
          .summary-row {
            display: grid;
            grid-template-columns: minmax(120px, 0.45fr) 1fr;
            gap: 12px;
            align-items: center;
            padding: 10px 12px;
            border-radius: 10px;
            background: rgba(248, 250, 252, 0.86);
          }
          .summary-row span {
            color: var(--muted);
            font-weight: 800;
          }
          .summary-row strong {
            color: #172033;
            font-weight: 900;
          }
          .compact-metrics {
            display: grid;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            gap: 14px;
            margin-top: 14px;
          }
          .compact-card {
            min-height: 112px;
            display: flex;
            flex-direction: column;
            justify-content: center;
            gap: 10px;
            padding: 18px 20px;
            border: 1px solid var(--line);
            border-radius: 16px;
            background: rgba(255, 255, 255, 0.82);
            box-shadow: 0 14px 32px rgba(15, 23, 42, 0.08);
          }
          .compact-card span {
            color: var(--muted);
            font-size: 0.86rem;
            font-weight: 900;
          }
          .compact-card strong {
            color: #073b4c;
            font-size: clamp(1.8rem, 3vw, 2.5rem);
            line-height: 1;
            font-weight: 900;
          }
          .compare-grid {
            display: grid;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            gap: 14px;
            margin: 8px 0 16px;
          }
          .compare-card {
            min-height: 142px;
            display: flex;
            flex-direction: column;
            justify-content: center;
            gap: 8px;
            padding: 18px 20px;
            border: 1px solid var(--line);
            border-radius: 16px;
            background: linear-gradient(180deg, rgba(255,255,255,0.88), rgba(240,253,250,0.66));
            box-shadow: 0 14px 32px rgba(15, 23, 42, 0.08);
          }
          .compare-card span {
            color: var(--teal);
            font-size: 0.86rem;
            font-weight: 900;
          }
          .compare-card strong {
            color: #073b4c;
            font-size: clamp(1.45rem, 2.4vw, 2.2rem);
            line-height: 1.05;
            font-style: normal;
            font-weight: 900;
          }
          .compare-card em {
            color: #0f172a;
            font-size: 1.05rem;
            font-style: normal;
            font-weight: 900;
          }
          .compare-card small {
            color: var(--muted);
            font-weight: 800;
          }
          .stButton > button,
          .stDownloadButton > button {
            border-radius: 12px;
            font-weight: 900;
            letter-spacing: 0;
          }
          button[data-testid="stBaseButton-primary"] {
            color: #ffffff;
            border: 0;
            background: linear-gradient(135deg, #0f9f9a, #0ea5e9);
            box-shadow: 0 12px 24px rgba(14, 116, 144, 0.20);
          }
          button[data-testid="stBaseButton-primary"]:hover {
            color: #ffffff;
            border: 0;
            background: linear-gradient(135deg, #0d9488, #0284c7);
          }
          .stDataFrame {
            border-radius: 14px;
            overflow: hidden;
          }
          @media (max-width: 760px) {
            .app-hero {
              min-height: 220px;
              padding: 24px;
              align-items: flex-start;
            }
            .hero-visual {
              display: none;
            }
            .summary-row {
              grid-template-columns: 1fr;
            }
            .compact-metrics,
            .compare-grid {
              grid-template-columns: 1fr;
            }
          }
        </style>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
