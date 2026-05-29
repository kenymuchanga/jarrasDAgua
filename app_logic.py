from __future__ import annotations

from typing import Any

from water_jug_solver import GoalSpec, SearchResult, State, solve_water_jugs, state_space_size


# Aqui ficam as opções que aparecem na interface.
# O texto da esquerda é o que o utilizador vê, e o valor da direita é o que o programa usa.
ALGORITHMS = {
    "BFS - menor número de passos": "bfs",
    "DFS - exploração em profundidade": "dfs",
    "Custo uniforme - menor custo acumulado": "ucs",
    "A* - busca informada": "astar",
}


# Estes nomes são usados nos relatórios e nos cartões educativos.
ALGORITHM_NAMES = {
    "bfs": "BFS",
    "dfs": "DFS",
    "ucs": "Custo uniforme",
    "astar": "A*",
}


# Textos curtos para explicar os algoritmos no modo educativo.
EDUCATIONAL_NOTES = {
    "BFS": "Explora por camadas. É bom para encontrar a solução com menos operações quando cada acção tem o mesmo peso.",
    "DFS": "Segue um ramo até ao fundo antes de voltar. Pode encontrar soluções rapidamente, mas não garante a menor sequência.",
    "Custo uniforme": "Expande primeiro o estado com menor custo acumulado. É indicado quando queremos minimizar volume movimentado.",
    "A*": "Usa custo acumulado e uma estimativa do que falta. Procura reduzir exploração mantendo uma direcção inteligente.",
}


# Tipos de custo que podemos optimizar.
COST_MODES = {
    "Número de operações": "operations",
    "Volume total movimentado": "volume",
}


# Tipos de objectivo que a aplicação permite escolher.
GOAL_MODES = {
    "Quantidade em qualquer jarro": "any_jug",
    "Soma dos jarros seleccionados": "selected_sum",
    "Qualquer combinação de jarros": "any_combination",
    "Estado exacto de todos os jarros": "exact_state",
}


def goal_cache_key(goal: GoalSpec) -> tuple[Any, ...]:
    # Transformo o objectivo numa tupla simples para poder guardar em cache.
    return (goal.mode, goal.target, tuple(goal.selected_indices), tuple(goal.exact_state))


def goal_from_cache_key(goal_key: tuple[Any, ...]) -> GoalSpec:
    # Recrio o objectivo original a partir da chave usada na cache.
    mode, target, selected_indices, exact_state = goal_key
    return GoalSpec(
        mode=mode,
        target=int(target),
        selected_indices=tuple(selected_indices),
        exact_state=tuple(exact_state),
    )


def compare_solutions(config: dict[str, Any]) -> dict[str, SearchResult]:
    # Aqui comparo automaticamente duas estratégias importantes:
    # uma para menos passos e outra para menor volume movimentado.
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


def problem_history_key(config: dict[str, Any]) -> tuple[Any, ...]:
    # Esta chave identifica o problema.
    # Se o utilizador mudar capacidades, objectivo ou estado inicial, é outro problema.
    goal: GoalSpec = config["goal"]
    return (
        tuple(config["capacities"]),
        tuple(config["initial_state"]),
        goal.mode,
        goal.target,
        tuple(goal.selected_indices),
        tuple(goal.exact_state),
    )


def cost_mode_name(cost_mode: str) -> str:
    # Nome simples para aparecer no relatório.
    if cost_mode == "operations":
        return "operações"
    return "volume"


def build_sequence_rows(result: SearchResult) -> list[dict[str, Any]]:
    # Preparo os dados da sequência para a tabela do Streamlit.
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
    # Converto a tabela da sequência para CSV para o botão de download.
    if not rows:
        return ""
    headers = list(rows[0].keys())
    lines = [";".join(headers)]
    for row in rows:
        lines.append(";".join(str(row[header]).replace(";", ",") for header in headers))
    return "\n".join(lines)


def format_state(state: State) -> str:
    # Mostro o estado no formato J1=0 L | J2=5 L.
    return " | ".join(f"J{index + 1}={value} L" for index, value in enumerate(state))


def describe_goal(goal: GoalSpec) -> str:
    # Crio uma frase curta para explicar o objectivo escolhido.
    if goal.mode == "any_jug":
        return f"{goal.target} L em pelo menos uma jarra"
    if goal.mode == "selected_sum":
        indices = goal.selected_indices or ()
        selected = ", ".join(f"J{index + 1}" for index in indices) if indices else "todas as jarras"
        return f"{goal.target} L na soma de {selected}"
    if goal.mode == "any_combination":
        return f"{goal.target} L em qualquer combinação de jarras"
    if goal.mode == "exact_state":
        return f"Estado exacto: {format_state(goal.exact_state)}"
    return "Objectivo configurado"


def validate_reachability_hint(config: dict[str, Any], reachability: Any) -> list[tuple[str, str]]:
    # Gero avisos sobre o tamanho do problema e se o objectivo é possível.
    capacities = tuple(config["capacities"])
    goal: GoalSpec = config["goal"]
    messages: list[tuple[str, str]] = []
    size = state_space_size(capacities)

    if size > 1_000_000:
        messages.append(("warning", f"O espaço pode chegar a {size:,} estados; considere reduzir capacidades ou limite.".replace(",", " ")))

    if goal.mode == "selected_sum" and not goal.selected_indices:
        messages.append(("error", "Seleccione pelo menos uma jarra para o objectivo por soma."))

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
    # Guardo a configuração num formato simples para saber se o utilizador mudou algo.
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


def build_algorithm_report(runs: dict[str, SearchResult]) -> tuple[str, list[dict[str, Any]]] | tuple[None, list[dict[str, Any]]]:
    # Crio o texto e a tabela do relatório quando já existem resultados suficientes.
    successful_runs = {label: result for label, result in runs.items() if result.success}
    if len(successful_runs) < 2:
        return None, []

    fastest_steps_label, fastest_steps = min(successful_runs.items(), key=lambda item: len(item[1].actions))
    lowest_volume_label, lowest_volume = min(successful_runs.items(), key=lambda item: item[1].total_volume_moved)
    lowest_nodes_label, lowest_nodes = min(successful_runs.items(), key=lambda item: item[1].nodes_explored)
    fastest_time_label, fastest_time = min(successful_runs.items(), key=lambda item: item[1].elapsed_ms)

    report_text = (
        f"Para este mesmo problema, {fastest_steps_label} apresentou a menor sequência "
        f"({len(fastest_steps.actions)} passos). "
        f"{lowest_volume_label} movimentou menos água ({lowest_volume.total_volume_moved}L). "
        f"{lowest_nodes_label} explorou menos nós ({lowest_nodes.nodes_explored}). "
        f"{fastest_time_label} teve o menor tempo medido ({fastest_time.elapsed_ms:.2f} ms)."
    )

    rows = []
    for label, result in runs.items():
        rows.append(
            {
                "Algoritmo": label,
                "Passos": len(result.actions) if result.success else "sem solução",
                "Volume": f"{result.total_volume_moved}L" if result.success else "-",
                "Nós explorados": f"{result.nodes_explored:,}".replace(",", " "),
                "Tempo": f"{result.elapsed_ms:.2f} ms",
                "Resultado": "Solução encontrada" if result.success else result.message,
            }
        )

    return report_text, rows
