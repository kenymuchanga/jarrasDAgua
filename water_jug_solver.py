from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import heapq
import math
import time
from itertools import combinations
from typing import Iterable, Literal


State = tuple[int, ...]
Algorithm = Literal["bfs", "dfs", "ucs", "astar"]
CostMode = Literal["operations", "volume"]
GoalMode = Literal["any_jug", "selected_sum", "any_combination", "exact_state"]


@dataclass(frozen=True)
class GoalSpec:
    # Esta classe guarda o tipo de objectivo que queremos atingir.
    mode: GoalMode
    target: int = 0
    selected_indices: tuple[int, ...] = ()
    exact_state: State = ()


@dataclass(frozen=True)
class Action:
    # Esta classe representa uma operação feita numa jarra.
    kind: Literal["fill", "empty", "transfer"]
    source: int | None
    target: int | None
    amount: int
    label: str


@dataclass(frozen=True)
class SearchResult:
    # Aqui guardo tudo que a busca devolve para a interface mostrar.
    success: bool
    states: list[State]
    actions: list[Action]
    total_cost: float
    total_volume_moved: int
    nodes_explored: int
    elapsed_ms: float
    message: str
    algorithm: Algorithm
    cost_mode: CostMode


@dataclass(frozen=True)
class ReachabilityAnalysis:
    possible: bool | None
    checked_states: int
    elapsed_ms: float
    reason: str
    witness_state: State | None = None


@dataclass(frozen=True)
class SearchTreeNode:
    id: int
    state: State
    depth: int
    is_goal: bool


@dataclass(frozen=True)
class SearchTreeEdge:
    source_id: int
    target_id: int
    action: Action


@dataclass(frozen=True)
class SearchTree:
    nodes: list[SearchTreeNode]
    edges: list[SearchTreeEdge]
    truncated: bool
    max_nodes: int
    max_depth: int
    goal_node_id: int | None = None


def state_space_size(capacities: Iterable[int]) -> int:
    total = 1
    for capacity in capacities:
        total *= capacity + 1
    return total


def is_goal(state: State, goal: GoalSpec) -> bool:
    if goal.mode == "any_jug":
        return any(value == goal.target for value in state)

    if goal.mode == "selected_sum":
        indices = goal.selected_indices or tuple(range(len(state)))
        return sum(state[index] for index in indices) == goal.target

    if goal.mode == "any_combination":
        return any(
            sum(state[index] for index in subset) == goal.target
            for subset in _non_empty_subsets(len(state))
        )

    if goal.mode == "exact_state":
        return state == goal.exact_state

    raise ValueError(f"Modo de objectivo desconhecido: {goal.mode}")


def generate_successors(state: State, capacities: State) -> Iterable[tuple[State, Action]]:
    # A partir de um estado, gero todas as acções possíveis:
    # encher, esvaziar e transferir água entre jarras.
    number_of_jugs = len(state)

    for index, current in enumerate(state):
        capacity = capacities[index]

        if current < capacity:
            moved = capacity - current
            next_state = list(state)
            next_state[index] = capacity
            yield (
                tuple(next_state),
                Action(
                    kind="fill",
                    source=None,
                    target=index,
                    amount=moved,
                    label=f"Encher {moved}L na J{index + 1}",
                ),
            )

        if current > 0:
            moved = current
            next_state = list(state)
            next_state[index] = 0
            yield (
                tuple(next_state),
                Action(
                    kind="empty",
                    source=index,
                    target=None,
                    amount=moved,
                    label=f"Esvaziar J{index + 1} despejando {moved}L",
                ),
            )

    for source in range(number_of_jugs):
        if state[source] == 0:
            continue

        for target in range(number_of_jugs):
            if source == target or state[target] == capacities[target]:
                continue

            moved = min(state[source], capacities[target] - state[target])
            if moved <= 0:
                continue

            next_state = list(state)
            next_state[source] -= moved
            next_state[target] += moved
            yield (
                tuple(next_state),
                Action(
                    kind="transfer",
                    source=source,
                    target=target,
                    amount=moved,
                    label=f"Transferir {moved}L da J{source + 1} para J{target + 1}",
                ),
            )


def solve_water_jugs(
    capacities: Iterable[int],
    goal: GoalSpec,
    *,
    initial_state: Iterable[int] | None = None,
    algorithm: Algorithm = "bfs",
    cost_mode: CostMode = "operations",
    max_nodes: int = 50_000,
) -> SearchResult:
    # Esta é a função principal da lógica.
    # Ela escolhe o algoritmo e tenta encontrar uma solução.
    capacities_state = tuple(int(value) for value in capacities)
    initial = tuple(int(value) for value in (initial_state or [0] * len(capacities_state)))
    started = time.perf_counter()

    _validate_problem(capacities_state, initial, goal, algorithm, cost_mode, max_nodes)

    if algorithm in {"bfs", "dfs"}:
        result = _solve_unweighted(
            capacities_state,
            initial,
            goal,
            algorithm=algorithm,
            cost_mode=cost_mode,
            max_nodes=max_nodes,
        )
    else:
        result = _solve_weighted(
            capacities_state,
            initial,
            goal,
            algorithm=algorithm,
            cost_mode=cost_mode,
            max_nodes=max_nodes,
        )

    elapsed_ms = (time.perf_counter() - started) * 1000
    return SearchResult(
        success=result.success,
        states=result.states,
        actions=result.actions,
        total_cost=result.total_cost,
        total_volume_moved=result.total_volume_moved,
        nodes_explored=result.nodes_explored,
        elapsed_ms=elapsed_ms,
        message=result.message,
        algorithm=algorithm,
        cost_mode=cost_mode,
    )


def analyse_reachability(
    capacities: Iterable[int],
    goal: GoalSpec,
    *,
    initial_state: Iterable[int] | None = None,
    max_nodes: int = 200_000,
) -> ReachabilityAnalysis:
    # Esta função tenta descobrir se o objectivo é possível antes de mostrar a solução.
    capacities_state = tuple(int(value) for value in capacities)
    initial = tuple(int(value) for value in (initial_state or [0] * len(capacities_state)))
    started = time.perf_counter()

    _validate_problem(capacities_state, initial, goal, "bfs", "operations", max_nodes)

    quick_reason = _quick_impossibility_reason(capacities_state, initial, goal)
    if quick_reason:
        return ReachabilityAnalysis(
            possible=False,
            checked_states=0,
            elapsed_ms=(time.perf_counter() - started) * 1000,
            reason=quick_reason,
        )

    if is_goal(initial, goal):
        return ReachabilityAnalysis(
            possible=True,
            checked_states=1,
            elapsed_ms=(time.perf_counter() - started) * 1000,
            reason="Objectivo já satisfeito no estado inicial.",
            witness_state=initial,
        )

    frontier: deque[State] = deque([initial])
    visited = {initial}

    while frontier and len(visited) <= max_nodes:
        state = frontier.popleft()
        for next_state, _ in generate_successors(state, capacities_state):
            if next_state in visited:
                continue
            visited.add(next_state)
            if is_goal(next_state, goal):
                return ReachabilityAnalysis(
                    possible=True,
                    checked_states=len(visited),
                    elapsed_ms=(time.perf_counter() - started) * 1000,
                    reason="Objectivo alcançável.",
                    witness_state=next_state,
                )
            if len(visited) > max_nodes:
                break
            frontier.append(next_state)

    if frontier:
        return ReachabilityAnalysis(
            possible=None,
            checked_states=len(visited),
            elapsed_ms=(time.perf_counter() - started) * 1000,
            reason=f"Não foi possível concluir a detecção automática dentro de {max_nodes:,} estados.",
        )

    return ReachabilityAnalysis(
        possible=False,
        checked_states=len(visited),
        elapsed_ms=(time.perf_counter() - started) * 1000,
        reason="Objectivo impossível: a busca exaustiva visitou todos os estados alcançáveis sem encontrar o objectivo.",
    )


def build_search_tree(
    capacities: Iterable[int],
    goal: GoalSpec,
    *,
    initial_state: Iterable[int] | None = None,
    max_nodes: int = 60,
    max_depth: int = 4,
) -> SearchTree:
    # Esta função cria uma árvore pequena da busca para mostrar no grafo.
    capacities_state = tuple(int(value) for value in capacities)
    initial = tuple(int(value) for value in (initial_state or [0] * len(capacities_state)))
    _validate_problem(capacities_state, initial, goal, "bfs", "operations", max_nodes)

    nodes = [SearchTreeNode(id=0, state=initial, depth=0, is_goal=is_goal(initial, goal))]
    edges: list[SearchTreeEdge] = []
    state_to_id = {initial: 0}
    frontier: deque[tuple[State, int]] = deque([(initial, 0)])
    truncated = False
    goal_node_id = 0 if nodes[0].is_goal else None

    while frontier:
        state, depth = frontier.popleft()
        if depth >= max_depth:
            truncated = True
            continue

        source_id = state_to_id[state]
        for next_state, action in generate_successors(state, capacities_state):
            if next_state in state_to_id:
                continue
            if len(nodes) >= max_nodes:
                truncated = True
                break

            node_id = len(nodes)
            node = SearchTreeNode(
                id=node_id,
                state=next_state,
                depth=depth + 1,
                is_goal=is_goal(next_state, goal),
            )
            nodes.append(node)
            edges.append(SearchTreeEdge(source_id=source_id, target_id=node_id, action=action))
            state_to_id[next_state] = node_id

            if node.is_goal and goal_node_id is None:
                goal_node_id = node_id
            frontier.append((next_state, depth + 1))

        if len(nodes) >= max_nodes:
            truncated = True
            break

    return SearchTree(
        nodes=nodes,
        edges=edges,
        truncated=truncated,
        max_nodes=max_nodes,
        max_depth=max_depth,
        goal_node_id=goal_node_id,
    )


def _solve_unweighted(
    capacities: State,
    initial: State,
    goal: GoalSpec,
    *,
    algorithm: Literal["bfs", "dfs"],
    cost_mode: CostMode,
    max_nodes: int,
) -> SearchResult:
    if is_goal(initial, goal):
        return _successful_result([initial], [], 0, algorithm, cost_mode, 0, "Objectivo já satisfeito.")

    frontier: deque[State] | list[State]
    if algorithm == "bfs":
        frontier = deque([initial])
    else:
        frontier = [initial]

    parents: dict[State, tuple[State, Action]] = {}
    visited = {initial}
    nodes_explored = 0

    while frontier and nodes_explored < max_nodes:
        if algorithm == "bfs":
            state = frontier.popleft()  # type: ignore[attr-defined]
        else:
            state = frontier.pop()  # type: ignore[attr-defined]

        nodes_explored += 1

        successors = list(generate_successors(state, capacities))
        if algorithm == "dfs":
            successors.reverse()

        for next_state, action in successors:
            if next_state in visited:
                continue

            parents[next_state] = (state, action)
            if is_goal(next_state, goal):
                states, actions = _reconstruct_path(initial, next_state, parents)
                return _successful_result(
                    states,
                    actions,
                    nodes_explored,
                    algorithm,
                    cost_mode,
                    _path_cost(actions, cost_mode),
                    "Solução encontrada.",
                )

            visited.add(next_state)
            if algorithm == "bfs":
                frontier.append(next_state)  # type: ignore[attr-defined]
            else:
                frontier.append(next_state)  # type: ignore[attr-defined]

    return _failure_result(initial, nodes_explored, algorithm, cost_mode, max_nodes)


def _solve_weighted(
    capacities: State,
    initial: State,
    goal: GoalSpec,
    *,
    algorithm: Literal["ucs", "astar"],
    cost_mode: CostMode,
    max_nodes: int,
) -> SearchResult:
    parents: dict[State, tuple[State, Action]] = {}
    best_cost: dict[State, float] = {initial: 0.0}
    frontier: list[tuple[float, int, float, State]] = []
    counter = 0
    initial_priority = _heuristic(initial, capacities, goal, cost_mode) if algorithm == "astar" else 0.0
    heapq.heappush(frontier, (initial_priority, counter, 0.0, initial))
    nodes_explored = 0

    while frontier and nodes_explored < max_nodes:
        _, _, current_cost, state = heapq.heappop(frontier)
        if current_cost > best_cost.get(state, math.inf):
            continue

        nodes_explored += 1

        if is_goal(state, goal):
            states, actions = _reconstruct_path(initial, state, parents)
            return _successful_result(
                states,
                actions,
                nodes_explored,
                algorithm,
                cost_mode,
                current_cost,
                "Solução encontrada.",
            )

        for next_state, action in generate_successors(state, capacities):
            next_cost = current_cost + _action_cost(action, cost_mode)
            if next_cost >= best_cost.get(next_state, math.inf):
                continue

            parents[next_state] = (state, action)
            best_cost[next_state] = next_cost
            counter += 1
            priority = next_cost
            if algorithm == "astar":
                priority += _heuristic(next_state, capacities, goal, cost_mode)
            heapq.heappush(frontier, (priority, counter, next_cost, next_state))

    return _failure_result(initial, nodes_explored, algorithm, cost_mode, max_nodes)


def _validate_problem(
    capacities: State,
    initial: State,
    goal: GoalSpec,
    algorithm: Algorithm,
    cost_mode: CostMode,
    max_nodes: int,
) -> None:
    if len(capacities) < 2:
        raise ValueError("Use pelo menos dois jarros.")
    if len(capacities) != len(initial):
        raise ValueError("O estado inicial deve ter o mesmo número de jarros que as capacidades.")
    if any(capacity <= 0 for capacity in capacities):
        raise ValueError("Todas as capacidades devem ser maiores que zero.")
    if any(value < 0 for value in initial):
        raise ValueError("O estado inicial não pode ter valores negativos.")
    if any(value > capacities[index] for index, value in enumerate(initial)):
        raise ValueError("Nenhum jarro pode começar acima da sua capacidade.")
    if algorithm not in {"bfs", "dfs", "ucs", "astar"}:
        raise ValueError("Algoritmo de busca inválido.")
    if cost_mode not in {"operations", "volume"}:
        raise ValueError("Modo de custo inválido.")
    if max_nodes <= 0:
        raise ValueError("O limite de nós deve ser maior que zero.")
    if goal.mode == "selected_sum":
        if any(index < 0 or index >= len(capacities) for index in goal.selected_indices):
            raise ValueError("A combinação seleccionada contém um jarro inválido.")
    if goal.mode == "exact_state":
        if len(goal.exact_state) != len(capacities):
            raise ValueError("O estado exacto deve indicar uma quantidade para cada jarro.")
        if any(value < 0 for value in goal.exact_state):
            raise ValueError("O objectivo exacto não pode ter valores negativos.")
        if any(value > capacities[index] for index, value in enumerate(goal.exact_state)):
            raise ValueError("O objectivo exacto não pode exceder as capacidades.")


def _quick_impossibility_reason(capacities: State, initial: State, goal: GoalSpec) -> str | None:
    common_divisor = 0
    for capacity in capacities:
        common_divisor = math.gcd(common_divisor, capacity)

    if goal.mode == "any_jug":
        if goal.target > max(capacities):
            return f"Objectivo impossível: nenhum jarro tem capacidade para {goal.target} L."
        if _starts_empty(initial) and common_divisor > 1 and goal.target % common_divisor != 0:
            return f"Objectivo impossível: {goal.target} L não é múltiplo do MDC {common_divisor} L das capacidades."

    if goal.mode == "selected_sum":
        indices = goal.selected_indices or tuple(range(len(capacities)))
        if goal.target > sum(capacities[index] for index in indices):
            return "Objectivo impossível: a soma pretendida excede a capacidade dos jarros seleccionados."
        if _starts_empty(initial) and common_divisor > 1 and goal.target % common_divisor != 0:
            return f"Objectivo impossível: {goal.target} L não é múltiplo do MDC {common_divisor} L das capacidades."

    if goal.mode == "any_combination":
        if goal.target > sum(capacities):
            return "Objectivo impossível: a quantidade pretendida excede a capacidade total dos jarros."
        if _starts_empty(initial) and common_divisor > 1 and goal.target % common_divisor != 0:
            return f"Objectivo impossível: {goal.target} L não é múltiplo do MDC {common_divisor} L das capacidades."

    if goal.mode == "exact_state":
        if _starts_empty(initial) and common_divisor > 1:
            impossible_values = [value for value in goal.exact_state if value % common_divisor != 0]
            if impossible_values:
                return f"Objectivo impossível: todas as quantidades alcançáveis são múltiplos de {common_divisor} L."

    return None


def _starts_empty(initial: State) -> bool:
    return all(value == 0 for value in initial)


def _reconstruct_path(
    initial: State,
    goal_state: State,
    parents: dict[State, tuple[State, Action]],
) -> tuple[list[State], list[Action]]:
    states = [goal_state]
    actions: list[Action] = []
    current = goal_state

    while current != initial:
        previous, action = parents[current]
        actions.append(action)
        states.append(previous)
        current = previous

    states.reverse()
    actions.reverse()
    return states, actions


def _successful_result(
    states: list[State],
    actions: list[Action],
    nodes_explored: int,
    algorithm: Algorithm,
    cost_mode: CostMode,
    total_cost: float,
    message: str,
) -> SearchResult:
    return SearchResult(
        success=True,
        states=states,
        actions=actions,
        total_cost=total_cost,
        total_volume_moved=sum(action.amount for action in actions),
        nodes_explored=nodes_explored,
        elapsed_ms=0,
        message=message,
        algorithm=algorithm,
        cost_mode=cost_mode,
    )


def _failure_result(
    initial: State,
    nodes_explored: int,
    algorithm: Algorithm,
    cost_mode: CostMode,
    max_nodes: int,
) -> SearchResult:
    limit_message = (
        f"Não foi encontrada solução dentro do limite de {max_nodes:,} nós explorados."
        if nodes_explored >= max_nodes
        else "Não foi encontrada solução para esta configuração."
    )
    return SearchResult(
        success=False,
        states=[initial],
        actions=[],
        total_cost=0,
        total_volume_moved=0,
        nodes_explored=nodes_explored,
        elapsed_ms=0,
        message=limit_message,
        algorithm=algorithm,
        cost_mode=cost_mode,
    )


def _path_cost(actions: Iterable[Action], cost_mode: CostMode) -> float:
    return sum(_action_cost(action, cost_mode) for action in actions)


def _action_cost(action: Action, cost_mode: CostMode) -> float:
    if cost_mode == "operations":
        return 1.0
    return float(action.amount)


def _heuristic(state: State, capacities: State, goal: GoalSpec, cost_mode: CostMode) -> float:
    distance = _goal_distance(state, goal)
    if distance == 0:
        return 0.0

    if cost_mode == "volume":
        if goal.mode == "exact_state":
            return distance / 2
        return float(distance)

    max_single_move = max(capacities)
    if goal.mode == "exact_state":
        max_single_move *= 2
    return math.ceil(distance / max_single_move)


def _goal_distance(state: State, goal: GoalSpec) -> int:
    if is_goal(state, goal):
        return 0

    if goal.mode == "any_jug":
        return min(abs(value - goal.target) for value in state)

    if goal.mode == "selected_sum":
        indices = goal.selected_indices or tuple(range(len(state)))
        return abs(sum(state[index] for index in indices) - goal.target)

    if goal.mode == "any_combination":
        return min(
            abs(sum(state[index] for index in subset) - goal.target)
            for subset in _non_empty_subsets(len(state))
        )

    if goal.mode == "exact_state":
        return sum(abs(value - goal.exact_state[index]) for index, value in enumerate(state))

    return 0


def _non_empty_subsets(size: int) -> Iterable[tuple[int, ...]]:
    indices = tuple(range(size))
    for length in range(1, size + 1):
        yield from combinations(indices, length)
