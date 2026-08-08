"""Strict StableHLO text frontend for schedule-free semantic features."""

from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .contracts import ContractError

_SSA = r"%[A-Za-z0-9_.$-]+"
_DTYPE_BYTES = {
    "i1": 1,
    "i8": 1,
    "ui8": 1,
    "si8": 1,
    "f8e4m3fn": 1,
    "f8e5m2": 1,
    "i16": 2,
    "f16": 2,
    "bf16": 2,
    "i32": 4,
    "f32": 4,
    "i64": 8,
    "f64": 8,
}
_CONTRACTIONS = {"dot", "dot_general", "convolution"}
_REDUCTIONS = {"reduce", "reduce_window"}
_MOVEMENT = {
    "broadcast_in_dim",
    "concatenate",
    "dynamic_broadcast_in_dim",
    "dynamic_reshape",
    "dynamic_slice",
    "dynamic_update_slice",
    "gather",
    "pad",
    "reshape",
    "reverse",
    "scatter",
    "slice",
    "transpose",
}
_EXPENSIVE = {
    "cbrt": 4.0,
    "cosine": 4.0,
    "divide": 4.0,
    "exponential": 4.0,
    "log": 4.0,
    "logistic": 6.0,
    "power": 8.0,
    "rsqrt": 4.0,
    "sine": 4.0,
    "sqrt": 4.0,
    "tanh": 6.0,
}
_UNSUPPORTED = {
    "after_all",
    "case",
    "custom_call",
    "infeed",
    "optimization_barrier",
    "outfeed",
    "recv",
    "send",
    "tuple",
    "while",
}
_ZERO_FLOP = _MOVEMENT | {"constant", "convert", "iota"}
NODE_FEATURE_DIM = 32


@dataclass(frozen=True)
class TensorType:
    shape: tuple[int, ...]
    dtype: str

    @property
    def elements(self) -> int:
        return math.prod(self.shape)

    @property
    def bytes(self) -> int:
        return self.elements * _DTYPE_BYTES[self.dtype]


@dataclass
class SemanticNode:
    name: str
    opcode: str
    operands: list[str]
    output: TensorType
    flops: float
    logical_bytes: float
    reduction_size: int = 0
    attributes: dict[str, Any] = field(default_factory=dict)
    is_argument: bool = False


@dataclass
class SemanticGraph:
    graph_id: str
    nodes: list[SemanticNode]
    edges: list[tuple[int, int]]
    outputs: list[str]
    total_flops: float
    logical_bytes: float
    minimum_io_bytes: float
    dtype: str

    def contraction_flops(self) -> float:
        return float(
            sum(node.flops for node in self.nodes if node.opcode in _CONTRACTIONS)
        )

    def other_flops(self) -> float:
        return max(0.0, float(self.total_flops) - self.contraction_flops())

    def feature_tensors(self) -> tuple[list[list[float]], list[list[float]]]:
        features = [_node_features(node) for node in self.nodes]
        adjacency = [[0.0] * len(self.nodes) for _ in self.nodes]
        for source, destination in self.edges:
            adjacency[destination][source] = 1.0
        return features, adjacency


def _parse_tensor(text: str) -> TensorType:
    match = re.fullmatch(r"tensor<([^<>]+)>", text.strip())
    if not match:
        raise ContractError(f"unsupported tensor type {text!r}")
    body = match.group(1)
    if "?" in body or "*" in body:
        raise ContractError(f"dynamic or unranked tensor unsupported: {text}")
    parts = body.split("x")
    dtype = parts[-1]
    if dtype not in _DTYPE_BYTES:
        raise ContractError(f"unsupported dtype {dtype!r}")
    try:
        shape = tuple(int(value) for value in parts[:-1])
    except ValueError as error:
        raise ContractError(f"invalid tensor shape {text!r}") from error
    return TensorType(shape, dtype)


def _tensor_types(text: str) -> list[TensorType]:
    return [_parse_tensor(value) for value in re.findall(r"tensor<[^<>]+>", text)]


def _operation_statements(body: str) -> list[str]:
    statements: list[str] = []
    current = ""
    for raw in body.splitlines():
        line = raw.strip()
        if not line:
            continue
        if re.match(rf"^{_SSA}\s*=", line) or re.match(r"^(?:func\.)?return\b", line):
            if current:
                statements.append(current)
            current = line
        elif current:
            current += " " + line
        else:
            raise ContractError(f"unsupported function statement {line!r}")
    if current:
        statements.append(current)
    return statements


def _functions(text: str) -> list[tuple[str, str, str]]:
    pattern = r"\bfunc\.func(?:\s+(?:public|private|nested))?\s+@([A-Za-z0-9_.$-]+)\s*\("
    functions: list[tuple[str, str, str]] = []
    for match in re.finditer(pattern, text):
        args_start = text.find("(", match.start())
        args_end = _matching(text, args_start, "(", ")")
        body_start = _function_body_start(text, args_end + 1)
        if body_start < 0:
            raise ContractError(f"function {match.group(1)} body is missing")
        body_end = _matching(text, body_start, "{", "}")
        functions.append(
            (match.group(1), text[args_start + 1 : args_end], text[body_start + 1 : body_end])
        )
    if not functions:
        raise ContractError("expected at least one func.func")
    return functions


def _function_body_start(text: str, start: int) -> int:
    levels = {"(": 0, "[": 0, "<": 0}
    closing = {")": "(", "]": "[", ">": "<"}
    previous = ""
    for index in range(start, len(text)):
        char = text[index]
        if char in levels:
            levels[char] += 1
        elif char in closing and not (char == ">" and previous == "-"):
            levels[closing[char]] = max(0, levels[closing[char]] - 1)
        elif char == "{" and all(value == 0 for value in levels.values()):
            return index
        previous = char
    return -1


def _matching(text: str, start: int, opening: str, closing: str) -> int:
    depth = 0
    quoted = False
    for index in range(start, len(text)):
        char = text[index]
        if char == '"':
            quoted = not quoted
        elif not quoted and char == opening:
            depth += 1
        elif not quoted and char == closing:
            depth -= 1
            if depth == 0:
                return index
    raise ContractError(f"unclosed {opening}")


def _arguments(text: str) -> dict[str, TensorType]:
    arguments: dict[str, TensorType] = {}
    for match in re.finditer(rf"({_SSA})\s*:\s*(tensor<[^<>]+>)", text):
        arguments[match.group(1)] = _parse_tensor(match.group(2))
    if not arguments and text.strip():
        raise ContractError("unable to parse function arguments")
    return arguments


def _output_type(statement: str) -> TensorType:
    arrow_types = re.findall(r"->\s*(tensor<[^<>]+>)", statement)
    if arrow_types:
        return _parse_tensor(arrow_types[-1])
    types = _tensor_types(statement)
    if not types:
        raise ContractError("operation has no ranked tensor result")
    return types[-1]


def _operand_syntax(payload: str) -> str:
    parts = re.split(r"\s:\s*(?=\(|tensor<)", payload)
    return parts[0]


def _estimate_flops(
    opcode: str,
    output: TensorType,
    operands: list[TensorType],
    statement: str,
) -> tuple[float, int]:
    if opcode in _ZERO_FLOP:
        return 0.0, 0
    if opcode in _CONTRACTIONS:
        if len(operands) < 2:
            raise ContractError(f"{opcode} requires two tensor operands")
        lhs = operands[0]
        contracting = re.search(r"contracting_dims\s*=\s*\[([0-9]+)", statement)
        axis = int(contracting.group(1)) if contracting else len(lhs.shape) - 1
        if axis < 0 or axis >= len(lhs.shape):
            raise ContractError(f"invalid contraction dimension for {opcode}")
        return float(2 * output.elements * lhs.shape[axis]), 0
    if opcode in _REDUCTIONS:
        input_elements = operands[0].elements if operands else output.elements
        reduction_size = max(1, input_elements // max(1, output.elements))
        return float(input_elements), reduction_size
    return float(output.elements) * _EXPENSIVE.get(opcode, 1.0), 0


def parse_stablehlo(text: str, graph_id: str | None = None) -> SemanticGraph:
    if not text.strip():
        raise ContractError("StableHLO text is empty")
    functions = _functions(text)
    main = next((function for function in functions if function[0] == "main"), functions[0])
    nodes: list[SemanticNode] = []
    function_arguments: dict[str, list[str]] = {}
    pending_calls: list[tuple[str, str]] = []
    main_outputs: list[str] = []

    for function_name, args_text, body in functions:
        values = _arguments(args_text)
        namespace = f"@{function_name}:"
        function_arguments[function_name] = []
        for argument_name, tensor in values.items():
            qualified = namespace + argument_name
            function_arguments[function_name].append(qualified)
            nodes.append(
                SemanticNode(
                    name=qualified,
                    opcode="argument",
                    operands=[],
                    output=tensor,
                    flops=0.0,
                    logical_bytes=float(tensor.bytes),
                    is_argument=True,
                )
            )
        outputs: list[str] = []
        for statement in _operation_statements(body):
            if re.match(r"^(?:func\.)?return\b", statement):
                outputs = [namespace + value for value in re.findall(_SSA, statement.split(":", 1)[0])]
                continue
            stable_match = re.match(
                rf"^({_SSA})\s*=\s*(?:\"?(stablehlo|chlo)\.([A-Za-z0-9_]+)\"?)\s*(.*)",
                statement,
            )
            call_match = re.match(
                rf"^({_SSA})\s*=\s*(?:func\.)?call\s+@([A-Za-z0-9_.$-]+)\s*(.*)",
                statement,
            )
            if stable_match:
                result, dialect, raw_opcode, payload = stable_match.groups()
                opcode = raw_opcode if dialect == "stablehlo" else f"{dialect}.{raw_opcode}"
                if raw_opcode in _UNSUPPORTED or "region" in payload or "^bb" in payload:
                    raise ContractError(f"unsupported StableHLO operation: {raw_opcode}")
                call_target = None
            elif call_match:
                result, call_target, payload = call_match.groups()
                opcode = "call"
            else:
                raise ContractError(f"unsupported operation syntax: {statement[:120]}")

            local_operands = list(dict.fromkeys(re.findall(_SSA, _operand_syntax(payload))))
            missing = [operand for operand in local_operands if operand not in values]
            if missing:
                raise ContractError(f"{result} references unknown operands {missing}")
            qualified_operands = [namespace + operand for operand in local_operands]
            output = _output_type(statement)
            operand_types = [values[operand] for operand in local_operands]
            flops, reduction_size = _estimate_flops(
                raw_opcode if stable_match else "call", output, operand_types, statement
            )
            logical_bytes = output.bytes + sum(value.bytes for value in operand_types)
            qualified_result = namespace + result
            nodes.append(
                SemanticNode(
                    name=qualified_result,
                    opcode=opcode,
                    operands=qualified_operands,
                    output=output,
                    flops=flops,
                    logical_bytes=float(logical_bytes),
                    reduction_size=reduction_size,
                    attributes={"syntax": _operand_syntax(payload).strip()},
                )
            )
            if call_target:
                pending_calls.append((qualified_result, call_target))
            values[result] = output
        if not outputs:
            raise ContractError(f"function {function_name} has no return")
        if function_name == main[0]:
            main_outputs = outputs

    node_index = {node.name: index for index, node in enumerate(nodes)}
    edges = [
        (node_index[operand], node_index[node.name])
        for node in nodes
        for operand in node.operands
    ]
    edges.extend(
        (node_index[source], node_index[destination])
        for source, target in pending_calls
        for destination in function_arguments.get(target, [])
        if source in node_index and destination in node_index
    )
    main_args = _arguments(main[1])
    output_bytes = sum(
        node.output.bytes for node in nodes if node.name in set(main_outputs)
    )
    input_bytes = sum(tensor.bytes for tensor in main_args.values())
    compute_nodes = [node for node in nodes if not node.is_argument]
    dtype = max(
        (node.output.dtype for node in compute_nodes),
        key=lambda value: sum(
            node.output.elements for node in compute_nodes if node.output.dtype == value
        ),
    )
    return SemanticGraph(
        graph_id=graph_id or main[0],
        nodes=nodes,
        edges=edges,
        outputs=main_outputs,
        total_flops=sum(node.flops for node in compute_nodes),
        logical_bytes=sum(node.logical_bytes for node in compute_nodes),
        minimum_io_bytes=float(input_bytes + output_bytes),
        dtype=dtype,
    )


def parse_stablehlo_file(path: str | Path, graph_id: str | None = None) -> SemanticGraph:
    return parse_stablehlo(Path(path).read_text(encoding="utf-8"), graph_id)


def _hash_bucket(value: str, buckets: int) -> int:
    return int.from_bytes(hashlib.blake2b(value.encode(), digest_size=4).digest(), "little") % buckets


def _node_features(node: SemanticNode) -> list[float]:
    opcode = node.opcode
    category = [
        float(opcode in _CONTRACTIONS),
        float(opcode in _REDUCTIONS),
        float(opcode in _MOVEMENT),
        float(opcode not in _CONTRACTIONS | _REDUCTIONS | _MOVEMENT),
        float(node.is_argument),
    ]
    op_hash = [0.0] * 8
    op_hash[_hash_bucket(opcode, len(op_hash))] = 1.0
    dtype_hash = [0.0] * 5
    dtype_hash[_hash_bucket(node.output.dtype, len(dtype_hash))] = 1.0
    dimensions = [math.log1p(value) for value in node.output.shape[:4]]
    dimensions += [0.0] * (4 - len(dimensions))
    numeric = [
        math.log1p(node.output.elements),
        math.log1p(node.output.bytes),
        len(node.output.shape) / 8.0,
        *dimensions,
        math.log1p(node.flops),
        math.log1p(node.logical_bytes),
        math.log1p(node.reduction_size),
        math.log1p(len(node.operands)),
        float(opcode in _EXPENSIVE),
        float(not node.output.shape),
        1.0,
    ]
    result = category + op_hash + dtype_hash + numeric
    if len(result) != NODE_FEATURE_DIM:
        raise AssertionError(f"node feature contract changed: {len(result)}")
    return result

