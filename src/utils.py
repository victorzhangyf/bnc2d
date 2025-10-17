import logging, sys

_logger = logging.getLogger("compile_bnc")

import subprocess, tempfile, pathlib, enum, itertools, pickle, time, functools
import dataclasses, json

import xml.etree.ElementTree as ET
from pyparsing import (CharsNotIn, Group, OneOrMore, Optional, Suppress, Word,
                       ZeroOrMore, alphanums, alphas, cppStyleComment, nums,
                       printables, pyparsing_common, Literal, Regex)
import global_config

config = global_config.Config()

import networkx as nx
import numpy as np
import graphviz


ACE_EVALUATE_BINARY = "./ace_v3.0_linux86/evaluate.sh"
ACE_COMPILE_BINARY = "./ace_v3.0_linux86/compile.sh"

CHECK_ZERO = False
CNF_VAR_OFFSET = 1  # CNF vars start from 1


class BNNodeType(enum.Flag):
    LEAF = enum.auto()
    INTERNAL = enum.auto()
    ROOT = enum.auto()


@dataclasses.dataclass
class BayesNet:
    network_name: str
    states: dict[str, list[str]]
    parents: dict[str, list[str]]
    cpts: dict[str, np.ndarray] = dataclasses.field(compare=False)
    G: nx.DiGraph = dataclasses.field(init=False)
    ordered_var_names: tuple[str, ...] = dataclasses.field(init=False)
    name_to_idx: dict[str, int] = dataclasses.field(init=False)

    def __post_init__(self) -> None:
        self.reorder_indices_with_ordeded_vars(tuple(sorted(
            self.all_var_strs)))

    def reorder_indices_with_ordeded_vars(
            self, ordered_var_names: tuple[str, ...]) -> None:
        self.ordered_var_names = ordered_var_names
        self.name_to_idx = {v: i for i, v in enumerate(ordered_var_names)}
        self.G = nx.DiGraph()
        for v, parents in self.parents.items():
            idx = self.name_to_idx[v]
            node_attr = {}
            if v in self.leaf_var_strs:
                node_attr['color'] = 'green'
            elif v in self.root_var_strs:
                node_attr["color"] = 'red'
            self.G.add_node(idx, **node_attr)
            for p in parents:
                p_idx = self.name_to_idx[p]
                self.G.add_edge(p_idx, idx)

        if config.DEBUG_LVL > 1:
            components = list(nx.weakly_connected_components(self.G))
            assert len(components) == 1

    @property
    def leaf_var_strs(self) -> frozenset[str]:
        var_set = set(self.parents.keys())
        for pars in self.parents.values():
            var_set -= set(pars)
        return frozenset(var_set)

    @property
    def root_var_strs(self) -> frozenset[str]:
        return frozenset([v for v, pars in self.parents.items() if not pars])

    @property
    def all_var_strs(self) -> frozenset[str]:
        return frozenset(self.parents.keys())

    def to_junction_tree(self, k=1) -> tuple[int, nx.Graph]:
        graphs = []
        for _ in range(k):
            # return nx.algorithms.tree.decomposition.junction_tree(self.G)
            clique_graph = nx.Graph()
            moral_G = nx.algorithms.moral.moral_graph(self.G)
            chordal_graph, _ = nx.algorithms.chordal.complete_to_chordal_graph(
                moral_G)

            cliques = [
                tuple(sorted(i)) for i in
                nx.algorithms.chordal.chordal_graph_cliques(chordal_graph)
            ]
            clique_graph.add_nodes_from(cliques, type="clique")

            for edge in itertools.combinations(cliques, 2):
                set_edge_0 = set(edge[0])
                set_edge_1 = set(edge[1])
                if not set_edge_0.isdisjoint(set_edge_1):
                    sepset = tuple(sorted(set_edge_0.intersection(set_edge_1)))
                    clique_graph.add_edge(edge[0],
                                          edge[1],
                                          weight=len(sepset),
                                          sepset=sepset)
            junction_tree = nx.maximum_spanning_tree(clique_graph)
            width = max(len(n) for n in junction_tree.nodes)
            graphs.append((width, junction_tree))
        return min(graphs, key=lambda x: x[0])

    def graphviz_render(self, name: str, out_dir: pathlib.Path) -> None:
        pdf_file = out_dir / (name + ".pdf")
        dot_file = out_dir / (name + ".gv")
        node_labels = {}
        for idx in self.G.nodes:
            node_labels[idx] = f"{self.ordered_var_names[idx]}({idx})"
        relabeled_G = nx.relabel_nodes(self.G, node_labels, copy=True)
        nx.drawing.nx_pydot.write_dot(relabeled_G, dot_file)
        graphviz.render('dot', filepath=dot_file, outfile=pdf_file)

    def to_hugin_str(self) -> str:
        lines = ["net", "{", "}"]
        for var_name, states in self.states.items():
            lines.extend([f"node {var_name}", "{"])
            states_quoted = [f'\"{s}\"' for s in states]
            lines.append(f"  states = ( {' '.join(states_quoted)} );")
            lines.append("}")
        for var_name, parents in self.parents.items():
            parents_str = ""
            if parents:
                parents_str = f" | {' '.join(parents)} "
            lines.append(f"potential ( {var_name} {parents_str})")
            lines.append("{")
            cpt = self.cpts[var_name]
            cpt_str = np.array2string(
                cpt,
                separator=" ",
                max_line_width=1000,
                formatter={'float_kind': lambda x: "%.8f" % x})
            cpt_str = cpt_str.replace('[', '(').replace(']', ')')
            lines.append(f"  data = {cpt_str} ;")
            lines.append("}")
        lines.append("")
        return "\n".join(lines)

    def ace_compile_then_eval(self, evidences: list[dict]) -> list[float]:
        network_filename = None
        try:
            hugin_str = self.to_hugin_str()
            with tempfile.NamedTemporaryFile(mode='w',
                                             suffix='.net',
                                             delete=False) as tmp_file:
                tmp_file.write(hugin_str)
                network_filename = tmp_file.name
            if config.DEBUG_LVL > 1:
                with open("./out/tmp.net", 'w') as f:
                    f.write(hugin_str)
            cmd_list: list[str] = [ACE_COMPILE_BINARY, network_filename]
            success, output = run_shell_command(' '.join(cmd_list))
            assert success, output
            rets = ace_eval(tmp_file.name, evidences=evidences)
        finally:
            if network_filename is not None:
                for suffix in ["", ".bmap", ".pmap", ".cnf"]:
                    pathlib.Path(network_filename +
                                 suffix).unlink(missing_ok=True)
        return rets

    def _is_dsep(self, x: frozenset[int], y: frozenset[int],
                 separator: frozenset[int]) -> bool:
        return nx.algorithms.d_separation.is_d_separator(
            self.G, x, y, separator)


@dataclasses.dataclass
class CNF:
    n_vars: int
    n_clauses: int
    n_eclaues: int
    clauses: tuple[tuple[int, ...], ...]

    def __post_init__(self) -> None:
        assert len(self.clauses) == self.n_clauses + self.n_eclaues

    @classmethod
    def parse_cnf(cls, cnf_content: str) -> "CNF":
        integer = pyparsing_common.signed_integer()
        comment_line = Regex(r'c.*')
        p_cnf_line = (Suppress("p cnf") + integer("var_count") +
                      integer("clause_count"))
        eclauses_line = (Suppress("eclauses") + integer("eclause_count"))

        clause_line = Group(
            OneOrMore(integer, stopOn=Literal("0")) + Suppress("0"))
        parser = (ZeroOrMore(Suppress(comment_line)) + p_cnf_line +
                  eclauses_line + OneOrMore(clause_line)("clauses"))

        result = parser.parseString(cnf_content)
        clauses = []
        for c in result.clauses:
            clauses.append(tuple(map(int, c)))
        cnf = CNF(n_vars=int(result.var_count),
                  n_clauses=int(result.clause_count),
                  n_eclaues=int(result.eclause_count),
                  clauses=tuple(clauses))
        return cnf


@dataclasses.dataclass
class C2DDTreeNode:
    # these are vars from the CNF
    # cnf vars index from 1
    idx: int
    variables: frozenset[int]
    children: tuple["C2DDTreeNode", "C2DDTreeNode"] | None = None
    parent: int = -1
    fam_idx: int = -1
    cutset: frozenset[int] = dataclasses.field(init=False)
    acut: frozenset[int] = dataclasses.field(init=False)
    cluster: frozenset[int] = dataclasses.field(init=False)
    is_leaf: bool = dataclasses.field(init=False)

    def __post_init__(self) -> None:
        if self.children is None:
            assert self.fam_idx >= 0
            self.is_leaf = True
        else:
            assert len(self.children) == 2
            assert self.fam_idx == -1
            self.is_leaf = False

    @property
    def context(self) -> frozenset[int]:
        return self.cluster & self.acut

    def __str__(self) -> str:
        children_str = f"Fam {self.fam_idx}"
        if not self.is_leaf:
            children_str = f"Children: {list(map(lambda c: c.idx, self.children))}"
        arr = [
            f"Index: {self.idx}", children_str, f"Cut: {sorted(self.cutset)}",
            f"Cluster: {sorted(self.cluster)}", f"Vars: {self.variables}",
            f"Parent: {self.parent}"
        ]
        return "  ".join(arr)

    def recur_set_acut_cutset_cluster(self,
                                      acut: frozenset[int] | None = None
                                      ) -> None:

        if self.children is None:
            assert acut
            self.acut = acut
            self.cutset = frozenset()
            self.cluster = self.variables
            assert len(self.cluster - acut) <= 1
            return

        child_l, child_r = self.children
        if acut is None:
            assert self.parent == -1
            self.cluster = child_l.variables & child_r.variables
            self.cutset = self.cluster
            self.acut = frozenset()
            new_acut = self.cluster
        else:
            shared_vars = child_l.variables & child_r.variables
            self.cutset = shared_vars - acut
            context = self.variables & acut
            self.cluster = self.cutset | context
            self.acut = acut
            new_acut = acut | self.cutset
        child_l.recur_set_acut_cutset_cluster(new_acut)
        child_r.recur_set_acut_cutset_cluster(new_acut)

    @classmethod
    def _cnf_to_dtree(cls, cnf: CNF,
                      c2d_args: dict[str, str]) -> "list[C2DDTreeNode]":
        dimacs_str = binary_cnf_to_eclause_DIMACS(cnf)
        cnf_filename = None
        try:
            with tempfile.NamedTemporaryFile(mode='w',
                                             suffix='.cnf',
                                             delete=False) as tmp_file:
                tmp_file.write(dimacs_str)
                cnf_filename = tmp_file.name
            if config.DEBUG_LVL > 1:
                with open("./out/tmp.cnf", 'w') as f:
                    f.write(dimacs_str)
            cmd_list: list[str] = [
                config.C2D_BINARY, "-in", cnf_filename, "-dt_out", "-silent"
            ]
            for k, v in c2d_args.items():
                cmd_list.append("-" + k)
                cmd_list.append(v)

            success, output = run_shell_command(' '.join(cmd_list))
            assert success, output
            node_list = cls._load_cnf_dtree(
                cnf_fpath=pathlib.Path(cnf_filename), y_idx=cnf.n_vars - 1)
        finally:
            if cnf_filename is not None:
                for suffix in ["", ".dtree", ".nnf"]:
                    pathlib.Path(cnf_filename + suffix).unlink(missing_ok=True)
        return node_list

    @staticmethod
    def _load_cnf_dtree(cnf_fpath: pathlib.Path,
                        y_idx: int) -> "list[C2DDTreeNode]":
        dt_file = cnf_fpath.with_suffix(".cnf.dtree")
        with open(cnf_fpath, 'r') as f:
            cnf_content = f.read()
        cnf = CNF.parse_cnf(cnf_content)
        with open(dt_file, 'r') as f:
            dt_lines = f.readlines()
        header_tks = dt_lines[0].split()
        assert header_tks[0] == "dtree"
        n_nodes = int(header_tks[1])
        assert n_nodes == cnf.n_clauses * 2 - 1
        node_list: list[C2DDTreeNode] = []
        for node_idx, line in enumerate(dt_lines[1:]):
            tks = line.split()
            nums = tuple(map(int, tks[1:]))
            if tks[0] == "L":
                assert len(nums) == 1
                fam_idx = nums[0]
                vs = frozenset(
                    map(lambda x: abs(x) - CNF_VAR_OFFSET,
                        cnf.clauses[fam_idx]))
                node = C2DDTreeNode(idx=node_idx,
                                    children=None,
                                    variables=vs,
                                    fam_idx=fam_idx)
            else:
                assert tks[0] == "I"
                assert len(nums) == 2
                l, r = nums
                child_l, child_r = node_list[l], node_list[r]
                l_has_y = y_idx in child_l.variables
                r_has_y = y_idx in child_r.variables
                if r_has_y and not l_has_y:
                    child_l, child_r = child_r, child_l
                child_l.parent = node_idx
                child_r.parent = node_idx
                vs = child_l.variables | child_r.variables
                node = C2DDTreeNode(idx=node_idx,
                                    children=(child_l, child_r),
                                    variables=vs)
            node_list.append(node)

        if config.DEBUG_LVL > 0:
            assert len(node_list) == n_nodes
            for n in node_list[:-1]:
                assert n.parent > 0

        root = node_list[-1]
        root.recur_set_acut_cutset_cluster()
        return node_list

    @staticmethod
    def gen_dtree(family_var_indices: tuple[tuple[int, ...], ...],
                  c2d_args: dict[str, str]) -> "list[C2DDTreeNode]":
        # each family: var index, then parent indices
        # each family XU: AND(U_i, ...) -> X
        n_vars = len(family_var_indices)
        clauses = []
        for var_indices in family_var_indices:
            var_idx = var_indices[0] + CNF_VAR_OFFSET
            parent_indices = [p + CNF_VAR_OFFSET for p in var_indices[1:]]
            clauses.append(tuple(-i for i in parent_indices) + (var_idx, ))
        cnf = CNF(n_vars, len(clauses), 0, tuple(clauses))
        node_list = C2DDTreeNode._cnf_to_dtree(cnf, c2d_args)
        node_list[-1]._check_dtree(frozenset(), family_var_indices)
        return node_list

    def get_width(self) -> int:
        assert self.parent == -1
        arr: list[tuple[int, int]] = []
        stack: list[C2DDTreeNode] = [self]
        while stack:
            node = stack.pop()
            if node.children:
                arr.append((len(node.cluster) - 1, node.idx))
                stack.extend(node.children)
        return sorted(arr, reverse=True)[0][0]

    def _check_dtree(
            self, acut: frozenset[int],
            family_var_indices: tuple[tuple[int, ...], ...]) -> frozenset[int]:
        # returns family indices in leafs
        if config.DEBUG_LVL < 1:
            return frozenset()
        assert self.idx >= 0
        assert self.cluster is not None
        assert self.acut is not None
        if self.is_leaf:
            assert len(self.variables - acut) <= 1
            expected_vars = frozenset(family_var_indices[self.fam_idx])
            assert expected_vars == self.variables
            return frozenset([self.fam_idx])
        else:
            assert self.children is not None
            child_l, child_r = self.children
            assert child_l.parent == self.idx
            assert child_l.parent == self.idx
            assert len({child_l.idx, child_r.idx, self.idx}) == 3

            assert not (self.cutset & acut)
            assert self.context == self.cluster - self.cutset

            assert self.variables == child_l.variables | child_r.variables
            shared_vars = child_l.variables & child_r.variables
            expected_cut = shared_vars - acut
            assert self.cutset == expected_cut
            new_acut = self.cutset | acut

            y_idx = len(family_var_indices) - 1
            if y_idx in child_r.variables:
                assert y_idx in child_l.variables

            l_leafs = child_l._check_dtree(new_acut, family_var_indices)
            r_leafs = child_r._check_dtree(new_acut, family_var_indices)
            assert not (l_leafs & r_leafs)
            combined_fams = l_leafs | r_leafs
            if self.parent == -1:
                assert len(combined_fams) == len(family_var_indices)
            return combined_fams


@dataclasses.dataclass
class NETReader:
    # src: pgmpy readwrite/NET.py
    def __init__(self, net_str: str) -> None:

        self.net_str = net_str

        if "/*" in self.net_str or "//" in self.net_str:
            self.net_str = cppStyleComment.suppress().transformString(
                self.net_str)  

        self.init_variable_grammar()
        self.init_probability_grammar()

        self.states: dict[str, list[str]] = self.parse_var_states()
        self.nary: dict[str, int] = {
            s: len(arr)
            for s, arr in self.states.items()
        }
        self.parents: dict[str, list[str]] = self.parse_parents()
        self.cpts: dict[str, np.ndarray] = self.parse_cpt_values()

        _DEBUG = 1
        if not _DEBUG:
            return

    def init_variable_grammar(self):
        word_expr = Word(alphanums + "_" + "-")("nodename")
        name_expr = Suppress("node ") + word_expr + Optional(Suppress("{"))

        word_expr2 = Word(initChars=printables,
                          excludeChars=["(", ")", ",", " "])
        state_expr = ZeroOrMore(word_expr2 + Optional(Suppress(",")))
        variable_state_expr = (Suppress("states") + Suppress("=") +
                               Suppress("(") +
                               Group(state_expr)("statenames") +
                               Suppress(")") + Suppress(";"))
        pexpr = Word(
            alphas.lower()) + Suppress("=") + CharsNotIn(";") + Suppress(";")
        property_expr = ZeroOrMore(pexpr)

        variable_property_expr = (Suppress("node ") +
                                  Word(alphanums + "_" + "-")("varname") +
                                  Suppress("{") +
                                  Group(property_expr)("properties") +
                                  Suppress("}"))

        self.name_expr, self.state_expr, self.property_expr = name_expr, variable_state_expr, variable_property_expr

    def init_probability_grammar(self):
        word_expr = Word(alphanums + "-" + "_") + Suppress(Optional("|"))

        potential_expr = (Suppress("potential") + Suppress("(") +
                          OneOrMore(word_expr) + Suppress(")"))

        num_expr = (Suppress(ZeroOrMore("(")) +
                    Word(nums + "-" + "+" + "e" + "E" + ".") +
                    Suppress(ZeroOrMore(")")))

        cpd_expr = Suppress("data") + Suppress("=") + OneOrMore(num_expr)

        self.potential_expr, self.cpd_expr = potential_expr, cpd_expr

    def parse_var_states(self):
        variable_states = {}
        for index, match in enumerate(self.name_expr.scanString(self.net_str)):
            result = match[0]
            name = result.nodename
            allstates = list(self.state_expr.scanString(self.net_str))
            states_unedited = list(
                allstates[index][0].statenames
            )
            states_edited = [
                state.replace('"', "") for state in states_unedited
            ]
            variable_states[name] = states_edited
        return variable_states

    def parse_parents(self):
        variable_parents = {}

        for match in self.potential_expr.scanString(self.net_str):
            vars_in_potential = match[0]
            variable_parents[vars_in_potential[0]] = vars_in_potential[1:]
        return variable_parents

    def parse_cpt_values(self) -> dict[str, np.ndarray]:
        variable_cpds: dict[str, np.ndarray] = {}
        variables = list(self.parents.keys())  # same var order as cpt list
        cpt_list = self.cpd_expr.scanString(self.net_str)

        for index, match in enumerate(cpt_list):
            var_name = variables[index]
            cpt_flat = np.array(match[0], dtype=np.float32)
            if CHECK_ZERO:
                assert np.all(cpt_flat > 0.0)
            family = self.parents[var_name] + [var_name]
            cpt_shape = [self.nary[v] for v in family]
            num_states = np.multiply.reduce(cpt_shape)
            assert cpt_flat.shape == (num_states, )
            original_cpt = cpt_flat.reshape(tuple(cpt_shape))
            variable_cpds[var_name] = original_cpt
        return variable_cpds


def read_hugin_network(network_name: str) -> BayesNet:
    network_file_path = config.NETWORK_BASE_DIR / network_name
    assert network_file_path.suffix == '.net'
    with open(network_file_path, 'r') as f:
        network_content = f.read()
    json_output_file = network_file_path.with_suffix('.net.json')
    if json_output_file.exists():
        with open(json_output_file, 'r') as f:
            network_json = json.load(f)
        assert str(network_file_path) == network_json["network_path"]
        assert network_content == network_json["network_content"]
        np_cpts = {
            x: np.array(arr, dtype=np.float32)
            for x, arr in network_json["cpts"].items()
        }
        bn = BayesNet(network_name=network_name,
                      states=network_json["states"],
                      parents=network_json["parents"],
                      cpts=np_cpts)
        return bn

    net = NETReader(network_content)
    bn = BayesNet(network_name=network_name,
                  states=net.states,
                  parents=net.parents,
                  cpts=net.cpts)
    py_cpts = {x: arr.tolist() for x, arr in bn.cpts.items()}
    data = {
        "states": bn.states,
        "parents": bn.parents,
        "cpts": py_cpts,
        "network_path": str(network_file_path),
        "network_content": network_content
    }
    _logger.warning(f"Writing {json_output_file}")
    with open(json_output_file, 'w') as f:
        json.dump(data, f)
    return bn


def run_shell_command(command):
    try:
        result = subprocess.run(command,
                                shell=True,
                                capture_output=True,
                                text=True,
                                check=True)
        return True, result.stdout
    except subprocess.CalledProcessError as e:
        return False, f"Command failed with exit code {e.returncode}: {e.stderr}"
    except Exception as e:
        return False, f"Error running command: {str(e)}"


def evidence_dict_to_xml_str(e) -> str:
    root = ET.Element('instantiation')

    for key, value in e.items():
        inst = ET.SubElement(root, 'inst')
        inst.set('id', str(key))
        inst.set('value', str(value))

    ET.indent(root, space='  ')
    return ET.tostring(root, encoding='unicode')


def parse_ace_eval_output(s: str) -> list[float]:
    probs = []
    for line in s.splitlines():
        if line.startswith("Pr(e) = "):
            s = line.split('=')[1]
            probs.append(float(s.strip()))
        else:
            break
    return probs


def ace_eval(network_file: str, evidences: list[dict]) -> list[float]:
    BATCH_MAX_SIZE = 1000

    def process_batch(ii) -> list[float]:
        inst_list = []
        try:
            for e in evidences[ii:ii + BATCH_MAX_SIZE]:
                xml_str = evidence_dict_to_xml_str(e)
                tmp_file = tempfile.NamedTemporaryFile(mode='w',
                                                       suffix='.inst',
                                                       delete=False)
                with tmp_file.file as f:
                    f.write(xml_str)
                inst_list.append(tmp_file.name)
            cmd_list: list[str] = [ACE_EVALUATE_BINARY, network_file
                                   ] + inst_list
            success, output = run_shell_command(' '.join(cmd_list))
        finally:
            for tmp in inst_list:
                pathlib.Path(tmp).unlink()

        assert success, (f"ACE eval failed {network_file}, \n output {output}")
        batch_probs = parse_ace_eval_output(output)
        return batch_probs

    num_evidences = len(evidences)
    all_probs: list[float] = []
    while len(all_probs) < num_evidences:
        all_probs.extend(process_batch(len(all_probs)))
    assert len(all_probs) == num_evidences
    return all_probs


def convert_to_net(network_path: pathlib.Path) -> None:
    from pgmpy.readwrite import UAIReader, NETWriter, NETReader, BIFReader
    if network_path.suffix == '.uai':
        reader = UAIReader(network_path)
    elif network_path.suffix == '.bif':
        reader = BIFReader(network_path)
    else:
        assert network_path.suffix == ".net"
        reader = NETReader(network_path)
    _logger.info(f"Read from {network_path}")
    writer = NETWriter(reader.get_model())
    output_network_path = network_path.with_suffix(".net")
    writer.write_net(output_network_path)
    _logger.info(f"Wrote to {output_network_path}")


def binary_cnf_to_eclause_DIMACS(cnf: CNF) -> str:
    assert cnf.n_eclaues == 0
    lines = [f"p cnf {cnf.n_vars} {cnf.n_clauses}", "eclauses 0"]
    for c in cnf.clauses:
        for lit in c:
            var_idx = abs(lit)
            assert var_idx <= cnf.n_vars
        c_output = c + (0, )
        lines.append(" ".join(map(str, c_output)))
    return "\n".join(lines)

def cleanup_gv_files(out_dir: pathlib.Path) -> None:
    for gv_file in out_dir.rglob("*.gv"):
        gv_file.unlink()
    _logger.info("Clean up .gv files")


def cleanup_prev_run_pdf(network_name: str) -> None:
    for f in config.OUT_DIR.glob(f"{network_name}*.pdf"):
        f.unlink()
    _logger.info(f"Clean up previous pdf for {network_name}")


def get_ground_truth_pkl_filepath(bn: BayesNet, y: str,
                                  n_features: int) -> pathlib.Path:
    s = ""
    if n_features < len(bn.leaf_var_strs):
        s = f"-{n_features}"
    return config.CACHE_DIR / f"{bn.network_name}_{y}{s}.pkl"


@functools.cache
def get_problem_var_name_order(
        all_vars: frozenset[str], features: frozenset[str],
        y: str) -> tuple[tuple[str, ...], dict[str, int]]:
    other_vars = all_vars - features - {y}
    var_names = tuple(sorted(features)) + tuple(sorted(other_vars)) + (y, )
    name_to_idx = {x: i for i, x in enumerate(var_names)}
    return var_names, name_to_idx

def try_load_ground_truth_joint_prob(
        bn: BayesNet, y: str, features: frozenset[str]) -> np.ndarray | None:
    with open(config.NETWORK_BASE_DIR / bn.network_name, 'r') as f:
        current_network_text = f.read()
    xy_names = features | {y}
    arr = None
    pkl_filename = get_ground_truth_pkl_filepath(bn, y, len(features))
    if pkl_filename.exists():
        with open(pkl_filename, 'rb') as f:
            ret = pickle.load(f)
        _logger.warning(
            f"Read existing ground truth: {bn.network_name}, Y={y}")
        network_text, maybe_arr = ret[:2]
        assert network_text == current_network_text
        if len(ret) == 3:
            _var_names = ret[2]
            if frozenset(_var_names) == xy_names:
                arr = maybe_arr
            else:
                pkl_filename.unlink()
        else:
            assert len(ret) == 2
            assert len(features) == len(bn.leaf_var_strs)
            network_text, maybe_arr = ret
            assert len(maybe_arr.shape) == len(xy_names)
            _logger.warning(f"Converting legacy joint prob pkl {pkl_filename}")
            arr = maybe_arr
            with open(pkl_filename, 'wb') as f:
                pickle.dump((current_network_text, maybe_arr, list(xy_names)),
                            f)
    return arr
