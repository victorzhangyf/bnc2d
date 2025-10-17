# %%
import logging

_logger = logging.getLogger("compile_bnc")
import sys, time, pickle
import enum, itertools, math, random

import numpy as np
import graphviz
import torch
import tqdm

import klay

import global_config

config = global_config.Config()

import utils
from factor_mat import FactorMat, FactorMatTrue


class NodeType(enum.Flag):
    LITERAL = enum.auto()
    CONST = enum.auto()
    MULT = enum.auto()
    ADD = enum.auto()
    LEAF = LITERAL | CONST
    DECOMPOSABLE = LITERAL | MULT


class ArithmeticCircuit:
    _EXPECTED_COMPILE_KIND = "ALWAYS_SUM"

    def __init__(self, bn: utils.BayesNet) -> None:
        _logger.info("Constructing AC")

        self.device = config.AC_DEVICE
        self.batch_size = config.AC_BATCH_SIZE
        self.bn = bn
        self.n_bn_vars = len(self.bn.states)
        self.network_name = bn.network_name
        self.network_base_path = config.NETWORK_BASE_DIR / self.network_name

        self.var_name_to_nary = {
            x: len(states)
            for x, states in self.bn.states.items()
        }

        # Mathematical space (NORMAL or LOG_E)
        self.space = "NORMAL"

        self.var_name_to_lits: dict[str, list[int]] = {}
        self.lit_is_const: np.ndarray

        self.lit_default_weight: np.ndarray

        self.node_to_type: list[NodeType] = []
        self.node_to_lit: dict[int, int] = {}

        self.node_operands: list[list[int]] = []
        self.node_to_parents: list[list[int]] = []

        self.lit_to_node_idx: np.ndarray

        # only for non-const
        self.lit_to_var_pos: dict[int, tuple[str, int]] = {}

        self.node_to_var_set: list[frozenset[str]] = []

        lmap_file = self.network_base_path.with_suffix('.net.lmap')
        self._neg_lit_nodes: set[int] = set()
        self.read_lit_pot_map(lmap_file)
        ac_file = self.network_base_path.with_suffix('.net.ac')
        self.read_circuit(ac_file)
        self.n_pos_lits = int((self.lit_default_weight.size - 1) / 2)

        _logger.info("AC init done")
        self.torch_ac = None

        self.inference_result_cache: dict[frozenset[str], FactorMat] = {}

    def ace_infer_cond_prob_cached(self,
                                   left_vars: frozenset[int],
                                   cond_vars: frozenset[int],
                                   var_idx_to_name: tuple[str, ...],
                                   cache: bool,
                                   use_shell: bool = True) -> FactorMat:
        assert config.DEBUG_LVL > 1
        assert not (left_vars & cond_vars)
        if not left_vars:
            return FactorMatTrue
        var_set = left_vars | cond_vars
        var_name_set = frozenset(var_idx_to_name[i] for i in var_set)
        if var_set not in self.inference_result_cache:
            var_list = list(sorted(var_set))
            joint_prob_shape = tuple(
                len(self.bn.states[var_idx_to_name[x]]) for x in var_list)

            if use_shell:
                prob_size = np.prod(joint_prob_shape)
                eval_var_names = tuple(var_idx_to_name[i]
                                       for i in sorted(var_list))
                _logger.info(
                    f"call shell ace: P({eval_var_names}), size {prob_size}")
                axis_indices = [np.arange(l) for l in joint_prob_shape]
                coords = itertools.product(*axis_indices)
                evidences = []
                for coord in coords:

                    evidences.append({
                        i: self.bn.states[i][j]
                        for i, j in zip(eval_var_names, coord)
                    })
                ace_eval_ret = utils.ace_eval(
                    str(config.NETWORK_BASE_DIR / self.network_name),
                    evidences)
                joint_prob = FactorMat(
                    mat=np.array(ace_eval_ret).reshape(joint_prob_shape),
                    vs=tuple(var_list),
                    nary=joint_prob_shape,
                    var_names=eval_var_names)
            else:
                joint_prob = self.klay_eval_joint(var_set, var_idx_to_name)

            if cache:
                self.inference_result_cache[var_name_set] = joint_prob
            else:
                _logger.info(f"Skip caching {joint_prob._print_description()}")
        else:
            joint_prob = self.inference_result_cache[var_name_set]
        if not cond_vars:
            cond_prob = joint_prob
        else:
            cond_prob = joint_prob.condition_on(cond_vars)
        return cond_prob

    def sample_eval_joint(
            self, y: str, features: frozenset[str],
            eval_size: int) -> tuple[FactorMat, tuple[int | slice, ...]]:
        assert eval_size <= len(features)
        assert eval_size <= config.DEBUG_CHECK_FEATURES_LIMIT
        var_names, _ = utils.get_problem_var_name_order(
            self.bn.all_var_strs, features, y)
        e = None
        f_list = list(range(len(features)))
        random.shuffle(f_list)
        eval_var_set = frozenset(f_list[:eval_size]) | {self.n_bn_vars - 1}
        if eval_size == len(features):
            idx = tuple(slice(None) for _ in range(len(features)))
        else:
            fixed_features = f_list[eval_size:]
            nary = [len(self.bn.states[var_names[v]]) for v in fixed_features]
            e = {
                v: random.randint(0, n - 1)
                for v, n in zip(fixed_features, nary)
            }
            idx = tuple(e[i] if i in e else slice(None)
                        for i in range(len(features)))
        fac = self.klay_eval_joint(eval_var_set, var_names, fixed_vals=e)
        return fac, idx

    def node_to_str(self, node_idx: int) -> str:
        node_type = self.node_to_type[node_idx]
        if node_type in NodeType.LEAF:
            lit = self.node_to_lit[node_idx]
            v = self.lit_default_weight[lit]
            node_txt = f"Lit {lit} ({v:.2f})"
        else:
            children_indices = self.node_operands[node_idx]
            children_strs = ",  ".join(
                [self.node_to_str(i) for i in children_indices])
            if node_type == NodeType.MULT:
                node_txt = f"* [{children_strs}]"
            else:
                node_txt = f"+ [{children_strs}]"
        return f"Node {node_idx}: {node_txt}"

    def _to_torch(self) -> None:
        _start_time = time.perf_counter()

        klay_nnf, _ = self.to_klay()

        semiring = 'real' if self.space == "NORMAL" else 'log'
        assert semiring == 'real'  # TODO: for now
        self.torch_ac = klay_nnf.to_torch_module(semiring=semiring).to(
            self.device)
        assert self.torch_ac is not None
        if self.batch_size > 0:
            self.torch_ac = torch.vmap(self.torch_ac)
        self.torch_ac = torch.compile(self.torch_ac, mode="reduce-overhead")
        _dur = time.perf_counter() - _start_time
        _logger.info(f"Klay AC constructed: {_dur:.2f} sec")

    def klay_eval_joint(self,
                        var_set: frozenset[int],
                        var_idx_to_name: tuple[str, ...],
                        fixed_vals: dict[int, int] | None = None) -> FactorMat:
        # _logger.info(f"AC eval with klay pytorch")
        if self.torch_ac is None:
            self._to_torch()
        assert self.torch_ac is not None
        _start_time = time.perf_counter()

        sorted_var_indices = list(sorted(var_set))
        var_names = tuple(var_idx_to_name[i] for i in sorted_var_indices)

        eval_var_nary = tuple(self.var_name_to_nary[x] for x in var_names)
        n_total_combinations = math.prod(eval_var_nary)

        tqdm_disable = n_total_combinations < 2**19

        var_to_changable: list[list[int]] = [
            self.var_name_to_lits[x] for x in var_names
        ]
        grids = torch.meshgrid(*[torch.tensor(x) for x in var_to_changable],
                               indexing='ij')
        flat_combinations = torch.stack(grids, dim=-1).reshape(
            n_total_combinations, len(var_names)).to(self.device)
        all_changable = list(itertools.chain.from_iterable(var_to_changable))
        default_weights = torch.tensor(
            self.lit_default_weight[:self.n_pos_lits + 1],
            dtype=torch.float32,
            device=self.device)
        neg_weights = torch.ones(self.n_pos_lits,
                                 dtype=torch.float32,
                                 device=self.device)

        default_weights[all_changable] = 0.0

        if fixed_vals is not None:
            assert not (var_set & frozenset(fixed_vals.keys()))
            for v, pos in fixed_vals.items():
                lits = self.var_name_to_lits[var_idx_to_name[v]]
                default_weights[lits] = 0.0
                default_weights[lits[pos]] = 1.0
        ret_forward = []
        with torch.no_grad():
            if self.batch_size > 0:
                for start_idx in tqdm.tqdm(range(0, n_total_combinations,
                                                 self.batch_size),
                                           disable=tqdm_disable):
                    end_idx = min(start_idx + self.batch_size,
                                  n_total_combinations)
                    l = end_idx - start_idx
                    rows_idx = torch.arange(l).reshape(l, 1).to(self.device)
                    batch_mask = flat_combinations[start_idx:end_idx].to(
                        self.device)
                    batch_weights = default_weights.repeat(l, 1)
                    batch_weights[rows_idx, batch_mask] = 1.0
                    batch_neg_weights = neg_weights.repeat(l, 1)
                    batch_result = self.torch_ac(batch_weights[:, 1:],
                                                 batch_neg_weights)
                    ret_forward.append(batch_result.detach().cpu().numpy())
            else:
                for start_idx in tqdm.tqdm(range(n_total_combinations),
                                           disable=tqdm_disable):
                    mask = flat_combinations[start_idx]
                    t_weights = default_weights.clone()
                    t_weights[mask] = 1.0
                    result = self.torch_ac(t_weights[1:], neg_weights)
                    ret_forward.append(result.detach().cpu().numpy())
        mat = np.concatenate(ret_forward).reshape(eval_var_nary)
        fac = FactorMat(mat=mat,
                        vs=tuple(sorted_var_indices),
                        nary=eval_var_nary,
                        var_names=var_names)
        return fac

    def to_klay(self) -> tuple[klay.Circuit, list]:
        ac = klay.Circuit()
        nodes = []
        pos_lits = set()
        for node_idx, node_type in enumerate(self.node_to_type):
            if node_type in NodeType.LEAF:
                lit = self.node_to_lit[node_idx]
                nodes.append(ac.literal_node(lit))
                pos_lits.add(abs(lit))
                continue
            children_indices = self.node_operands[node_idx]
            children_nodes = [nodes[i] for i in children_indices]
            if node_type == NodeType.MULT:
                nodes.append(ac.and_node(children_nodes))
            else:
                assert node_type == NodeType.ADD
                nodes.append(ac.or_node(children_nodes))
        assert len(pos_lits) == self.n_pos_lits
        ac.set_root(nodes[-1])
        return ac, nodes

    def eval_naive(
        self,
        evidence: np.ndarray,
        start_from: tuple[np.ndarray, int] | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        if start_from is None:
            node_values = np.zeros(len(self.node_to_type), dtype=np.float32)
        else:
            node_values, start_idx = start_from
        is_node_zero = np.zeros(len(self.node_to_type), dtype=np.bool)
        for node_idx, node_type in enumerate(self.node_to_type):
            if node_type in NodeType.LEAF:
                lit = self.node_to_lit[node_idx]
                node_values[node_idx] = evidence[lit]
                continue
            children_indices = self.node_operands[node_idx]
            if node_type == NodeType.MULT:
                new_val = np.prod(node_values[children_indices])
            else:
                new_val = np.sum(node_values[children_indices])
            node_values[node_idx] = new_val
            is_node_zero[node_idx] = new_val == 0
        return node_values, is_node_zero

    def to_graphviz(self):
        g = graphviz.Digraph(f'{self.network_name}-AC')
        for node_idx, node_type in enumerate(self.node_to_type):
            node_attributes = {}
            if node_type in NodeType.LEAF:
                lit = self.node_to_lit[node_idx]
                if self.lit_is_const[lit]:
                    node_label = self.lit_default_weight[lit]
                else:
                    node_label = self.lit_to_var_pos[lit]
                node_attributes['shape'] = 'box'
            elif node_type == NodeType.MULT:
                node_label = "*"
            else:
                assert node_type == NodeType.ADD
                node_label = "+"
            if node_type not in NodeType.LEAF:
                for child_idx in self.node_operands[node_idx]:
                    g.edge(str(child_idx), str(node_idx))
            node_label = f"{node_idx}. {node_label}"
            if node_type == NodeType.MULT:
                node_label += f" {list(sorted(self.node_to_var_set[node_idx]))}"
            g.node(name=str(node_idx), label=node_label, **node_attributes)
        return g

    def read_lit_pot_map(self, lmap_file) -> None:
        delimiter = "$"
        num_lits = sys.maxsize
        lits_finished = 0
        n_const_lits = 0
        n_I_lits = 0
        v_count = -1
        t_count = -1

        with open(lmap_file, 'r') as f:
            reader = f.readlines()

        for line in reader:
            assert lits_finished <= num_lits
            line = line.strip()
            if not line.startswith("cc" + delimiter):
                continue

            tokens = line.split(delimiter)
            type_token = tokens[1]

            # Compile kind
            # cc K compileKind
            if type_token == "K":
                assert self._EXPECTED_COMPILE_KIND == tokens[2]
                continue

            # Mathematical space
            # cc S space
            if type_token == "S":
                self.space = tokens[2]
                continue

            # Literal count
            # cc N numLogicVariables
            if type_token == "N":
                num_logic_vars = int(tokens[2])
                num_lits = num_logic_vars * 2
                arr_len = num_lits + 1  # index 0 is just padding
                # for lit < 0, store weight to the end of the array (for neg indexing)
                self.lit_default_weight = np.zeros(arr_len, dtype=np.float32)
                self.lit_is_const = np.ones(arr_len, dtype=np.bool)
                continue

            # Variable count
            # cc v varCount
            if type_token == "v":
                assert v_count < 0
                v_count = int(tokens[2])
                continue

            # Potential count
            # cc t pot_count
            if type_token == "t":
                assert t_count < 0
                t_count = int(tokens[2])
                continue

            # Indicator and parameter
            # cc V name count
            # cc T name count
            if type_token == "V" or type_token == "T":
                name = tokens[2]
                val_count = int(tokens[3])
                if type_token == "V":
                    self.var_name_to_lits[name] = [0] * val_count
                continue

            # Literal descriptions (I, P, C)
            # cc I lit weight elimOp varName varValue (indicator literal)
            # cc P lit weight elimOp potName potPos (parameter literal) # never appears
            # cc C lit weight elimOp (constant literal)
            assert type_token in ["I", "C"]
            lit = int(tokens[2])
            assert lit != 0
            weight = float(tokens[3])

            assert len(self.lit_default_weight) > abs(lit * 2)
            self.lit_default_weight[lit] = weight

            if type_token == "C":
                self.lit_is_const[lit] = True
                n_const_lits += 1
            else:
                # indicator
                assert lit > 0
                self.lit_is_const[lit] = False
                assert type_token == "I"
                n_I_lits += 1
                var_name = tokens[5]
                pos = int(tokens[6])
                self.var_name_to_lits[var_name][pos] = lit
                self.lit_to_var_pos[lit] = (var_name, pos)

            lits_finished += 1

        assert lits_finished == num_lits
        assert v_count == t_count
        assert len(self.var_name_to_lits) == v_count
        assert len(self.lit_to_var_pos) == n_I_lits
        _logger.info(f"Read lmap for {self.network_name} with {v_count} vars")

        _logger.info(
            f"Total {num_lits} lits: {n_const_lits} const, {n_I_lits} indicator, {num_lits - n_const_lits - n_I_lits} params"
        )

    def read_circuit(self, ac_file) -> None:
        num_nodes = sys.maxsize
        node_idx = 0
        with open(ac_file, 'r') as f:
            reader = f.readlines()

        for line in reader:
            assert node_idx == len(self.node_to_type)
            if node_idx == num_nodes:
                break
            line = line.strip()
            if not line or line.startswith("c"):
                continue

            tokens = line.split()
            line_type = tokens[0]

            # Header line: "nnf" numNodes numEdges numVars
            if line_type == "nnf":
                num_nodes = int(tokens[1])
                num_edges = int(tokens[2])
                num_binary_vars = int(tokens[3])
                self.lit_to_node_idx = np.full(num_binary_vars * 2 + 1,
                                               -1,
                                               dtype=np.int32)
                continue
            var_list: list[str] = []
            self.node_to_parents.append([])
            node_operand_indices = []
            # Below are Node lines
            # Literal nodes
            # L literal
            if line_type.lower() == 'l':
                assert len(tokens) == 2
                lit = int(tokens[1])
                self.node_to_lit[node_idx] = lit
                self.lit_to_node_idx[lit] = node_idx
                if lit < 0:
                    assert self.lit_is_const
                    self._neg_lit_nodes.add(node_idx)
                    assert self.lit_default_weight[lit] == 1
                if self.lit_is_const[lit]:
                    self.node_to_type.append(NodeType.CONST)

                else:
                    self.node_to_type.append(NodeType.LITERAL)
                    var_list.append(self.lit_to_var_pos[lit][0])
            else:
                assert len(tokens) >= 4
                child_start_idx = 2
                num_children = int(tokens[1])
                # Multiplication nodes (AND)
                # A/* numChildren child+
                if line_type == 'A' or line_type == '*':
                    self.node_to_type.append(NodeType.MULT)

                # Explicit ADD nodes
                # + numChildren child+
                elif line_type == '+':
                    self.node_to_type.append(NodeType.ADD)

                # Explicit MAX nodes
                # X numChildren child+
                elif line_type.lower() == 'x':
                    assert False
                    # self.node_to_type.append(NodeType.MAX)

                else:
                    # OR nodes that become ADD or MAX based on elimination operation
                    # currently always ADD
                    assert line_type == 'O'
                    # O logic_var numChildren child+
                    child_start_idx = 3
                    num_children = int(tokens[2])
                    # logic_var = int(tokens[1])
                    self.node_to_type.append(NodeType.ADD)
                assert len(tokens) == child_start_idx + num_children
                for child_idx in map(int, tokens[child_start_idx:]):
                    if child_idx in self._neg_lit_nodes:
                        continue
                    node_operand_indices.append(child_idx)
                    self.node_to_parents[child_idx].append(node_idx)
                    var_list.extend(self.node_to_var_set[child_idx])
            var_set = set(var_list)
            if self.node_to_type[-1] in NodeType.DECOMPOSABLE:
                assert len(var_set) == len(var_list)

            self.node_operands.append(node_operand_indices)
            self.node_to_var_set.append(frozenset(var_set))
            node_idx += 1

        assert len(self.node_to_type) == num_nodes
        assert len(self.node_operands) == num_nodes
        assert self.lit_to_node_idx[0] == -1
        _logger.info(f"Loaded {ac_file}: {num_nodes} nodes, {num_edges} edges")
        if config.DEBUG_LVL > 1:
            for pars in self.node_to_parents:
                assert len(pars) == len(set(pars))

    def get_ground_truth_joint_prob(self, y: str,
                                    features: frozenset[str]) -> np.ndarray:
        assert features
        assert features <= self.bn.leaf_var_strs
        arr = utils.try_load_ground_truth_joint_prob(self.bn, y, features)
        if arr is None:
            arr = self._infer_and_save_ground_truth_joint_prob(y, features)
        return arr

    def _infer_and_save_ground_truth_joint_prob(
            self, y: str, features: frozenset[str]) -> np.ndarray:
        _logger.warning(
            f"Computing ground truth: {self.bn.network_name}, Y={y}")
        assert y in self.bn.root_var_strs
        ordered_var_names, name_to_idx = utils.get_problem_var_name_order(
            self.bn.all_var_strs, features, y)
        xy_names = features | {y}
        xy_indices_set = frozenset(name_to_idx[i] for i in xy_names)
        joint_prob = self.klay_eval_joint(xy_indices_set, ordered_var_names)
        arr = joint_prob.mat
        with open(config.NETWORK_BASE_DIR / self.bn.network_name, 'r') as f:
            current_network_text = f.read()
        pkl_filename = utils.get_ground_truth_pkl_filepath(
            self.bn, y, len(features))
        with open(pkl_filename, 'wb') as f:
            pickle.dump((current_network_text, arr, list(xy_names)), f)
        _logger.warning(f"Saved ground truth to {pkl_filename}")
        return arr


