import logging

_logger = logging.getLogger("compile_bnc")
import dataclasses, time, itertools, pickle, math, random

import graphviz
import numpy as np
import global_config
import random

config = global_config.Config()
from java_inflib import JavaInflibWrapper

from factor_mat import FactorMat, FactorMatTrue

from utils import C2DDTreeNode, BayesNet


@dataclasses.dataclass
class JointTreeNode:
    idx: int
    separators: dict[int, frozenset[int]]  # neighbor idx, sep vars
    feas: dict[int, frozenset[int]]
    cluster: frozenset[int]
    fams: frozenset[int]  # assign cpt factors

    def __post_init__(self) -> None:
        for i, v in self.feas.items():
            assert i in self.separators
            assert v


@dataclasses.dataclass
class DirectedFeatureMap:
    idx: int
    p_idx: int
    d: dict[int, frozenset[int]]

    def __post_init__(self) -> None:
        assert self.d
        for i, s in self.d.items():
            assert i >= 0
            assert len(s) > 0

    @property
    def downstream_neighbor_features(self) -> frozenset[int]:
        combined: set[int] = set()
        for s in self.d.values():
            combined.update(s)
        return frozenset(combined)


class JointTree:

    def __init__(self, family_var_indices: tuple[tuple[int, ...], ...],
                 n_features: int, ordered_var_names: tuple[str, ...],
                 bn: BayesNet, var_n_ary: np.ndarray[tuple[int],
                                                     np.dtype[np.int32]],
                 method: str, gen_tree_args: dict[str, str]) -> None:
        self.n_features = n_features
        self.n_vars = len(family_var_indices)
        self.y_idx = self.n_vars - 1
        self.family_var_indices = family_var_indices
        self.bn = bn
        self.network_name = self.bn.network_name
        self.ordered_var_names = ordered_var_names
        self.var_name_to_idx = {s: i for i, s in enumerate(ordered_var_names)}
        self.var_n_ary = var_n_ary
        self.cpt_factors = self.rearrange_cpt()

        self._nodes: dict[int, JointTreeNode] = {}
        assert method in ["c2d", "samiam"]
        self._method = method

        self.ordered_downstream_neighbors: dict[int, tuple[JointTreeNode,
                                                           ...]] = {}

        self.decide_nodes: set[int] = set()
        self.downward_msg_nodes: set[int] = set()
        self._directed_feature_maps: dict[int, DirectedFeatureMap] = {}
        self.root_idx = -1
        self._gen_tree_args = gen_tree_args
        if self._method == "c2d":  # init jointree from dtree
            self._init_jointree_from_c2d_dtree(gen_tree_args)
        else:
            self.init_jointree_from_samiam()

        self._gen_tree_args["method"] = method

        assert self.root_idx >= 0
        self.root = self._nodes[self.root_idx]
        assert self.y_idx in self.root.cluster

        if config.DEBUG_LVL > 1:
            fams = self._check_joint_tree(self.root, None, frozenset(),
                                          frozenset())
            assert len(fams) == self.n_vars

        self._node_may_absorb_features: set[int] = set()

        self._create_directed_feature_map(self.root, -1)

        if config.DEBUG_LVL > 1:
            fams = self._check_joint_tree(self.root, None, frozenset(),
                                          frozenset())
            assert len(fams) == self.n_vars

        self.ordered_expected_features: dict[int, tuple[tuple[int, ...],
                                                        ...]] = {}
        l = self._compute_ordered_expected_features(self.root_idx, -1)
        assert len(tuple(itertools.chain(*l))) == self.n_features
        _seen_features: set[int] = set()
        self._compute_decide_nodes(self.root, frozenset(), _seen_features)
        assert len(_seen_features) == self.n_features

    @classmethod
    def try_jointrees(
            cls, family_var_indices: tuple[tuple[int, ...], ...],
            n_features: int, ordered_var_names: tuple[str, ...], bn: BayesNet,
            var_n_ary: np.ndarray[tuple[int],
                                  np.dtype[np.int32]]) -> "JointTree":

        JOINTREE_PKL = config.CACHE_DIR / f"{bn.network_name}-{n_features}-jt.pkl"
        if config.DEBUG_READ_EXISTING_JTREE and JOINTREE_PKL.exists():
            _logger.warning("Read jt pkl")
            with open(JOINTREE_PKL, 'rb') as f:
                jt: JointTree = pickle.load(f)
            assert jt.family_var_indices == family_var_indices
            assert jt.n_features == n_features
            assert jt.ordered_var_names == ordered_var_names
            assert jt.bn.cpts.keys() == bn.cpts.keys()
            assert all(
                np.array_equal(jt.bn.cpts[k], bn.cpts[k])
                for k in bn.cpts.keys())
            return jt
        method = config.JOINT_TREE_METHOD

        start_time = time.perf_counter()

        jt_list: list[JointTree] = []
        _MULTIPLIER = math.floor(math.log2(max(2, n_features - 8)))
        if "c2d" in method:
            _logger.info(
                f"Trying c2d dt: bal range {config.C2D_DT_BAL_RANGE}, multiplier: {_MULTIPLIER}"
            )
            c2d_args_list: list[dict] = []
            for _ in range(_MULTIPLIER):
                for c2d_dt_bal in range(*config.C2D_DT_BAL_RANGE):
                    c2d_args_list.append({
                        "dt_method": str(1),
                        "dt_count": str(1),
                        "ubfs": f"{c2d_dt_bal} {c2d_dt_bal}"
                    })
            _logger.info(f"Trying c2d joint tree method 2,3,4")
            for _ in range(_MULTIPLIER):
                for dt_method in [2, 3, 4]:
                    c2d_args_list.append({"dt_method": str(dt_method)})

            for d in c2d_args_list:
                jt = JointTree(family_var_indices, n_features,
                               ordered_var_names, bn, var_n_ary, "c2d", d)
                jt_list.append(jt)
        if "samiam" in method:
            _logger.info(f"Trying samiam joint tree")
            for _ in range(5 * _MULTIPLIER):
                jt_list.append(
                    JointTree(family_var_indices, n_features,
                              ordered_var_names, bn, var_n_ary, "samiam",
                              {"seed": random.randint(0, 1000000)}))

        jt_list.sort(key=lambda x: x.get_quality())
        time_dur = time.perf_counter() - start_time
        jt = jt_list[0]
        _logger.info(
            f"JT quality {jt.quality}, original width {jt._original_width}, l width {jt.max_l_width}, r width {jt.max_r_width}"
        )
        if config.DEBUG_LVL > 0:
            _logger.info(
                f"Got {len(jt_list)} jt: {time_dur:.2f} sec, best {jt.quality}, worst {jt_list[-1].quality}"
            )
            if config.DEBUG_LVL > 1:
                _logger.info(f"Chose with param {jt._gen_tree_args}")
                with open(JOINTREE_PKL, 'wb') as f:
                    pickle.dump(jt, f)
                _logger.warning("SAVED jt pkl")
        return jt

    def prepare_for_bnc(self, odds_thresh: float = -1) -> None:
        if self.var_n_ary[-1] == 2:
            assert odds_thresh > 0
            neg_class = 1 - config.TARGET_CLS
            y_fac = self.cpt_factors[-1]
            y_prior = y_fac.mat.copy()
            assert len(y_prior.shape) == 1
            y_prior[neg_class] *= odds_thresh
            y_prior = (y_prior / np.sum(y_prior)).astype(np.float32)
            new_y_fac = FactorMat(y_prior, y_fac.vs, y_fac.nary,
                                  y_fac.var_names, y_fac.cond_vars)
            self.cpt_factors[-1] = new_y_fac
        else:
            assert odds_thresh < 0

        self.src_dest_messages, self.original_cluster_joints = self._message_passing(
            self.root)

        self._global_last_nodes: set[int] = set()

        self._tag_global_last_nodes()
        self.downward_msg_div: dict[int, dict[int, FactorMat]] = {}
        self._process_downward_msg_nodes(self.root, frozenset())

        self.original_pvz: dict[int, FactorMat] = {}
        self._initialize_pvz(self.root, -1)

        if config.DEBUG_LVL > 1:
            self.to_graphviz().render(directory=config.OUT_DIR)
            _logger.info("Saved joint tree graph")

    def get_fea_to_global_order(self) -> tuple[int, ...]:
        stack = [self.root]
        l: list[int] = []
        while stack:
            n = stack.pop()
            visit_order = self.ordered_downstream_neighbors[n.idx]
            if not visit_order:
                feas = self.ordered_expected_features[n.idx]
                assert len(feas) == 1
                assert feas[0]
                l.extend(sorted(feas[0]))
            else:
                stack.extend(reversed(visit_order))
        assert len(l) == self.n_features
        assert len(l) == len(frozenset(l))
        ret = [0 for _ in range(self.n_features)]
        for i, j in enumerate(l):
            ret[j] = i
        return tuple(ret)

    def _initialize_pvz(self, n: JointTreeNode, p_idx: int) -> None:
        fac = self.original_cluster_joints[n.idx]
        visit_order = self.ordered_downstream_neighbors[n.idx]
        if p_idx >= 0:
            p_factor = self.original_cluster_joints[p_idx]
            p_sep = n.separators[p_idx]
            _t = frozenset(fac.vs) & frozenset(p_factor.vs)
            assert p_sep == _t
            fac = fac.factor_mult(p_factor.project_to(p_sep), divide=True)
        assert n.idx not in self.original_pvz
        self.original_pvz[n.idx] = fac
        for neighbor in visit_order:
            self._initialize_pvz(neighbor, n.idx)

    def _tag_global_last_nodes(self) -> None:
        n = self.root
        while True:
            assert n.idx not in self._global_last_nodes
            self._global_last_nodes.add(n.idx)
            visit_order = self.ordered_downstream_neighbors[n.idx]
            if not visit_order:
                break
            n = visit_order[-1]

    def _process_downward_msg_nodes(self, n: JointTreeNode,
                                    remained_sep: frozenset[int]) -> bool:
        assert n.idx not in self.downward_msg_nodes
        visit_order = self.ordered_downstream_neighbors[n.idx]
        is_global_last = n.idx in self._global_last_nodes
        if is_global_last:
            assert not remained_sep
            self.downward_msg_nodes.add(n.idx)
        else:
            if visit_order:
                n_feas = len(
                    list(
                        itertools.chain.from_iterable(
                            self.ordered_expected_features[n.idx])))
            else:
                n_feas = len([i for i in n.fams if i < self.n_features])
            if n_feas <= len(remained_sep):
                return False
        if not visit_order:
            self.downward_msg_nodes.add(n.idx)
            return True
        d = {}
        r = visit_order[-1]
        r_ret = self._process_downward_msg_nodes(r, remained_sep)
        if n.idx in self._global_last_nodes:
            assert r_ret
        if not r_ret:
            return False
        r_cluster_joint = self.original_cluster_joints[r.idx]
        r_sep = n.separators[r.idx]
        d[r.idx] = r_cluster_joint.project_to(r_sep)
        if len(visit_order) == 2:
            l, _ = visit_order
            l_combined_sep = remained_sep | r_sep
            l_ret = self._process_downward_msg_nodes(l, l_combined_sep)
            if l_ret:
                l_cluster_joint = self.original_cluster_joints[l.idx]
                r_sep = n.separators[l.idx]
                d[l.idx] = l_cluster_joint.project_to(r_sep)

        if d:
            self.downward_msg_nodes.add(n.idx)
            self.downward_msg_div[n.idx] = d
            return True
        return False

    def _compute_decide_nodes(self, n: JointTreeNode,
                              remained_sep: frozenset[int],
                              seen_features: set[int]) -> bool:
        # criteria:
        # has visited Y and at least one feature
        # Y is not in the original separator
        is_decide = all([
            len(seen_features) > 0,
            self.y_idx not in n.cluster,
            self.y_idx not in remained_sep,
        ])

        _expected_fea = self.ordered_expected_features[n.idx]
        visit_order = self.ordered_downstream_neighbors[n.idx]
        ordered_sep = tuple(n.separators[neighbor.idx]
                            for neighbor in visit_order)
        if visit_order:
            if len(visit_order) == 2:
                b = self._compute_decide_nodes(visit_order[0],
                                               remained_sep | ordered_sep[1],
                                               seen_features)
                if is_decide:
                    assert b
            b = self._compute_decide_nodes(visit_order[-1], remained_sep,
                                           seen_features)
            if is_decide:
                assert b
        else:
            feature_fams = frozenset(_expected_fea[0])
            assert not feature_fams & seen_features
            seen_features.update(feature_fams)

        assert n.idx not in self.decide_nodes
        if is_decide:
            self.decide_nodes.add(n.idx)
        return is_decide

    def _compute_ordered_downstream_neighbors(
            self, n_idx: int, p_idx: int) -> tuple[JointTreeNode, ...]:
        ret: tuple[JointTreeNode, ...] = ()
        node = self._nodes[n_idx]
        candidates = [i for i in node.feas if i != p_idx]
        for i in node.feas:
            assert node.feas[i]
        if candidates:
            sep_dict = node.separators
            neighbor_ysep_feas = [
                (
                    i,
                    self.y_idx in sep_dict[i],  # y in sep
                    len(node.feas[i]),  # n features
                    len(sep_dict[i]),  # sep size
                ) for i in candidates
            ]
            assert len(neighbor_ysep_feas) > 0

            if len(neighbor_ysep_feas) == 1:
                ret = (self._nodes[neighbor_ysep_feas[0][0]], )
            else:
                neighbor_ysep_feas.sort(key=lambda x: x[2] + x[3])
                neighbor_ysep_feas.sort(key=lambda x: x[1], reverse=True)

                ret = tuple(self._nodes[x[0]] for x in neighbor_ysep_feas)
        return ret

    def _compute_ordered_expected_features(
            self, n_idx: int, p_idx: int) -> tuple[tuple[int, ...], ...]:
        # first list the vars in the cluster itself
        # return a tuple of length: 1 + len(visit_order)
        n = self._nodes[n_idx]
        fam_features = tuple(
            sorted(i for i in n.cluster if i < self.n_features))
        l = [fam_features]
        assert n_idx not in self.ordered_downstream_neighbors
        visit_order = self._compute_ordered_downstream_neighbors(n_idx, p_idx)
        if fam_features:
            assert not visit_order
        self.ordered_downstream_neighbors[n_idx] = visit_order
        for neighbor in visit_order:
            assert neighbor != p_idx
            neigbhor_ret = self._compute_ordered_expected_features(
                neighbor.idx, n_idx)
            l.append(tuple(sorted(
                itertools.chain.from_iterable(neigbhor_ret))))
            flat = list(itertools.chain.from_iterable(l))
            assert len(flat) == len(set(flat))
        ret = tuple(l)
        self.ordered_expected_features[n_idx] = ret
        return ret

    def get_subtree_width(self, n: JointTreeNode, p_idx: int) -> int:
        visit_order = self.ordered_downstream_neighbors[n.idx]
        m = len(n.cluster) - 1
        for c in visit_order:
            m = max(m, self.get_subtree_width(c, n.idx))
        return m

    def get_quality(self) -> tuple[int, ...]:
        l_width_list = []
        r_width_list = []

        stack = [(self.root, True)]
        added = {self.root.idx}
        self.ftree_width = -1
        u_size = 0  # features outside subtree
        while stack:
            n, is_global_last = stack.pop()
            self.ftree_width = max(self.ftree_width, len(n.cluster) - 1)
            r_width_list.append(len(n.cluster) - 1 + u_size)
            visit_order = self.ordered_downstream_neighbors[n.idx]
            feas = self.ordered_expected_features[n.idx]
            if not visit_order:
                continue
            if is_global_last and len(visit_order) == 2:
                l_max_c = self.get_subtree_width(visit_order[0], n.idx)
                l_width_list.append(l_max_c + len(feas[1]))
                u_size += len(feas[1])

            r = visit_order[-1]
            assert r.idx not in added
            added.add(r.idx)
            stack.append((r, True))
           
        self.max_l_width = max(l_width_list)
        self.max_r_width = max(r_width_list)
        width_list = sorted(
            [self._original_width, self.max_l_width, self.max_r_width],
            reverse=True)
        assert max(width_list) == width_list[0]
        self.quality = width_list[0]
        return tuple(width_list)

    def _message_passing(
        self, chosen_root: JointTreeNode
    ) -> tuple[dict[tuple[int, int], FactorMat], dict[int, FactorMat]]:
        start_time = time.perf_counter()
        # init, compute all original factors
        remained_adj: dict[int, set[int]] = {}
        node_factors: dict[int, FactorMat] = {}
        stack = [chosen_root]
        added_indices: set[int] = set({chosen_root.idx})
        while stack:
            n = stack.pop()
            assert n.idx not in node_factors
            assert n.idx not in remained_adj
            fac = FactorMatTrue
            for fam_idx in n.fams:
                fac = fac.factor_mult(self.cpt_factors[fam_idx])
            node_factors[n.idx] = fac
            remained_adj[n.idx] = set(n.separators.keys())
            for adj_idx in n.separators:
                if adj_idx not in added_indices:
                    stack.append(self._nodes[adj_idx])
                    added_indices.add(adj_idx)

        # pull
        ready_nodes = [i for i, ns in remained_adj.items() if len(ns) == 1]
        sent_nodes = set()
        message_seq: list[tuple[int, int, FactorMat]] = []
        src_dest_messages = {}
        while ready_nodes:
            n_idx = ready_nodes.pop()
            if n_idx == chosen_root.idx:
                continue
            assert n_idx not in sent_nodes
            assert len(remained_adj[n_idx]) == 1
            dest = tuple(remained_adj[n_idx])[0]
            del remained_adj[n_idx]
            assert remained_adj[dest]
            assert n_idx in remained_adj[dest]
            n = self._nodes[n_idx]
            sep = n.separators[dest]
            msg = node_factors[n_idx].project_to(sep)
            node_factors[dest] = node_factors[dest].factor_mult(msg)
            assert (n_idx, dest) not in src_dest_messages
            src_dest_messages[(n_idx, dest)] = msg
            message_seq.append((n_idx, dest, msg))
            sent_nodes.add(n_idx)
            remained_adj[dest].remove(n_idx)
            if len(remained_adj[dest]) == 1:
                ready_nodes.append(dest)
        assert len(remained_adj) == 1
        assert chosen_root.idx in remained_adj
        assert len(remained_adj[chosen_root.idx]) == 0
        assert len(message_seq) == len(added_indices) - 1

        # push
        while message_seq:
            new_dest, new_src, old_msg = message_seq.pop()
            sep = self._nodes[new_src].separators[new_dest]
            msg = node_factors[new_src].factor_mult(
                old_msg, divide=True).project_to(sep)
            assert (new_src, new_dest) not in src_dest_messages
            src_dest_messages[(new_src, new_dest)] = msg
            node_factors[new_dest] = node_factors[new_dest].factor_mult(msg)
        time_dur = time.perf_counter() - start_time
        _logger.info(f"Message passing done: {time_dur:.2f} sec")
        if config.DEBUG_LVL > 0:
            for n_idx, fac in node_factors.items():
                expected_vars = self._nodes[n_idx].cluster
                assert frozenset(
                    fac.vs) == expected_vars, frozenset(fac.vs) - expected_vars
        return src_dest_messages, node_factors

    def init_jointree_from_samiam(self) -> None:
        assert self._method == "samiam" and not self._nodes
        self._java_jt = JavaInflibWrapper().get_jointree(
            self.network_name, **self._gen_tree_args)
        self._original_width = self._java_jt.width
        # _logger.info(f"JT original width: {self._original_width}")
        var_map = {}
        for old_idx, name in enumerate(self._java_jt.var_names):
            var_map[old_idx] = self.var_name_to_idx[name]
        to_new_indices = lambda vs: frozenset(var_map[v] for v in vs)
        remained_fams = set(range(self.n_vars))
        _fam_var_set = [frozenset(v) for v in self.family_var_indices]
        seen = set()

        def recur(node_idx, p_idx: int) -> frozenset[int]:
            seen.add(node_idx)
            fams = set()
            cluster = to_new_indices(self._java_jt.clusters[node_idx])

            sep_dict: dict[int, frozenset[int]] = {}

            feas_dict = {}
            all_c_feas: set[int] = set()
            for j, s in self._java_jt.separators[node_idx].items():
                sep_dict[j] = to_new_indices(s)
                if j not in seen:
                    c_feas = recur(j, node_idx)
                    if c_feas:
                        feas_dict[j] = c_feas
                        all_c_feas.update(c_feas)

            for v in cluster:
                if v < self.n_features:
                    assert v in remained_fams
                    assert _fam_var_set[v] <= cluster
                    all_c_feas.add(v)
                if v in remained_fams and (_fam_var_set[v] <= cluster):
                    fams.add(v)
                    remained_fams.remove(v)
            p_feas = frozenset(range(self.n_features)) - all_c_feas
            if p_idx >= 0 and p_feas:
                feas_dict[p_idx] = p_feas
            else:
                assert not p_feas
            j_node = JointTreeNode(node_idx,
                                   separators=sep_dict,
                                   cluster=cluster,
                                   feas=feas_dict,
                                   fams=frozenset(fams))
            self._nodes[node_idx] = j_node

            return frozenset(all_c_feas)

        recur(0, -1)
        assert not remained_fams
        self.select_root_node(0)
        self.maybe_split_nodes(self.root, -1)

    def maybe_split_nodes(self, node: JointTreeNode, p_idx: int) -> None:
        # no more than 2 downstream children with features
        # create a new node, put two neighbors with the old node
        # connect new node to the rest of neighbors and old node
        remained_feas_nei_indices = set(node.feas.keys()) - {p_idx}

        to_visit: list[tuple[JointTreeNode, int]] = []
        while True:
            visit_order = self._compute_ordered_downstream_neighbors(
                node.idx, p_idx)
            next_idx = max(self._nodes.keys()) + 1
            fam_feas = frozenset(i for i in node.cluster
                                 if i < self.n_features)
            if fam_feas:
                assert fam_feas <= node.fams
                if not visit_order:
                    return
                leaf_cluster = frozenset(
                    itertools.chain.from_iterable(self.family_var_indices[v]
                                                  for v in fam_feas))
                assert leaf_cluster <= node.cluster
                leaf_feas = {
                    node.idx:
                    frozenset(itertools.chain.from_iterable(
                        node.feas.values()))
                }
                leaf_seps = {node.idx: leaf_cluster - fam_feas}
                self._nodes[next_idx] = JointTreeNode(next_idx, leaf_seps,
                                                      leaf_feas, leaf_cluster,
                                                      fam_feas)
                node.cluster -= fam_feas
                node.fams -= fam_feas
                node.feas[next_idx] = fam_feas
                node.separators[next_idx] = leaf_seps[node.idx]
                remained_feas_nei_indices.add(next_idx)
                continue
            if len(visit_order) <= 2:
                for nei in visit_order:
                    to_visit.append((nei, node.idx))
                break

            assert len(node.feas.keys() -
                       {p_idx}) == len(remained_feas_nei_indices)
            chosen_nei = visit_order[0]
            chosen_nei_idx = chosen_nei.idx
            assert chosen_nei_idx in remained_feas_nei_indices
            remained_nei_feas = node.feas[chosen_nei_idx]
            assert remained_nei_feas
            new_node_sep_dict = {node.idx: node.cluster}
            new_node_feas = {node.idx: remained_nei_feas}
            move_neighbors = frozenset(
                node.feas.keys()) - {p_idx, chosen_nei_idx}
            assert move_neighbors < remained_feas_nei_indices
            feas_set = set()
            for nei_idx in move_neighbors:
                nei = self._nodes[nei_idx]
                nei.separators[next_idx] = nei.separators[node.idx]
                del nei.separators[node.idx]
                assert nei.feas[node.idx]
                nei.feas[next_idx] = nei.feas[node.idx]
                del nei.feas[node.idx]

                new_node_sep_dict[nei_idx] = node.separators[nei_idx]
                del node.separators[nei_idx]
                assert node.feas[nei_idx]
                assert not (feas_set & node.feas[nei_idx])
                feas_set.update(node.feas[nei_idx])
                new_node_feas[nei_idx] = node.feas[nei_idx]
                del node.feas[nei_idx]

            node.separators[next_idx] = node.cluster
            node.feas[next_idx] = frozenset(feas_set)
            assert node.feas[next_idx]
            # old node keeps all the fams
            new_node = JointTreeNode(next_idx, new_node_sep_dict,
                                     new_node_feas, node.cluster, frozenset())
            self._nodes[next_idx] = new_node


            remained_feas_nei_indices.remove(chosen_nei_idx)
            assert (new_node_feas.keys() -
                    {node.idx}) == remained_feas_nei_indices
            to_visit.append((chosen_nei, node.idx))
            p_idx = node.idx
            node = new_node

        for nei, p_idx in to_visit:
            self.maybe_split_nodes(nei, p_idx)

    def _init_jointree_from_c2d_dtree(self, c2d_args: dict) -> None:
        assert self._method == "c2d" and not self._nodes
        self._c2d_dtree = C2DDTreeNode.gen_dtree(self.family_var_indices,
                                                 c2d_args)

        self._original_width = self._c2d_dtree[-1].get_width()

        seen_feas = self._init_from_dtree(self._c2d_dtree[-1], -1, frozenset(),
                                          set())

        assert len(seen_feas) == self.n_features
        self.select_root_node(self._c2d_dtree[-1].idx)
        assert self.root_idx >= 0
        self.maybe_split_nodes(self.root, -1)

    def select_root_node(self, start_idx: int) -> None:
        self.root_candidates: dict[int, int] = {}
        # start with any node in the joint tree
        stack = [start_idx]
        added = {stack[0]}
        while stack:
            n_idx = stack.pop()
            assert n_idx in added
            node = self._nodes[n_idx]
            assert n_idx not in self.root_candidates
            new_neighbors = node.separators.keys() - added
            stack.extend(new_neighbors)
            added.update(new_neighbors)
            if self.y_idx in node.cluster:
                score = 0
                for neighbor_idx in node.feas:
                    assert neighbor_idx in self._nodes
                    assert neighbor_idx in node.separators
                    if self.y_idx not in node.separators[neighbor_idx]:
                        neighbor = self._nodes[neighbor_idx]
                        assert self.y_idx not in neighbor.cluster
                        score += len(node.feas[neighbor_idx])
                self.root_candidates[n_idx] = score
        self.root_idx = max(self.root_candidates.items(),
                            key=lambda x: x[1])[0]
        self.best_root_score = self.root_candidates[self.root_idx]
        self.root = self._nodes[self.root_idx]
        assert self.root_idx >= 0

    def _init_from_dtree(self, dtnode: C2DDTreeNode, p_idx: int,
                         p_sep: frozenset[int],
                         seen_fams: set[int]) -> frozenset[int]:
        # return seen features
        assert dtnode.idx not in self._nodes
        sep_dict = {}
        feas_dict = {}
        if p_idx >= 0:
            sep_dict[p_idx] = p_sep
        ret_features: set[int] = set()
        if dtnode.children is None:
            assert p_idx >= 0
            assert dtnode.fam_idx not in seen_fams
            seen_fams.add(dtnode.fam_idx)
            fam_set: frozenset[int] = frozenset([dtnode.fam_idx])
        else:
            neighbor_list: list[JointTreeNode] = []
            fams: list[int] = []
            for c in dtnode.children:
                assert c.idx not in self._nodes
                if c.cluster <= dtnode.cluster and c.children is None:
                    if c.fam_idx not in seen_fams:
                        fams.append(c.fam_idx)
                        seen_fams.add(c.fam_idx)
                    continue
                sep = c.cluster & dtnode.cluster
                neighbor_feas = self._init_from_dtree(c, dtnode.idx, sep,
                                                      seen_fams)
                if neighbor_feas:
                    ret_features.update(neighbor_feas)
                    feas_dict[c.idx] = neighbor_feas
                sep_dict[c.idx] = sep
                neighbor = self._nodes[c.idx]
                neighbor_list.append(neighbor)
                if len(neighbor_feas) < self.n_features:
                    neighbor.feas[dtnode.idx] = frozenset(
                        range(self.n_features)) - neighbor_feas
            may_absorb_list: list[JointTreeNode] = []
            if config.ABSORB_NEIGHBOR_CLUSTER:
                for neighbor in neighbor_list:
                    assert neighbor.idx in self._nodes
                    to_absorb = all([
                        neighbor.cluster <= dtnode.cluster,
                    ])
                    if to_absorb:
                        may_absorb_list.append(neighbor)
                        continue
                may_absorb_list.sort(key=lambda x: len(x.separators),
                                     reverse=True)
            while may_absorb_list:
                # get neighbor with fewest neighbor node
                neighbor = may_absorb_list.pop()
                sep_size_after = len(sep_dict) + len(may_absorb_list) + len(
                    neighbor.separators) - 1
                sz_limit = 2 if self.y_idx in dtnode.cluster else 3
                if sep_size_after <= sz_limit:
                    self._absorb_neighbor_inplace(dtnode, sep_dict, feas_dict,
                                                  fams, neighbor)
            fam_set = frozenset(fams)
        if p_idx >= 0:
            p_feas = frozenset(range(self.n_features)) - ret_features - fam_set
            feas_dict[p_idx] = p_feas
        assert feas_dict.keys() <= sep_dict.keys()
        feas_keys = list(feas_dict.keys())
        for k in feas_keys:
            if not feas_dict[k]:
                del feas_dict[k]
        j_node = JointTreeNode(idx=dtnode.idx,
                               separators=sep_dict,
                               cluster=dtnode.cluster,
                               feas=feas_dict,
                               fams=fam_set)
        if self.root_idx < 0 and self.y_idx in j_node.cluster:
            self.root_idx = j_node.idx

        self._nodes[j_node.idx] = j_node
        ret_features.update([i for i in j_node.fams if i < self.n_features])

        # return j_node
        return frozenset(ret_features)

    def _absorb_neighbor_inplace(self, dtnode: C2DDTreeNode,
                                 sep_dict: dict[int, frozenset[int]],
                                 feas_dict: dict[int, frozenset[int]],
                                 fams: list[int],
                                 neighbor: JointTreeNode) -> None:
        # modify sep and fams inplace
        fams.extend(neighbor.fams)
        if neighbor.idx in feas_dict:
            del feas_dict[neighbor.idx]
        del sep_dict[neighbor.idx]
        for new_neighbor_idx in neighbor.separators:
            if new_neighbor_idx == dtnode.idx:
                continue
            new_neighbor = self._nodes[new_neighbor_idx]
            sep = dtnode.cluster & new_neighbor.cluster
            if new_neighbor_idx in neighbor.feas:
                feas_dict[new_neighbor_idx] = neighbor.feas[new_neighbor_idx]
            sep_dict[new_neighbor_idx] = sep
            new_neighbor_sep = new_neighbor.separators
            del new_neighbor_sep[neighbor.idx]
            new_neighbor_sep[dtnode.idx] = sep
        del self._nodes[neighbor.idx]

    def _check_joint_tree(self, n: JointTreeNode, p: JointTreeNode | None,
                          p_side_vars: frozenset[int],
                          fams: frozenset[int]) -> frozenset[int]:
        if config.DEBUG_LVL < 2:
            return frozenset()
        assert len(n.separators) >= 1
        assert len(n.feas) <= 3
        if p is None:
            assert len(n.feas) <= 2
        for v in n.feas.values():
            assert v
        if n.fams:
            assert all([i >= 0 for i in n.fams])
            assert not (n.fams & fams)
            fams |= n.fams
        if p is not None:
            assert p.idx in n.separators
            sep = n.separators[p.idx]
            assert sep <= n.cluster
            assert p.cluster & n.cluster == sep
            assert p_side_vars & n.cluster == sep
            assert not (p_side_vars & (n.cluster - sep))
        p_side_vars = p_side_vars | n.cluster
        for other_idx, sep in n.separators.items():
            if p is None and other_idx < 0:
                continue
            if p is not None and other_idx == p.idx:
                continue
            neighbor = self._nodes[other_idx]
            fams = self._check_joint_tree(neighbor, n, p_side_vars, fams)
        for other_idx, feas in n.feas.items():
            assert feas
            assert other_idx in n.separators
            assert other_idx in self._nodes
        return fams

    def rearrange_cpt(self) -> list[FactorMat]:
        cpt_facs: list[FactorMat] = [None] * self.n_vars
        for var_name, original_cpt in self.bn.cpts.items():
            var_idx = self.var_name_to_idx[var_name]
            parents_names = self.bn.parents[var_name]
            parents_indices = [self.var_name_to_idx[p] for p in parents_names]
            original_vs = parents_indices + [var_idx]
            vs = list(sorted(original_vs))
            if vs != original_vs:
                _logger.debug(
                    f"Rearranging CPT of {var_name} | parents: {parents_names}"
                )
            original_shape = tuple(self.var_n_ary[original_vs].tolist())
            assert original_cpt.shape == original_shape
            new_axis = np.searchsorted(vs, original_vs)
            mat = np.copy(np.moveaxis(original_cpt,
                                      np.arange(len(original_vs)), new_axis),
                          order='C')
            cpt_facs[var_idx] = FactorMat(
                mat, tuple(vs), tuple(self.var_n_ary[vs].tolist()),
                tuple(self.ordered_var_names[i] for i in vs),
                frozenset(parents_indices))
        return cpt_facs

    def to_graphviz(self) -> graphviz.Graph:
        assert self._nodes
        g = graphviz.Graph(f'{self.network_name}-jointtree')

        def indices_to_names(indices) -> str:
            names = [
                f"{self.ordered_var_names[i]}({i})" for i in sorted(indices)
            ]
            return ", ".join(names)

        visited: set[int] = set()
        stack = [self.root]
        seen_features: set[int] = set()
        n_new_features_node: list[int] = []
        while stack:
            n = stack.pop()
            idx = n.idx
            visited.add(idx)
            decide_indicator = "**" if idx in self.decide_nodes else ""
            msg_down_indicator = "VV" if idx in self.downward_msg_nodes else ""
            label_strs = [
                f"{decide_indicator}{msg_down_indicator}Node {idx}",
                f"Cluster: {indices_to_names(n.cluster)}",
                f"Fams: {indices_to_names(n.fams)}"
            ]
            node_attr = {}
            visit_order_edges = {}
            features = {x for x in n.cluster if x < self.n_features}
            if n.idx in self.ordered_downstream_neighbors:
                m = self.ordered_downstream_neighbors[n.idx]
                node_attr['color'] = 'red'
                visit_order_edges = {x.idx: i for i, x in enumerate(m)}
            if self.y_idx in n.cluster:
                node_attr['shape'] = 'box'
            if features:
                new_features = features - seen_features
                if new_features:
                    n_new_features_node.append(len(new_features))
                    label_strs.append(
                        f"{len(new_features)} new features: {indices_to_names(new_features)}"
                    )
                    node_attr['fontcolor'] = 'green'
                    seen_features.update(new_features)
            g.node(name=str(idx), label="\n".join(label_strs), **node_attr)
            for neighbor_idx, sep in n.separators.items():
                if neighbor_idx in visited:
                    continue
                stack.append(self._nodes[neighbor_idx])
                edge_color = 'Black'
                edge_label = f"{len(sep)}"
                if neighbor_idx in visit_order_edges:
                    edge_color = 'Red'
                    edge_label += f"\n#{visit_order_edges[neighbor_idx]}"
                g.edge(str(n.idx),
                       str(neighbor_idx),
                       label=edge_label,
                       color=edge_color)
        assert len(seen_features) == self.n_features
        return g

    def _create_directed_feature_map(self, n: JointTreeNode,
                                     p_idx: int) -> DirectedFeatureMap | None:
        assert n.idx not in self._directed_feature_maps
        d: dict[int, frozenset[int]] = {}
        may_absorb = config.ABSORB_NEIGHBOR_FEATURE
        for neighbor_idx in n.separators:
            if neighbor_idx == p_idx:
                continue
            neighbor = self._nodes[neighbor_idx]
            if len(neighbor.separators) == 1:
                s = frozenset(i for i in neighbor.fams if i < self.n_features)
                if s:
                    d[neighbor.idx] = s
                continue
            neighbor_map = self._create_directed_feature_map(neighbor, n.idx)
            if neighbor.idx not in self._node_may_absorb_features:
                may_absorb = False
            neighbor_side_features = {
                x
                for x in neighbor.cluster if x < self.n_features
            }
            if neighbor_map is not None:
                neighbor_side_features.update(
                    neighbor_map.downstream_neighbor_features)
            if neighbor_side_features:
                d[neighbor.idx] = frozenset(neighbor_side_features)
        if d:
            downsteam_feas: frozenset[int] = frozenset.union(*(d.values()))
            after_size = len(n.cluster | downsteam_feas)
            may_absorb = may_absorb and after_size <= config.FAC_SZ_LIMIT
            if may_absorb:
                self._node_may_absorb_features.add(n.idx)
                if self._decide_absorb(n, d):
                    n.cluster |= downsteam_feas
                    for neighbor_idx, f_set in d.items():
                        neighbor = self._nodes[neighbor_idx]
                        n.separators[neighbor_idx] |= f_set
                        self._add_feature_to_path(neighbor_idx, f_set, n.idx)
                        del n.feas[neighbor_idx]
                    return None
            m = DirectedFeatureMap(idx=n.idx, p_idx=p_idx, d=d)
            self._directed_feature_maps[n.idx] = m
            return m
        if may_absorb:
            self._node_may_absorb_features.add(n.idx)
        return None

    def _add_feature_to_path(self, n_idx: int, fea_set: frozenset[int],
                             p_idx: int) -> None:
        n = self._nodes[n_idx]
        assert p_idx >= 0
        assert not (fea_set & n.separators[p_idx])
        n.separators[p_idx] |= fea_set
        if fea_set <= n.cluster:
            assert n.idx not in self._directed_feature_maps
            return
        assert n_idx in self._node_may_absorb_features
        _expected_down = fea_set - n.cluster
        n.cluster |= fea_set
        d = self._directed_feature_maps[n.idx].d
        _actual_down = frozenset.union(*(d.values()))
        assert _expected_down == _actual_down
        for neighbor_idx, s in d.items():
            n.separators[neighbor_idx] |= s
            self._add_feature_to_path(neighbor_idx, s, n_idx)
            del n.feas[neighbor_idx]
        del self._directed_feature_maps[n.idx]

    def _decide_absorb(self, n: JointTreeNode, d: dict[int,
                                                       frozenset]) -> bool:
        assert d
        n_total_feas = len(frozenset.union(*(d.values())))
        if self.y_idx in n.cluster:
            return False
        if n_total_feas > config.MAX_LEAF_FEATURES_RATIO(self.n_features):
            return False
        max_fea_size = max(len(x) for x in d.values())
        max_sep_size = max(len(n.separators[i]) for i in d)
        return max_fea_size < max_sep_size
