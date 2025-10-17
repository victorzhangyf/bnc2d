import logging

logger = logging.getLogger("compile_bnc")
import typing, dataclasses, enum, itertools, collections, functools

import numpy as np
import graphviz

import global_config

config = global_config.Config()
FeaturesInst = np.ndarray[tuple[int], np.dtype[np.int8]]
FeaturesInstSet = np.ndarray[tuple[int, int], np.dtype[np.int8]]
FeaturesInstMask = np.ndarray[tuple[int, ...], np.dtype[np.bool]]


class NNFNodeType(enum.IntEnum):
    AND = 1
    OR = 2
    LITERAL = 4
    CONST = 8


@dataclasses.dataclass(slots=True, weakref_slot=True, frozen=True)
class NNFNode:
    idx: int
    ma: int  # max
    mi: int  # min
    typ: NNFNodeType
    payload: tuple[int, ...]

    def __str__(self) -> str:
        if self.typ == NNFNodeType.LITERAL:
            assert len(self.payload) == 2
            var_idx, pos = self.payload
            s = f"X{var_idx}({pos})"
        else:
            s = "/\\" if self.typ == NNFNodeType.AND else "\\/"
            s += str(self.payload)
        return s

    def __post_init__(self) -> None:
        if self.typ == NNFNodeType.OR:
            assert len(self.payload) >= 2
        elif self.typ != NNFNodeType.CONST:
            assert len(self.payload) == 2


def _set_or_mask(mask: np.ndarray[tuple[int, ...], np.dtype[np.bool]],
                 rows: FeaturesInstSet) -> None:
    for row in rows:
        tuple_idx = tuple(slice(None) if x == -1 else x for x in row.tolist())
        if config.DEBUG_CHECK_DETERMINISTIC:
            # print("NOT DETERMINISTIC")
            assert np.all(~mask[tuple_idx])
        mask[tuple_idx] = True


class NNF:
    FALSE = NNFNode(idx=0, typ=NNFNodeType.CONST, payload=(0, ), ma=-1, mi=-1)
    TRUE = NNFNode(idx=1, typ=NNFNodeType.CONST, payload=(1, ), ma=-1, mi=-1)

    def __init__(self, network_name: str, feature_nary: tuple[int, ...],
                 feature_to_lvl: tuple[int, ...]) -> None:
        self.nary = feature_nary
        self.fea2lvl = feature_to_lvl
        self.n_features = len(feature_to_lvl)
        self.network_name = network_name
        self._nnf_idx_to_node: list[NNFNode] = [NNF.FALSE, NNF.TRUE]
        self.NNF_FALSE_IDX, self.NNF_TRUE_IDX = 0, 1

        self._lit_to_nnf_idx: dict[tuple[int, int], int] = {}

        # look up internal nodes
        self._and_children_to_nnf_idx: dict[tuple[int, int], int] = {}
        # or children are sorted by idx
        self._or_children_to_nnf_idx: dict[tuple[int, ...], int] = {}

    def get_graph_size(self, nnf_idx: int) -> tuple[int, int]:
        # return n_nodes, n_edges
        visited = set()
        n_edges = -1
        stack = [nnf_idx]
        while stack:
            i = stack.pop()
            n_edges += 1
            if i in visited:
                continue
            visited.add(i)
            assert i > 0
            if i == 1:
                continue
            n = self[i]
            if n.typ == NNFNodeType.AND or n.typ == NNFNodeType.OR:
                stack.extend(reversed(n.payload))

        return len(visited), n_edges

    def get_all_models(
            self,
            nnf_idx: int,
            features: frozenset[int] | None = None) -> FeaturesInstMask:
        ret, feature_nary = self._get_model_masks(nnf_idx, features)
        mask = np.zeros(feature_nary, dtype=np.bool)
        _set_or_mask(mask, ret)
        return mask

    def _get_model_masks(
        self, nnf_idx: int, features: frozenset[int] | None
    ) -> tuple[FeaturesInstSet, tuple[int, ...]]:
        ret = self._get_models_mask_recur(nnf_idx) - 1
        feature_nary = self.nary
        if features is not None:  # default select all features
            assert features
            delete_features = [
                i for i in range(self.n_features) if i not in features
            ]
            ret = np.delete(ret, delete_features, axis=1)
            feature_nary = tuple(feature_nary[x] for x in sorted(features))
        return ret, feature_nary

    @functools.cache
    def _get_models_mask_recur(self, nnf_idx: int) -> FeaturesInstSet:
        assert nnf_idx > 0
        if nnf_idx == 1:
            return np.zeros((1, self.n_features), dtype=np.int8)

        nnf_node = self[nnf_idx]
        if nnf_node.typ == NNFNodeType.LITERAL:
            var_idx, pos_idx = nnf_node.payload
            arr = np.zeros((1, self.n_features), dtype=np.int8)
            arr[0, var_idx] = pos_idx + 1
            return arr

        if nnf_node.typ == NNFNodeType.OR:
            assert len(nnf_node.payload) >= 2
            children_models = [
                self._get_models_mask_recur(i) for i in nnf_node.payload
            ]
            ret = np.vstack(children_models)
            if config.DEBUG_CHECK_DETERMINISTIC:
                _fea_cols = np.sum(ret, axis=0) > 0
                _t = ret[:, _fea_cols]
                _nary = np.max(_t, axis=0)
                _mask = np.zeros(_nary, dtype=np.bool)
                _set_or_mask(_mask, _t - 1)
        else:
            assert nnf_node.typ == NNFNodeType.AND
            assert len(nnf_node.payload) == 2
            child_l, child_r = nnf_node.payload
            l_models = self._get_models_mask_recur(child_l)
            r_models = self._get_models_mask_recur(child_r)
            n_combinations = len(l_models) * len(r_models)
            l_models_reshape = l_models[:, None, :]
            r_models_reshape = r_models[None, :, :]
            if config.DEBUG_CHECK_DECOMPOSABLE:
                assert np.all(l_models_reshape * r_models_reshape == 0)
            combined = l_models_reshape + r_models_reshape
            ret = combined.reshape(n_combinations, self.n_features)

        return ret

    def _merge_left(self, or_children: collections.deque[NNFNode],
                    seen: set[int], l2r: dict[int, int], r2l: dict[int, int],
                    l1_idx: int, l2_idx: int, r_idx: int) -> bool:
        # r_idx = l2r[l1_idx]
        del l2r[l1_idx]
        del r2l[r_idx]
        new_l_idx = self.make_OR(frozenset([l2_idx, l1_idx]))
        if new_l_idx == self.NNF_TRUE_IDX:
            idx = r_idx
        else:
            idx = self.make_AND(new_l_idx, r_idx)
        if idx not in seen:
            or_children.append(self[idx])
            seen.add(idx)
            return True
        return False

    def _merge_right(self, or_children: collections.deque[NNFNode],
                     seen: set[int], l2r: dict[int, int], r2l: dict[int, int],
                     r1_idx: int, r2_idx: int, l_idx: int) -> bool:
        # l_idx = r2l[r1_idx]
        del r2l[r1_idx]
        del l2r[l_idx]
        new_r_idx = self.make_OR(frozenset([r1_idx, r2_idx]))
        if new_r_idx == self.NNF_TRUE_IDX:
            idx = l_idx
        else:
            idx = self.make_AND(l_idx, new_r_idx)
        if idx not in seen:
            or_children.append(self[idx])
            seen.add(idx)
            return True
        return False

    @functools.cache
    def make_OR(self, or_children_indices: frozenset[int]) -> int:
        assert or_children_indices
        if len(or_children_indices) == 1:
            return tuple(or_children_indices)[0]
        or_children = collections.deque([self[i] for i in or_children_indices])
        seen = set(or_children_indices)
        lits: dict[int, set[int]] = collections.defaultdict(set)
        l2r: dict[int, int] = {}
        r2l: dict[int, int] = {}
        if config.SIMPLIFY_NNF:
            ma2l: dict[int, set[int]] = collections.defaultdict(set)
            mi2r: dict[int, set[int]] = collections.defaultdict(set)
            l_max, r_min = -1, self.n_features
        no_more_merge = False
        while True:
            if not or_children:
                if no_more_merge:
                    break
                or_children.extend([
                    self[i]
                    for i in itertools.chain.from_iterable(lits.values())
                ])
                no_more_merge = True
                continue

            assert len(l2r) == len(r2l)
            or_child = or_children.popleft()
            if or_child.typ == NNFNodeType.CONST:
                if or_child.idx == self.NNF_TRUE_IDX:
                    return self.NNF_TRUE_IDX
                continue
            if or_child.typ == NNFNodeType.OR:
                or_children.extend([self[i] for i in or_child.payload])
                seen.update(or_child.payload)
                continue
            if or_child.typ == NNFNodeType.LITERAL:
                i, _ = or_child.payload
                lits[i].add(or_child.idx)
                if len(lits[i]) == self.nary[i]:
                    return self.NNF_TRUE_IDX
                # continue
                if config.SIMPLIFY_NNF:
                    if or_child.ma <= l_max:
                        l1_idx = -1
                        while (or_child.ma <= l_max) and (l_max >= 0):
                            if ma2l[l_max]:
                                t = ma2l[l_max].pop()
                                if t in l2r:
                                    l1_idx = t
                                    break
                            else:
                                l_max -= 1
                        if l1_idx > 0:
                            no_more_merge |= self._merge_left(
                                or_children, seen, l2r, r2l, l1_idx,
                                or_child.idx, l2r[l1_idx])
                            continue
                    if or_child.mi >= r_min:
                        r1_idx = -1
                        while (or_child.mi >= r_min) and (r_min
                                                          < self.n_features):
                            if mi2r[r_min]:
                                t = mi2r[r_min].pop()
                                if t in r2l:
                                    r1_idx = t
                                    break
                            else:
                                r_min += 1
                        if r1_idx > 0:
                            no_more_merge |= self._merge_right(
                                or_children, seen, l2r, r2l, r1_idx,
                                or_child.idx, r2l[r1_idx])
                            continue
                continue
            assert or_child.typ == NNFNodeType.AND
            l_idx, r_idx = or_child.payload
            if r_idx in r2l:
                no_more_merge |= self._merge_left(or_children, seen, l2r, r2l,
                                                  r2l[r_idx], l_idx, r_idx)
                continue
            if l_idx in l2r:
                no_more_merge |= self._merge_right(or_children, seen, l2r, r2l,
                                                   l2r[l_idx], r_idx, l_idx)
                continue

            l2r[l_idx] = r_idx
            r2l[r_idx] = l_idx
            if config.SIMPLIFY_NNF:
                l, r = self[l_idx], self[r_idx]
                l_max = max(l.ma, l_max)
                ma2l[l.ma].add(l_idx)
                r_min = min(r.mi, r_min)
                mi2r[r.mi].add(r_idx)

        assert len(l2r) == len(r2l)
        s: list[int] = list(itertools.chain.from_iterable(lits.values()))
        for l_idx, r_idx in l2r.items():
            c_idx = self.make_AND(l_idx, r_idx)
            assert c_idx not in s
            s.append(c_idx)
        assert s
        if len(s) == 1:
            return s[0]
        if config.DEBUG_LVL > 0:
            assert len(s) == len(frozenset(s))
        return self._make_final_internal(NNFNodeType.OR, tuple(sorted(s)))

    def make_lit(self, v: int, pos: int) -> int:
        lit = (v, pos)
        if lit not in self._lit_to_nnf_idx:
            assert v >= 0 and pos >= 0
            assert pos < self.nary[v]
            m = self.fea2lvl[v]
            n = NNFNode(idx=len(self._nnf_idx_to_node),
                        typ=NNFNodeType.LITERAL,
                        payload=lit,
                        ma=m,
                        mi=m)
            self._nnf_idx_to_node.append(n)
            self._lit_to_nnf_idx[lit] = n.idx
        nnf_idx = self._lit_to_nnf_idx[lit]
        return nnf_idx

    def make_AND_binary_tree(self, xs: tuple[int, ...]) -> int:
        l = len(xs)
        if l == 1:
            return xs[0]
        if l == 2:
            return self.make_AND(*xs)
        mid = l // 2
        return self.make_AND(self.make_AND_binary_tree(xs[:mid]),
                             self.make_AND_binary_tree(xs[mid:]))

    def make_AND(self, l: int, r: int) -> int:
        if config.DEBUG_LVL > 0:
            assert l != r
            assert l > self.NNF_FALSE_IDX
            assert r > self.NNF_FALSE_IDX
        if l == self.NNF_TRUE_IDX:
            return r
        if r == self.NNF_TRUE_IDX:
            return l
        if config.DEBUG_LVL > 0:
            nl, nr = self[l], self[r]
            assert nl.ma < nr.ma
            if nl.typ == NNFNodeType.LITERAL and nr.typ == NNFNodeType.LITERAL:
                assert nl.payload[0] != nr.payload[0]
        return self._make_final_internal(NNFNodeType.AND, (l, r))

    def _make_final_internal(self, typ: NNFNodeType,
                             children_indices: tuple[int, ...]) -> int:
        if config.DEBUG_LVL > 1:
            assert typ != NNFNodeType.LITERAL
            assert len(children_indices) == len(frozenset(children_indices))
            if typ == NNFNodeType.OR:
                assert tuple(sorted(children_indices)) == children_indices
        cache = self._or_children_to_nnf_idx if typ == NNFNodeType.OR else self._and_children_to_nnf_idx
        if children_indices in cache:
            cached_idx = cache[children_indices]
            assert cached_idx > 1
            return cached_idx

        nnf_idx = len(self._nnf_idx_to_node)
        ma = max(self[i].ma for i in children_indices)
        mi = min(self[i].mi for i in children_indices)
        n = NNFNode(idx=nnf_idx,
                    typ=typ,
                    payload=children_indices,
                    ma=ma,
                    mi=mi)

        self._nnf_idx_to_node.append(n)
        cache[children_indices] = nnf_idx
        return nnf_idx

    def node_to_str(self, node_idx: int) -> str:
        n = self._nnf_idx_to_node[node_idx]
        if n.typ == NNFNodeType.CONST:
            s = "T" if n else "F"
        elif n.typ == NNFNodeType.LITERAL:
            assert len(n.payload) == 2
            var_idx, pos = n.payload
            s = f"X{var_idx}_{pos}"
        else:
            s = " x " if n.typ == NNFNodeType.AND else " + "
            op_arr = [self.node_to_str(i) for i in n.payload]
            s = f"({s.join(op_arr)})"
        return s

    def __getitem__(self, node_idx: int) -> NNFNode:
        return self._nnf_idx_to_node[node_idx]

    def to_graphviz(self, node_idx: int,
                    ordered_var_names: tuple[str, ...]) -> graphviz.Digraph:
        graph = graphviz.Digraph(f'{self.network_name}-nnf')
        stack: list[tuple[int, int]] = [(-1, node_idx)]
        visited: set[int] = set()
        num_leafs = 0
        num_edges = 0
        while stack:
            parent_idx, node_idx = stack.pop()
            node = self[node_idx]
            if parent_idx >= 0:
                graph.edge(str(parent_idx), str(node_idx))
                num_edges += 1
            if node_idx in visited:
                continue
            visited.add(node_idx)
            node_id_str = str(node_idx)
            node_attr: dict[str, str] = {}
            if node_idx == self.NNF_TRUE_IDX:
                label = "T"
            elif node.typ == NNFNodeType.LITERAL:
                assert len(node.payload) == 2
                num_leafs += 1
                var_idx = node.payload[0]
                var_name = ordered_var_names[var_idx]
                if self.nary[var_idx] == 2:
                    label = "" if node.payload[1] else "¬"
                    label += var_name
                else:
                    label = f"{var_name}_{node.payload[1]}"
            else:
                assert len(node.payload) >= 2
                for c in reversed(node.payload):
                    stack.append((node_idx, c))
                label = "AND" if node.typ == NNFNodeType.AND else "OR"
            graph.node(name=node_id_str, label=label, **node_attr)
        logger.info(
            f"graphviz NNF n_leafs: {num_leafs}, n_nodes: {len(visited)}, n_edges: {num_edges}"
        )
        return graph
