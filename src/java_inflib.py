import  dataclasses

import jpype
import jpype.imports
import global_config

config = global_config.Config()
import utils


@dataclasses.dataclass(frozen=True)
class InflibJoinTree:
    network_name: str
    var_names: tuple[str, ...]
    clusters: dict[int, frozenset[int]]
    separators: dict[int, dict[int, frozenset[int]]]

    @property
    def width(self) -> int:
        return max(len(x) for x in self.clusters.values()) - 1

    def __post_init__(self) -> None:
        self._debug_check_tree()

    def _debug_check_tree(self) -> None:
        if config.DEBUG_LVL > 0:
            for i, v in self.clusters.items():
                assert isinstance(i, int)
                assert all(isinstance(j, int) for j in v)
            for i, d in self.separators.items():
                assert isinstance(i, int)
                for k, vv in d.items():
                    assert isinstance(k, int)
                    assert all(isinstance(j, int) for j in vv)
            _ref_bn = utils.read_hugin_network(self.network_name)
            assert frozenset(_ref_bn.cpts.keys()) == frozenset(self.var_names)
            _seen_vars = frozenset.union(*(self.clusters.values()))
            n_vars = len(self.var_names)
            assert len(_seen_vars) == n_vars
            assert all((i < n_vars and i >= 0) for i in _seen_vars)


class JavaInflibWrapper:
    INFLIB_JAR_PATH = "lib/inflib.jar"

    _instance: "JavaInflibWrapper | None" = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            assert not jpype.isJVMStarted()
            jpype.startJVM(classpath=[JavaInflibWrapper.INFLIB_JAR_PATH])
        return cls._instance

    def get_jointree(
            self,
            network_name: str,
            seed: int,
            elim_last: frozenset[str] | None = None) -> InflibJoinTree:
        import java.util
        from il2.inf import Algorithm
        from il2.inf.structure import EliminationOrders
        from il2.model import BayesianNetwork
        from il2.bridge import IO
        from il2.util import IntSet
        network_path = config.NETWORK_BASE_DIR / network_name
        bn: BayesianNetwork = IO.readNetwork(str(network_path))
        n_vars = bn.size()
        var_names = tuple(str(bn.domain().name(i)) for i in range(n_vars))
        if seed is not None:
            j_rand = java.util.Random(jpype.JInt(seed))
        else:
            j_rand = java.util.Random()
        if elim_last is None:
            eor = EliminationOrders.minFill(java.util.Arrays.asList(bn.cpts()),
                                            jpype.JInt(1), j_rand)

        else:
            name_to_idx = {x: i for i, x in enumerate(var_names)}
            elim_last_indices = [name_to_idx[x] for x in elim_last]
            eor = EliminationOrders.constrainedMinFill(
                java.util.Arrays.asList(bn.cpts()),
                IntSet(jpype.JInt[:](elim_last_indices)))
        o2jt = Algorithm.Order2JoinTree.traditional
        jt: EliminationOrders.JT = o2jt.induce(bn, eor.order)
        clusters = {
            int(i): frozenset(int(j) for j in list(v.toArray()))
            for i, v in dict(jt.clusters).items()
        }
        separators = {}
        for n_idx in list(jt.tree.vertices().toArray()):
            n_idx = int(n_idx)
            my_cluster = clusters[n_idx]
            d: dict[int, frozenset[int]] = {}
            for other_idx in list(jt.tree.neighbors(n_idx).toArray()):
                other_idx = int(other_idx)
                other_cluster = clusters[other_idx]
                s = my_cluster & other_cluster
                d[other_idx] = s
            separators[n_idx] = d
        return InflibJoinTree(network_name, var_names, clusters, separators)


def _test():
    NETWORK_NAME = "binarynetworks/andes.net"
    target_var = "CHOOSE47"
    inflib_wrapper = JavaInflibWrapper()
    jointree = inflib_wrapper.get_jointree(NETWORK_NAME, 42,
                                           frozenset([target_var]))
    jointree._debug_check_tree()
    print(f"width {jointree.width}")


if __name__ == "__main__":
    _test()
