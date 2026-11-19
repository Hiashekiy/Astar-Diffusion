# field-guided-search

独立最小包：**占用栅格地图 + 一条引导路径（可以穿墙）→ 全局引导场 `F(x)` → 在该场上的路径搜索结果**。

只依赖 `numpy`（`scipy` 可选，装了场构建快 20 倍以上）。复制整个目录即可独立使用：
`pip install -e .`，或直接把 `src/field_guided_search` 放进你项目的 `sys.path`。

```python
from field_guided_search import GuidanceField

gf    = GuidanceField(occupancy)      # (H, W)，0 = 自由
field = gf.build(trajectory)          # (N, 2) 折线，[row, col]；允许穿过障碍
res   = gf.search(start, goal)        # 纯引导场；alpha=0.5 融合目标距离

res["path"]      # (M, 2) 整数栅格路径 [row, col]
res["length"]    # 路径长度（步数）
res["expanded"]  # 扩展节点数
res["status"]    # success | unreachable | cutoff | invalid
```

---

## 安装

```bash
pip install -e .              # 只依赖 numpy
pip install -e ".[fast]"      # 加 scipy：场构建快 20 倍以上（推荐）
```

`scipy` 缺失时自动回落到纯 Python 实现，结果完全相同（实测最大差 6e-16）。

## 最小示例

```python
import numpy as np
from field_guided_search import GuidanceField

occ = np.zeros((64, 64), np.int8)          # 0 = 自由，1 = 障碍
occ[0, :] = occ[-1, :] = 1
occ[:, 0] = occ[:, -1] = 1
occ[20, 5:50] = 1

trajectory = np.array([[10., 10.], [10., 45.], [45., 50.]])   # 引导路径
start, goal = (5, 5), (55, 55)

gf = GuidanceField(occ, lam=0.02, weight="linear")
gf.build(trajectory)

print("A*         ", gf.astar(start, goal)["expanded"], "expanded")
print("field-only ", gf.search(start, goal)["expanded"], "expanded")
```

## API

### `GuidanceField(occupancy, ...)`

| 参数 | 默认 | 说明 |
|---|---|---|
| `occupancy` | — | `(H, W)`，`0` 自由，其它值障碍 |
| `lam` | `0.02` | 衰减率：越大越"只在路径附近有效"，越小越全局。经验区间 `lam*L ≈ 1…4` |
| `weight` | `"linear"` | `uniform` / `linear`（w=u）/ `quadratic`（w=u²） |
| `num_samples` | `64` | 引导路径采样点数 |
| `source_method` | `"drop"` | `drop`（= M0，核心实验方案）：丢弃障碍内采样点、保留其余原始进度；`project`（= M1）：移到 `projection_radius` 内最近自由格 |
| `projection_radius` | `3` | `project` 的搜索半径 |
| `method` | `"auto"` | 场后端：`auto` / `scipy` / `heap` |
| `distance` | `"manhattan"` | 目标距离模型：`manhattan`（忽略障碍，**发表实验所用**）或 `geodesic`（自由空间 BFS 距离，会绕墙） |

| 方法 | 返回 |
|---|---|
| `build(trajectory)` | `(H, W)` 引导场，障碍与不可达处为 `NaN`，同时存入 `gf.field` / `gf.sources` |
| `search(start, goal, alpha=0.0, gamma=1.0, adaptive=False, ...)` | 搜索结果 dict |
| `astar(start, goal)` | 最短路径基线 dict |
| `is_valid(path, start, goal)` | `{'valid', 'length'}` 或 `{'valid': False, 'reason'}` |
| `normalised()` | `{'F_hat', 'field'}`，搜索实际使用的标量图 |

搜索结果的时间分三段，避免把预处理算成搜索开销：

| 键 | 含义 |
|---|---|
| `preprocess_ms` | 目标距离、场归一化、优先级图构建 |
| `search_ms` | 优先队列搜索本身（`time_ms` 是它的别名，保留兼容） |
| `total_ms` | 两者之和 |

### 搜索优先级

```
J(n) = alpha(n) · d̂_G(n)^gamma − (1 − alpha(n)) · F̂(n)
```

| 用法 | 含义 |
|---|---|
| `alpha=0`（默认） | 纯引导场跟随 |
| `alpha=1, gamma=1` | 纯目标距离贪心 |
| `0 < alpha < 1` | 线性融合（实测 α=0.25 最好：扩展数 0.20× A*，路径长 4%） |
| `gamma=0.5` | 非线性融合：靠近目标时更看重距离 |
| `adaptive=True` | `alpha` 随剩余距离从 `alpha_far=0.25` 变到 `alpha_near=0.75` |

`alpha` / `gamma` 作用在**归一化后**的 `F̂`、`d̂_G` 上，所以同一组参数在不同地图上含义一致。只有 `astar` 声称最短路径最优。

### 一次性函数

```python
from field_guided_search import field_from_trajectory, guided_search

field = field_from_trajectory(occ, trajectory, lam=0.02, weight="linear")
res   = guided_search(occ, start, goal, field=field, alpha=0.5)
res   = guided_search(occ, start, goal, trajectory=trajectory, alpha=0.5)

# 已经持有 Grid / 场时用底层的 Grid 级引擎（同名会混淆，故单独导出）
from field_guided_search import guided_search_grid
res = guided_search_grid(grid, start, goal, field, alpha=0.5)
```

### 与原实验代码的数值一致性

`tests/test_package_parity.py`（在工作区）逐项对照本包与 `src/guidance_field`：M0 源点、
场值、`d̂_G`、A\* 以及 α=0/0.25/0.5/1 的引导搜索 **expanded 计数完全相同**。改动本包后
务必重跑该测试，它是防止两者悄悄漂移的唯一保障。

## 原理

```
F(x) = max_i [ w(u_i) − lam · d_free(x, s_i) ]
```

* `u_i` 是第 i 个采样点在**原始路径**上的归一化弧长；处理碰撞时只删/移采样点，**不重新归一化** `u`，否则会凭空造出路径从未有过的进度；
* `d_free` 是四邻域测地距离，场不会穿墙（障碍格永远是 `NaN`）；
* 场等价于"沿路径衰减的势"，搜索沿场的等高线前进，所以比纯目标距离更快收敛到终点方向。

场构建：把 `C(x) = min_i[d + (w_max − w_i)/lam]` 展开成一次多源最短路，再用 `w_max − lam·C` 还原。默认后端是一个虚拟超级源的稀疏 Dijkstra（精确）；`method="heap"` 是等价的纯 Python 多源最大堆。

## 注意

* 坐标一律 `(row, col)`；matplotlib 需要 `(col, row)`。
* 引导路径若一个合法采样点都留不下，`build()` 直接报错而不是给出空场——那种"看起来能跑"的结果其实是盲搜。只剩 1 个采样点是允许的，对应退化场 `F(x) = w_1 − λ·d(x, s_1)`。
* 飞出地图边界的采样点会被**丢弃并计数**（`sources['num_out_of_bounds']`），不会被裁剪到边缘——裁剪会凭空造出路径从未经过的源点。
* `lam`、`weight`、`distance`、`num_samples`、起终点坐标都会校验，非法值抛 `ValueError`。
* `drop` 保留真实位置但会在进度轴留洞；`project` 保持采样完整但位置轻微偏移。碰撞严重时 `project` 更稳。
* 场与地图绑定（缓存了 CSR 邻接矩阵），换地图请新建对象；同一地图多次查询可复用场。

## 测试

```bash
python -m unittest discover -s tests      # 28 项
```

## 文件

```
src/field_guided_search/
    __init__.py   公开 API
    api.py        GuidanceField + 一次性函数
    grid.py       占用栅格、四邻域表、BFS 距离
    sources.py    路径重采样、障碍采样点投影/丢弃
    field.py      场构建（scipy 稀疏 Dijkstra / 纯 Python 多源堆）
    search.py     A*、引导搜索、归一化、路径校验
```
