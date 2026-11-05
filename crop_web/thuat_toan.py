"""
thuat_toan.py
=============
Cài đặt TỪ ĐẦU (chỉ dùng numpy/pandas/matplotlib, KHÔNG dùng scikit-learn) cho đề tài
"Hệ thống hỗ trợ lựa chọn cây trồng dựa trên điều kiện đất (N, P, K, pH)".

Nội dung
--------
1. Độ tạp chất: Gini, Entropy
2. MyDecisionTree  : cây quyết định (CART, tách nhị phân theo ngưỡng)
3. MyRandomForest  : rừng ngẫu nhiên (bootstrap + chọn ngẫu nhiên đặc trưng + bỏ phiếu + OOB)
4. Chỉ số đánh giá : ma trận nhầm lẫn, Accuracy, Precision, Recall, F1 (macro, weighted)
5. Chia K-fold phân tầng, tìm tham số bằng cross-validation
6. Kiểm định so sánh hai mô hình: McNemar chính xác, bootstrap cặp
7. Hàm dự đoán mẫu mới (top-3, cảnh báo ngoài miền)

Quy ước tên tham số giống scikit-learn (criterion, max_depth, ...) để dễ đối chiếu.
"""
import math
import time
import itertools
import numpy as np
import pandas as pd

THU_TU_CHI_SO = ["N", "P", "K", "pH"]


# =====================================================================
# 1. ĐỘ TẠP CHẤT
# =====================================================================
def gini(dem):
    """Gini = 1 - sum(p_k^2). `dem` là vector số mẫu của từng lớp tại một nút."""
    dem = np.asarray(dem, dtype=float)
    n = dem.sum()
    if n == 0:
        return 0.0
    p = dem / n
    return float(1.0 - np.sum(p ** 2))


def entropy(dem):
    """Entropy = - sum(p_k * log2(p_k)), quy ước 0*log0 = 0."""
    dem = np.asarray(dem, dtype=float)
    n = dem.sum()
    if n == 0:
        return 0.0
    p = dem[dem > 0] / n
    return float(-np.sum(p * np.log2(p)))


def do_tap_chat(dem, tieu_chi="gini"):
    return gini(dem) if tieu_chi == "gini" else entropy(dem)


def _tap_chat_hang(C, n, tieu_chi):
    """Độ tạp chất cho NHIỀU nút cùng lúc. C: (m, K) số mẫu từng lớp; n: (m,) tổng mẫu."""
    p = C / n[:, None]
    if tieu_chi == "gini":
        return 1.0 - np.sum(p ** 2, axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        return -np.sum(np.where(p > 0, p * np.log2(p), 0.0), axis=1)


def gain_theo_nguong(x, y, K, tieu_chi="gini", min_leaf=1):
    """
    Với MỘT đặc trưng x, tính độ giảm tạp chất (information gain) cho MỌI ngưỡng có thể.
    Ngưỡng nằm giữa hai giá trị liên tiếp đã sắp xếp. Trả về (ngưỡng, gain, hợp_lệ, số_mẫu_trái).
        gain = tạp_chất(cha) - [ n_trái/n * tạp_chất(trái) + n_phải/n * tạp_chất(phải) ]
    Dùng cộng dồn số mẫu từng lớp nên chỉ cần sắp xếp một lần: O(n log n).
    """
    n = len(y)
    o = np.argsort(x, kind="stable")
    xs = x[o]
    Y = np.zeros((n, K))
    Y[np.arange(n), y[o]] = 1.0
    cong_don = np.cumsum(Y, axis=0)
    tong = cong_don[-1]
    trai = cong_don[:-1]
    phai = tong - trai
    nl = np.arange(1, n, dtype=float)
    nr = n - nl
    cha = do_tap_chat(tong, tieu_chi)
    gain = cha - (nl * _tap_chat_hang(trai, nl, tieu_chi) + nr * _tap_chat_hang(phai, nr, tieu_chi)) / n
    hop_le = (xs[1:] != xs[:-1]) & (nl >= min_leaf) & (nr >= min_leaf)
    nguong = (xs[1:] + xs[:-1]) / 2.0
    return nguong, gain, hop_le, nl


# =====================================================================
# 2. CÂY QUYẾT ĐỊNH
# =====================================================================
class MyDecisionTree:
    """
    Cây quyết định phân loại (thuật toán tham lam kiểu CART).

    Tham số
    -------
    criterion         : "gini" hoặc "entropy"
    max_depth         : độ sâu tối đa (None = không giới hạn)
    min_samples_split : nút có ít hơn số mẫu này thì không tách nữa
    min_samples_leaf  : mỗi nút con sau khi tách phải có ít nhất số mẫu này
    max_features      : None (xét hết), số nguyên, hoặc "sqrt" (dùng trong Random Forest)
    random_state      : hạt giống để tái lập (chỉ ảnh hưởng khi max_features khác None)
    """

    def __init__(self, criterion="gini", max_depth=None, min_samples_split=2,
                 min_samples_leaf=1, max_features=None, random_state=42):
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.random_state = random_state

    # ---------- huấn luyện ----------
    def fit(self, X, y):
        self.feature_names_ = list(X.columns) if hasattr(X, "columns") else [f"x{i}" for i in range(np.shape(X)[1])]
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        y_idx = np.searchsorted(self.classes_, y)
        return self._fit_idx(X, y_idx, len(self.classes_), self.feature_names_)

    def _fit_idx(self, X, y_idx, K, feature_names=None):
        """Huấn luyện khi nhãn đã mã hoá thành số 0..K-1 (Random Forest dùng hàm này)."""
        self.K_ = K
        self.n_features_ = X.shape[1]
        if feature_names is not None:
            self.feature_names_ = feature_names
        self.rng_ = np.random.default_rng(self.random_state)
        self._feat, self._thr, self._left, self._right = [], [], [], []
        self._counts, self._n, self._imp, self._gain = [], [], [], []
        self._build(X, y_idx, 0)
        self.feature_ = np.array(self._feat)
        self.threshold_ = np.array(self._thr)
        self.left_ = np.array(self._left)
        self.right_ = np.array(self._right)
        self.counts_ = np.array(self._counts, dtype=float)
        self.n_samples_ = np.array(self._n, dtype=float)
        self.impurity_ = np.array(self._imp)
        self.gain_ = np.array(self._gain)
        for a in ("_feat", "_thr", "_left", "_right", "_counts", "_n", "_imp", "_gain", "rng_"):
            delattr(self, a)
        self._tinh_do_quan_trong()
        return self

    def _chon_dac_trung(self, p):
        mf = self.max_features
        if mf is None:
            return np.arange(p)
        if mf == "sqrt":
            mf = max(1, int(math.sqrt(p)))
        mf = min(int(mf), p)
        return self.rng_.choice(p, mf, replace=False)

    def _tach_tot_nhat(self, X, y):
        """Duyệt các đặc trưng được chọn, mọi ngưỡng; trả về (đặc trưng, ngưỡng, gain) tốt nhất."""
        best_f, best_t, best_g = None, None, 1e-12
        for f in self._chon_dac_trung(X.shape[1]):
            nguong, gain, hop_le, _ = gain_theo_nguong(X[:, f], y, self.K_, self.criterion, self.min_samples_leaf)
            if not hop_le.any():
                continue
            gain = np.where(hop_le, gain, -np.inf)
            i = int(np.argmax(gain))
            if gain[i] > best_g:
                best_f, best_t, best_g = int(f), float(nguong[i]), float(gain[i])
        return best_f, best_t, best_g

    def _build(self, X, y, depth):
        k = len(self._feat)
        dem = np.bincount(y, minlength=self.K_)
        n = len(y)
        imp = do_tap_chat(dem, self.criterion)
        self._feat.append(-1); self._thr.append(np.nan); self._left.append(-1); self._right.append(-1)
        self._counts.append(dem); self._n.append(n); self._imp.append(imp); self._gain.append(0.0)
        # điều kiện dừng
        if imp == 0.0 or n < self.min_samples_split or n < 2 * self.min_samples_leaf:
            return k
        if self.max_depth is not None and depth >= self.max_depth:
            return k
        f, t, g = self._tach_tot_nhat(X, y)
        if f is None:
            return k
        trai = X[:, f] <= t
        self._feat[k], self._thr[k], self._gain[k] = f, t, g
        self._left[k] = self._build(X[trai], y[trai], depth + 1)
        self._right[k] = self._build(X[~trai], y[~trai], depth + 1)
        return k

    def _tinh_do_quan_trong(self):
        """Độ quan trọng (MDI): tổng mức giảm tạp chất có trọng số mẫu mà mỗi đặc trưng đem lại."""
        imp = np.zeros(self.n_features_)
        for i in np.where(self.feature_ >= 0)[0]:
            l, r = self.left_[i], self.right_[i]
            giam = (self.n_samples_[i] * self.impurity_[i]
                    - self.n_samples_[l] * self.impurity_[l]
                    - self.n_samples_[r] * self.impurity_[r])
            imp[self.feature_[i]] += giam
        tong = imp.sum()
        self.feature_importances_ = imp / tong if tong > 0 else imp

    # ---------- dự đoán ----------
    def apply(self, X):
        """Trả về chỉ số nút lá mà từng mẫu rơi vào (duyệt đồng thời cho mọi mẫu)."""
        X = np.asarray(X, dtype=float)
        nut = np.zeros(len(X), dtype=int)
        while True:
            f = self.feature_[nut]
            dang_di = np.where(f >= 0)[0]
            if len(dang_di) == 0:
                break
            ni = nut[dang_di]
            sang_trai = X[dang_di, f[dang_di]] <= self.threshold_[ni]
            nut[dang_di] = np.where(sang_trai, self.left_[ni], self.right_[ni])
        return nut

    def predict_proba(self, X):
        la = self.apply(X)
        c = self.counts_[la]
        return c / c.sum(axis=1, keepdims=True)

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]

    # ---------- thông tin cây ----------
    def get_depth(self):
        def d(k):
            return 0 if self.feature_[k] < 0 else 1 + max(d(self.left_[k]), d(self.right_[k]))
        return d(0)

    def get_n_leaves(self):
        return int(np.sum(self.feature_ < 0))

    def get_n_nodes(self):
        return len(self.feature_)

    def export_rules(self, max_depth=4, ten_chi_so=None):
        """In cây dưới dạng luật if/else."""
        ten = ten_chi_so or self.feature_names_
        dong = []

        def duyet(k, d):
            le = "    " * d
            top = self.classes_[int(np.argmax(self.counts_[k]))]
            if self.feature_[k] < 0:
                dong.append(f"{le}→ {top}  (n={int(self.n_samples_[k])})")
            elif d >= max_depth:
                dong.append(f"{le}→ ... (đa số: {top}, n={int(self.n_samples_[k])})")
            else:
                dong.append(f"{le}nếu {ten[self.feature_[k]]} <= {self.threshold_[k]:.2f}:")
                duyet(self.left_[k], d + 1)
                dong.append(f"{le}ngược lại:")
                duyet(self.right_[k], d + 1)

        duyet(0, 0)
        return "\n".join(dong)


# =====================================================================
# 3. RỪNG NGẪU NHIÊN
# =====================================================================
class MyRandomForest:
    """
    Random Forest = nhiều cây quyết định, mỗi cây:
      (1) học trên một mẫu BOOTSTRAP (lấy ngẫu nhiên có hoàn lại n mẫu),
      (2) mỗi lần tách chỉ xét ngẫu nhiên `max_features` đặc trưng.
    Dự đoán: bỏ phiếu. vote="hard": mỗi cây một phiếu cho một lớp (đa số thắng);
             vote="soft": lấy trung bình xác suất lớp tại lá của các cây (cách scikit-learn dùng).
    Ước lượng OOB: mỗi mẫu chỉ được dự đoán bởi các cây KHÔNG thấy nó khi huấn luyện.
    """

    def __init__(self, n_estimators=100, criterion="gini", max_depth=None, min_samples_split=2,
                 min_samples_leaf=1, max_features="sqrt", bootstrap=True, vote="soft", random_state=42):
        self.n_estimators = n_estimators
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.vote = vote
        self.random_state = random_state

    def fit(self, X, y):
        self.feature_names_ = list(X.columns) if hasattr(X, "columns") else [f"x{i}" for i in range(np.shape(X)[1])]
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        K = len(self.classes_)
        y_idx = np.searchsorted(self.classes_, y)
        n = len(y)
        rng = np.random.default_rng(self.random_state)
        self.trees_, self.so_mau_duy_nhat_ = [], []
        oob_soft = np.zeros((n, K))
        oob_hard = np.zeros((n, K))
        oob_dem = np.zeros(n)
        for _ in range(self.n_estimators):
            ib = rng.integers(0, n, n) if self.bootstrap else np.arange(n)
            cay = MyDecisionTree(self.criterion, self.max_depth, self.min_samples_split, self.min_samples_leaf,
                                 self.max_features, int(rng.integers(2 ** 31 - 1)))
            cay._fit_idx(X[ib], y_idx[ib], K, self.feature_names_)
            cay.classes_ = self.classes_
            self.trees_.append(cay)
            self.so_mau_duy_nhat_.append(len(np.unique(ib)))
            if self.bootstrap:
                oob = np.setdiff1d(np.arange(n), ib)
                if len(oob):
                    pr = cay.predict_proba(X[oob])
                    oob_soft[oob] += pr
                    oob_hard[oob, np.argmax(pr, axis=1)] += 1
                    oob_dem[oob] += 1
        self.feature_importances_ = np.mean([t.feature_importances_ for t in self.trees_], axis=0)
        if self.bootstrap:
            m = oob_dem > 0
            self.oob_coverage_ = float(m.mean())
            diem = oob_soft if self.vote == "soft" else oob_hard
            self.oob_score_ = float(np.mean(np.argmax(diem[m], axis=1) == y_idx[m]))
        return self

    def predict_proba(self, X):
        X = np.asarray(X, dtype=float)
        K = len(self.classes_)
        if self.vote == "soft":
            return np.mean([t.predict_proba(X) for t in self.trees_], axis=0)
        phieu = np.zeros((len(X), K))
        for t in self.trees_:
            phieu[np.arange(len(X)), np.argmax(t.predict_proba(X), axis=1)] += 1
        return phieu / len(self.trees_)

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]

    def get_params(self):
        return dict(n_estimators=self.n_estimators, criterion=self.criterion, max_depth=self.max_depth,
                    min_samples_split=self.min_samples_split, min_samples_leaf=self.min_samples_leaf,
                    max_features=self.max_features, bootstrap=self.bootstrap, vote=self.vote)


def do_quan_trong_hoan_vi(mo_hinh, X, y, n_repeats=10, random_state=42):
    """
    Permutation importance: xáo trộn ngẫu nhiên MỘT cột rồi xem accuracy giảm bao nhiêu.
    Giảm càng nhiều => mô hình càng phụ thuộc vào chỉ số đó. Tính trên tập kiểm tra.
    """
    rng = np.random.default_rng(random_state)
    ten = list(X.columns)
    Xv = X.values.astype(float)
    y = np.asarray(y)
    goc = np.mean(mo_hinh.predict(Xv) == y)
    kq = []
    for j, c in enumerate(ten):
        giam = []
        for _ in range(n_repeats):
            Xp = Xv.copy()
            Xp[:, j] = rng.permutation(Xp[:, j])
            giam.append(goc - np.mean(mo_hinh.predict(Xp) == y))
        kq.append({"Chỉ số": c, "Accuracy giảm (TB)": np.mean(giam), "Độ lệch chuẩn": np.std(giam)})
    return pd.DataFrame(kq).sort_values("Accuracy giảm (TB)", ascending=False).reset_index(drop=True)


# =====================================================================
# 4. CHỈ SỐ ĐÁNH GIÁ
# =====================================================================
def ma_tran_nham_lan(y_that, y_dudoan, nhan):
    """cm[i, j] = số mẫu có lớp THẬT là nhan[i] nhưng bị dự đoán thành nhan[j]."""
    vt = {c: i for i, c in enumerate(nhan)}
    cm = np.zeros((len(nhan), len(nhan)), dtype=int)
    np.add.at(cm, ([vt[c] for c in y_that], [vt[c] for c in y_dudoan]), 1)
    return cm


def bao_cao_theo_lop(cm, nhan):
    """Precision, Recall, F1, support cho từng lớp."""
    tp = np.diag(cm).astype(float)
    du_doan = cm.sum(axis=0).astype(float)    # số mẫu bị dự đoán là lớp đó
    that = cm.sum(axis=1).astype(float)       # số mẫu thật của lớp đó
    p = np.divide(tp, du_doan, out=np.zeros_like(tp), where=du_doan > 0)
    r = np.divide(tp, that, out=np.zeros_like(tp), where=that > 0)
    f = np.divide(2 * p * r, p + r, out=np.zeros_like(tp), where=(p + r) > 0)
    return pd.DataFrame({"Lớp": nhan, "Precision": p, "Recall": r, "F1": f, "Số mẫu": that.astype(int)}).set_index("Lớp")


def tong_hop_chi_so(y_that, y_dudoan, nhan):
    """Accuracy, và Precision/Recall/F1 theo hai cách: macro (trung bình đều các lớp) và weighted (theo số mẫu)."""
    cm = ma_tran_nham_lan(y_that, y_dudoan, nhan)
    b = bao_cao_theo_lop(cm, nhan)
    w = b["Số mẫu"] / b["Số mẫu"].sum()
    return {
        "Accuracy": float(np.trace(cm) / cm.sum()),
        "Precision (macro)": float(b["Precision"].mean()),
        "Recall (macro)": float(b["Recall"].mean()),
        "F1 (macro)": float(b["F1"].mean()),
        "Precision (weighted)": float((b["Precision"] * w).sum()),
        "Recall (weighted)": float((b["Recall"] * w).sum()),
        "F1 (weighted)": float((b["F1"] * w).sum()),
    }


def cap_nham_nhieu_nhat(cm, nhan, top=10):
    """Các cặp (lớp thật -> lớp bị dự đoán nhầm) nhiều nhất."""
    ds = [(nhan[i], nhan[j], int(cm[i, j])) for i in range(len(nhan)) for j in range(len(nhan)) if i != j and cm[i, j] > 0]
    ds.sort(key=lambda t: -t[2])
    return pd.DataFrame(ds[:top], columns=["Cây thật", "Bị đoán nhầm thành", "Số mẫu"])


def ve_ma_tran_nham_lan(cm, nhan, tieu_de, duong_dan=None, chuan_hoa=True, hien=True):
    """Vẽ heatmap ma trận nhầm lẫn (chuẩn hoá theo hàng = tỉ lệ mẫu thật được dự đoán thành từng lớp)."""
    import matplotlib.pyplot as plt
    m = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1) if chuan_hoa else cm
    fig, ax = plt.subplots(figsize=(13, 11))
    im = ax.imshow(m, cmap="Blues", vmin=0, vmax=1 if chuan_hoa else None)
    ax.set_xticks(range(len(nhan))); ax.set_xticklabels(nhan, rotation=90)
    ax.set_yticks(range(len(nhan))); ax.set_yticklabels(nhan)
    for i in range(len(nhan)):
        for j in range(len(nhan)):
            if cm[i, j] > 0:
                v = m[i, j]
                ax.text(j, i, f"{v:.2f}" if chuan_hoa else int(v), ha="center", va="center", fontsize=6,
                        color="white" if v > 0.5 else "black")
    ax.set_xlabel("Dự đoán"); ax.set_ylabel("Thực tế"); ax.set_title(tieu_de)
    fig.colorbar(im, ax=ax, fraction=0.04)
    fig.tight_layout()
    if duong_dan:
        fig.savefig(duong_dan, dpi=150, bbox_inches="tight")
    if hien:
        plt.show()
    plt.close(fig)


def ve_cay(mo_hinh, toi_da=3, ten_chi_so=None, tieu_de="", duong_dan=None, hien=True):
    """Vẽ `toi_da` tầng đầu của một MyDecisionTree bằng matplotlib."""
    import matplotlib.pyplot as plt
    ten = ten_chi_so or mo_hinh.feature_names_
    vt = {}
    dem_la = [0]

    def bo_tri(k, d):
        if mo_hinh.feature_[k] < 0 or d >= toi_da:
            vt[k] = (dem_la[0], -d); dem_la[0] += 1
        else:
            bo_tri(mo_hinh.left_[k], d + 1); bo_tri(mo_hinh.right_[k], d + 1)
            vt[k] = ((vt[mo_hinh.left_[k]][0] + vt[mo_hinh.right_[k]][0]) / 2, -d)

    bo_tri(0, 0)
    fig, ax = plt.subplots(figsize=(max(8, dem_la[0] * 1.9), 2.6 * (toi_da + 1)))

    def ve(k, d):
        x, y = vt[k]
        la = mo_hinh.feature_[k] < 0
        top = mo_hinh.classes_[int(np.argmax(mo_hinh.counts_[k]))]
        if la:
            nd = f"{top}\nn={int(mo_hinh.n_samples_[k])}"
            mau = "#CDE8C8"
        elif d >= toi_da:
            nd = f"... (đa số: {top})\nn={int(mo_hinh.n_samples_[k])}"
            mau = "#EEEEEE"
        else:
            nd = (f"{ten[mo_hinh.feature_[k]]} <= {mo_hinh.threshold_[k]:.2f}\n"
                  f"{mo_hinh.criterion}={mo_hinh.impurity_[k]:.2f}\nn={int(mo_hinh.n_samples_[k])}\nđa số: {top}")
            mau = "#DCE9F8"
        if not la and d < toi_da:
            for con, nhan in ((mo_hinh.left_[k], "đúng"), (mo_hinh.right_[k], "sai")):
                ax.plot([x, vt[con][0]], [y, vt[con][1]], color="gray", zorder=1)
                ax.text((x + vt[con][0]) / 2, (y + vt[con][1]) / 2, nhan, fontsize=7, color="gray", ha="center")
                ve(con, d + 1)
        ax.text(x, y, nd, ha="center", va="center", fontsize=7, zorder=2,
                bbox=dict(boxstyle="round", fc=mau, ec="black", lw=.6))

    ve(0, 0)
    ax.axis("off"); ax.set_title(tieu_de)
    fig.tight_layout()
    if duong_dan:
        fig.savefig(duong_dan, dpi=150, bbox_inches="tight")
    if hien:
        plt.show()
    plt.close(fig)


# =====================================================================
# 5. K-FOLD PHÂN TẦNG VÀ TÌM THAM SỐ
# =====================================================================
def chia_k_fold_phan_tang(y, k=5, random_state=42):
    """Chia k phần sao cho mỗi phần giữ nguyên tỉ lệ các lớp. Trả về list[(chỉ_số_train, chỉ_số_val)]."""
    rng = np.random.default_rng(random_state)
    y = np.asarray(y)
    cac_phan = [[] for _ in range(k)]
    for c in np.unique(y):
        idx = rng.permutation(np.where(y == c)[0])
        for i, j in enumerate(idx):
            cac_phan[i % k].append(j)
    cac_phan = [np.array(sorted(p)) for p in cac_phan]
    return [(np.concatenate([cac_phan[j] for j in range(k) if j != i]), cac_phan[i]) for i in range(k)]


def cross_validate(tao_mo_hinh, X, y, k=5, random_state=42):
    """Trả về DataFrame kết quả từng fold (Accuracy, F1 macro, Accuracy train)."""
    X = X.reset_index(drop=True) if hasattr(X, "reset_index") else X
    y = np.asarray(y)
    nhan = list(np.unique(y))
    kq = []
    for i, (tr, va) in enumerate(chia_k_fold_phan_tang(y, k, random_state), 1):
        m = tao_mo_hinh().fit(X.iloc[tr], y[tr])
        ct = tong_hop_chi_so(y[va], m.predict(X.iloc[va]), nhan)
        kq.append({"Fold": i, "Accuracy": ct["Accuracy"], "F1 (macro)": ct["F1 (macro)"],
                   "Accuracy train": float(np.mean(m.predict(X.iloc[tr]) == y[tr]))})
    return pd.DataFrame(kq)


def tim_tham_so(lop_mo_hinh, luoi, X, y, k=5, random_state=42, n_iter=None, co_dinh=None, in_tien_do=True):
    """
    Tìm tham số bằng cross-validation CHỈ TRÊN TẬP TRAIN.
    luoi   : dict {tên_tham_số: [các giá trị thử]}
    n_iter : None = thử mọi tổ hợp (grid search); số nguyên = thử ngẫu nhiên n_iter tổ hợp (random search)
    co_dinh: dict tham số cố định (ví dụ n_estimators)
    Trả về DataFrame sắp xếp theo F1 macro trung bình giảm dần.
    """
    co_dinh = co_dinh or {}
    ten = list(luoi.keys())
    tat_ca = list(itertools.product(*[luoi[t] for t in ten]))
    if n_iter is not None and n_iter < len(tat_ca):
        rng = np.random.default_rng(random_state)
        tat_ca = [tat_ca[i] for i in rng.choice(len(tat_ca), n_iter, replace=False)]
    X = X.reset_index(drop=True)
    y = np.asarray(y)
    nhan = list(np.unique(y))
    folds = chia_k_fold_phan_tang(y, k, random_state)
    hang = []
    t0 = time.time()
    for s, gt in enumerate(tat_ca, 1):
        tham_so = dict(zip(ten, gt))
        a, f, atr = [], [], []
        for tr, va in folds:
            m = lop_mo_hinh(**tham_so, **co_dinh).fit(X.iloc[tr], y[tr])
            ct = tong_hop_chi_so(y[va], m.predict(X.iloc[va]), nhan)
            a.append(ct["Accuracy"]); f.append(ct["F1 (macro)"])
            atr.append(float(np.mean(m.predict(X.iloc[tr]) == y[tr])))
        hang.append({**tham_so, "Accuracy CV": np.mean(a), "Độ lệch (Acc)": np.std(a),
                     "F1 macro CV": np.mean(f), "Độ lệch (F1)": np.std(f), "Accuracy train": np.mean(atr)})
        if in_tien_do and (s % max(1, len(tat_ca) // 10) == 0 or s == len(tat_ca)):
            print(f"  đã thử {s}/{len(tat_ca)} tổ hợp ({time.time() - t0:.0f}s)")
    return pd.DataFrame(hang).sort_values("F1 macro CV", ascending=False).reset_index(drop=True)


# =====================================================================
# 6. SO SÁNH HAI MÔ HÌNH
# =====================================================================
def mcnemar_chinh_xac(y_that, du_doan_a, du_doan_b):
    """
    Kiểm định McNemar (chính xác, hai phía) cho hai mô hình trên CÙNG tập kiểm tra.
    b = số mẫu A đúng, B sai;  c = số mẫu A sai, B đúng.
    H0: hai mô hình có tỉ lệ sai như nhau. Dưới H0, b ~ Binomial(b+c, 0.5).
    """
    y = np.asarray(y_that)
    a_dung = np.asarray(du_doan_a) == y
    b_dung = np.asarray(du_doan_b) == y
    b = int(np.sum(a_dung & ~b_dung))
    c = int(np.sum(~a_dung & b_dung))
    n = b + c
    if n == 0:
        return {"A đúng - B sai": b, "A sai - B đúng": c, "p-value": 1.0}
    kmin = min(b, c)
    p = sum(math.comb(n, i) for i in range(kmin + 1)) / 2 ** n * 2
    return {"A đúng - B sai": b, "A sai - B đúng": c, "p-value": float(min(1.0, p))}


def bootstrap_chenh_lech(y_that, du_doan_a, du_doan_b, n_boot=2000, random_state=42):
    """Khoảng tin cậy 95% (bootstrap cặp) cho Accuracy(A) - Accuracy(B) trên tập kiểm tra."""
    rng = np.random.default_rng(random_state)
    y = np.asarray(y_that)
    da = (np.asarray(du_doan_a) == y).astype(float)
    db = (np.asarray(du_doan_b) == y).astype(float)
    n = len(y)
    ch = np.array([(da[i] - db[i]).mean() for i in (rng.integers(0, n, n) for _ in range(n_boot))])
    return {"Chênh lệch": float(da.mean() - db.mean()), "KTC 95% dưới": float(np.percentile(ch, 2.5)),
            "KTC 95% trên": float(np.percentile(ch, 97.5))}


# =====================================================================
# 7. DỰ ĐOÁN MẪU MỚI
# =====================================================================
def tao_ten_to_hop(to_hop):
    return " + ".join(to_hop)


def du_doan_mau_moi(cac_mo_hinh, mau, X_train, top=3):
    """
    cac_mo_hinh: dict {tuple_tên_chỉ_số: mô_hình_đã_huấn_luyện}
    mau        : dict chỉ số người dùng nhập, ví dụ {"N": 90, "P": 42, "K": 43, "pH": 6.5}
                 (có thể nhập thiếu; hệ thống chọn mô hình huấn luyện đúng với các chỉ số đã nhập)
    Trả về dict gồm top-k gợi ý và danh sách cảnh báo.
    """
    cot = tuple(c for c in THU_TU_CHI_SO if c in mau)
    if not cot:
        raise ValueError("Cần nhập ít nhất một chỉ số đất (N, P, K hoặc pH).")
    m = cac_mo_hinh[cot]
    x = pd.DataFrame([[float(mau[c]) for c in cot]], columns=list(cot))
    pr = m.predict_proba(x)[0]
    thu_tu = np.argsort(-pr)[:top]
    canh_bao = []
    for c in cot:
        lo, hi = X_train[c].min(), X_train[c].max()
        if not lo <= float(mau[c]) <= hi:
            canh_bao.append(f"{c} = {mau[c]} nằm NGOÀI khoảng dữ liệu huấn luyện [{lo:.2f}, {hi:.2f}]; kết quả không đáng tin.")
    if pr[thu_tu[0]] < 0.5:
        canh_bao.append(f"Độ tin cậy thấp ({pr[thu_tu[0]]:.0%}): nhiều loại cây có điều kiện đất tương tự.")
    if len(cot) < 4:
        canh_bao.append("Nhập càng đủ 4 chỉ số thì dự đoán càng chính xác.")
    return {"mô hình dùng": tao_ten_to_hop(cot),
            "gợi ý": [(m.classes_[i], float(pr[i])) for i in thu_tu],
            "cảnh báo": canh_bao}
