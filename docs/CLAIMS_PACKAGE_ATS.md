# 校正済み主張パッケージ — ATS (Advanced Theory and Simulations) 投稿用（2026-11-06）

対象：第1〜10報（トイモデル：2D 正方格子タイトバインディング、Landauer–NEGF／厳密 T 行列、W=20×L=40 と W=4 細線、E_F=−1.1t、欠陥 U=8t 中心に 0.5〜8t）。

---

## 1. 核の新規性（1 文）

**界面欠陥の幾何が電子コンダクタンスを変える機構は二つあり──横モードごとの欠陥重なり（モード選択）と、周期配列の超格子ストップバンド（集団伝播）──どちらが支配するかは周期数で切り替わる。前者は実空間幾何記述子では書けず、後者は自由電子 Bragg 条件では書けない。**

### 圧縮テスト
- 述語テスト：この研究は、同数の欠陥でもコンダクタンスが 2 倍変わる原因が、実空間の配置統計ではなく（i）横モード分解した欠陥重なりと（ii）周期配列の Bloch 伝播条件にあり、両者が周期数 ≈4〜6 を境に交代することを示す。
- なぜ必要か：欠陥数・密度・局所相関に基づく既存の記述では、規則配列の散乱増強を系統的に過小評価し（未知族で R² 0.80、grid でバイアス −0.14）、自由電子 Bragg 条件は多チャネル細線のストップバンド位置と無相関（|r| ≤ 0.08）だから。
- 編集者圧縮：「欠陥配置による界面伝導の変化を、モード重なりと超格子伝播の二機構に分解し、未知配置で予測可能性を示した」。
- 逆テスト：「機構」「二つ」「交代」はそれぞれ特徴重要度＋直接説明、独立した記述子群、周期数スキャン 4 角度で裏付け。「必要」「一般」「初」は使わない。

---

## 2. 主張台帳

| # | 主語 | 述語 | 証拠 | 射程 | 状態 | 対抗解釈 | 決定的ギャップ |
|---|---|---|---|---|---|---|---|
| C1 | 同数の欠陥配置 | 配置だけで G/G0 が 0.47〜0.96 と変わり、構造化配置の差はランダム配置の分散（0.65±0.025）を桁で超える | 第1報 Exp A（12 欠陥、6 構造化配置＋150 ランダム） | 2D TB、U=8t、E_F=−1.1t | **実証** | なし（直接計算） | — |
| C2 | コンダクタンス | G/G0 = (N_eff/N0)·D_eff、N_eff=S1²/S2、D_eff=S2/S1 は閾値なしの恒等式で、「チャネル数を減らす」と「全チャネルを濁らせる」を分離する（壁⊥ 0.51/0.91 vs ランダム 0.81/0.84） | 第2報 B-0′、第3報 C1 | 一般（透過固有値が定義できる任意の系） | **実証**（数学） | 透過固有値の参加比は既知の量である可能性が高い → 「導入」でなく「用いる」 | 先行文献照合 |
| C3 | 実空間幾何記述子（欠陥数・射影被覆・最近接・クラスター・g(r)） | 分布内では最大の追加価値（R² 0.62→0.87）だが、未学習の規則配列族では欠陥数のみ（0.80）に劣り（0.46〜0.66）、規則配列を系統的に過小予測する | 第4報 M0〜M3、第5報 3 E_F | GBR、580／390 構造、族外テスト | **実証**（本モデル内） | 分割・seed 依存（0.66 vs 0.79） | seed 反復済み（±0.002〜0.015）、族別分散を報告 |
| C4 | 横モード分解した欠陥重なり A_mn（特に A_fwd_diag=\|Σφ_n(y_j)²\|²/v_n） | 実空間＋構造因子に加えると未知族 R² 0.806→0.885（5 seed ±0.002）、grid バイアス −0.135→−0.039。規則配列の過小予測は、行周期が特定横モードの腹／節に乗ることで説明される | 第6報 F1、重要度 A_fwd_diag 0.27 ≈ M1_yproj 0.27 | W=20、E_F=−1.1t、U=8t | **実証**（予測改善）＋**解釈**（腹／節の説明） | S_fwd は A_fwd_diag の代理だった（第5報→第6報で吸収） | 腹／節の説明を直接検証する 1 変数実験（行を固定して y をスキャン） |
| C5 | 有限周期配列（W=4、6 欠陥）の透過暗帯 | 無限超格子のストップバンドと一致（corr 0.50@U=0.5 → 0.76@U=8；帯内／帯外 G 0.42/0.69）。自由電子 Bragg 距離とは無相関（≤0.08、U=0.5 でも） | 第7報 G2/G3 | W=4 細線、行 1 欠陥、U 0.5〜8 | **実証** | Bragg 距離指標が粗すぎる可能性 → ただし転送行列はチャネル間 Bragg を含むので上位互換 | — |
| C6 | 周期配列の多重散乱 | ランダム配置より高次の多重散乱を要求する（U=2：pair 近似誤差 grid 2.0 vs random 0.65）；U=8t では切断級数は全て無意味 | 第6報 F3（厳密 T 行列 U スキャン） | 390 構造、E_F=−1.1t | **実証** | — | — |
| C7 | 規則配列の散乱増強 | Fermi 窓のエネルギー平均（kT ≤ 0.12t）で消えない（grid バイアス −0.039→−0.063） | 第7報 G4 | 静的ポテンシャル、弾性散乱のみ | **実証**（限定） | — | フォノン・位相緩和は未含（非主張） |
| C8 | Bloch 伝播記述子の価値 | 周期数とともに増える：W=4 で corr 0.37(2)→0.83(10 周期)；W=20・N_def 固定で corr 0.15→0.30（N=24）、0.46→0.64（N=48）；ΔR² +0.02（3〜4 周期）vs +0.18（6〜8 周期）；未見周期の LOCO で 0.30→0.52、1 変数で 0.43 vs 少数周期 0.07 | 第8〜10報 | GBR／Ridge、離散周期 a∈{2,3,4(,5)}、各層 n=60〜220 | **支持**（4 独立角度、ただし層別の絶対 R² は低い） | N_def 交絡 → 第10報で統制済み；木モデルの補間失敗 → LOCO で対処 | 周期の連続化（傾斜配列）、層別 n の増加 |
| C9 | 二機構の関係 | 補完でなく**レジーム交代**：少数周期ではモード選択、多周期で集団伝播 | C4＋C5＋C8 | 本トイモデル | **解釈**（C8 の支持に基づく） | 両機構が常に加法的に寄与し、周期数は単に Bloch 項の振幅を上げているだけ | 両記述子を同時に入れた交互作用項の検定 |
| C10 | 固定資源（平均ボンド強度）下の不均一性 | N_eff↓と D_eff↑が競合し、中間強度で s に内部最大（v=0.5：+45%）；乱れを v に結合しても G(v) はほぼ不変（N↔D 交換はほぼ無損失） | 第3報 C1/C2 | 粒界 1 列、算術平均固定、クリップ t≤1 | **支持** | クリップによる平均ずれ、n=10 | クリップ無し再計算 |
| C11 | SSOC（イオン伝導 σ=N·D·F）との同型 | 分解構造と「支配項の交代」が電子系でも再現される | C2, C10 と SSOC 論文 | 構造的類比 | **解釈** | 式形が似ているだけ | 同一制御座標での横断データ |
| C12 | 実 Cu CSL 粒界 | 面内周期が面内運動量結合を決め、「結晶学的に整った粒界 ≠ 電子的に透明な粒界」となりうる | なし（本研究は 2D トイ） | — | **仮説** | — | fcc Cu TB／第一原理 NEGF、Σ3/Σ5/Σ11 |

---

## 3. 意味崩壊の所見（今の草稿語彙で消えかけているもの）
- 「記述子を段階的に追加して予測性能を測った」→ 実際に示したのは **実空間記述子の転移限界（C3）と、それを回復する物理量の同定（C4）**。「ΔL を測った」という手続き語に畳まない。
- 「grid 族の残差」→ 残差の話ではなく **規則配列の散乱増強という現象**（C5〜C7）。「回帰誤差」と呼ばない。
- 「H4 は効かなかった（第8報）」→ 否定ではなく **レジームの境界を定量化した**（周期数 ≈4〜6）。負の結果を境界として前に出す。
- 「Bragg 条件で説明できる」と書きたくなる誘惑 → 証拠は逆（無相関）。**「自由電子 Bragg では書けず、転送行列の Bloch 条件で書ける」**が正しい述語。
- 「トイモデルなので」で全体を弱めない。射程は「2D 単一バンド TB・弾性散乱・E_F 一点（＋Fermi 窓）」と一度だけ明示する。

---

## 4. 校正済み表現

### タイトル候補
1. **Mode Selection and Collective Propagation: Two Regimes by Which Defect Geometry Controls Interfacial Electron Transport**
2. Why Equal Defect Counts Give Unequal Conductance: Mode-Resolved Overlap and Superlattice Propagation in a Tight-Binding Interface Model
3. Beyond Real-Space Descriptors: Mode Overlap and Bloch Propagation Predict Conductance of Unseen Defect Arrangements

推奨は 1（述語が主語）。2 は問いが立つが長い。3 は ML 寄りに読まれるリスク。

### アブストラクト（英、約 220 語、投稿可能水準）
> Interfaces with the same defect density can differ severalfold in electrical conductance, yet descriptors based on defect counts and real-space correlations do not explain why. Using a two-dimensional tight-binding model of a defected interface solved with Landauer–NEGF and an exact on-site T-matrix, we show that defect geometry controls conductance through two distinct mechanisms. First, the conductance decomposes exactly as G/G₀ = (N_eff/N₀)·D_eff into an effective number of transmitting channels and a per-channel transparency; configurations with identical defect counts differ by up to a factor of two because they remove channels or dim them. Second, the descriptor that transfers to unseen arrangements is not real-space geometry but the transverse-mode-resolved defect overlap A_mn: adding it raises held-out R² from 0.81 to 0.885 and removes the systematic under-prediction of periodic arrays. Third, periodic arrays enhance scattering through superlattice stop bands: the dark bands of finite arrays coincide with the Bloch gaps of the infinite superlattice (r = 0.76 at U = 8t) while being uncorrelated with the free-electron Bragg condition, they require higher-order multiple scattering than random arrangements, and they survive Fermi-window averaging. Finally, the two mechanisms exchange roles with the number of periods: at fixed defect count, the Bloch descriptor's added value grows from +0.02 (3–4 periods) to +0.18 (6–8 periods). We discuss the structural parallel to the N·D·F decomposition of ionic conductivity and the route to Cu coincidence-site-lattice boundaries.

### 貢献リスト（導入末尾）
1. 同数欠陥の配置依存（2 倍差）を、厳密分解 G/G₀=(N_eff/N₀)·D_eff で「チャネル数の減少」と「透明度の低下」に分離した（C1, C2）。
2. 実空間記述子の族外転移限界を定量し（C3）、転移を回復する物理量として横モード分解欠陥重なり A_mn を同定、規則配列の過小予測の原因を横モードの腹／節に帰した（C4）。
3. 規則配列の散乱増強が、自由電子 Bragg 条件ではなく無限超格子のストップバンドで決まり、高次多重散乱を要し、Fermi 窓平均で消えないことを示した（C5〜C7）。
4. 二機構が周期数で交代することを、周期数スキャン・未知族・未見周期予測・欠陥数統制の 4 角度から支持した（C8, C9）。
5. 付随：固定資源下の不均一性による N_eff↔D_eff 交換（C10）。

### 図構成（本文 6 図 ＋ SI）
- Fig.1 問いと分解：配置 6 種＋ランダム分布の T/T₀；透過固有値スペクトル；(N_eff/N₀, D_eff) 平面に各配置（figA＋figC2）
- Fig.2 記述子階層と族外転移：M0→M1→M2→S(q)→A_mn の R²（同族／未知族）、grid の parity（figJ, figL, figO）
- Fig.3 モード選択の機構：A_fwd_diag の定義図、重要度、行周期と横モード腹／節の対応（新規作図 1 枚）
- Fig.4 集団伝播：W=4 の周期×E_F マップ ＋ 無限超格子 Bloch 帯（U=0.5, 2, 8 の 3 段）、Bragg 距離との無相関（figR 抜粋）
- Fig.5 多重散乱階層と熱平均：U スキャン pair 誤差 grid vs random；G(E) 族平均と kT 依存（results_F3_Uscan, figS）
- Fig.6 レジーム交代：周期数 vs corr（W=4）、N_def 固定の ΔR²、LOCO（figU 改訂＋J）
- SI：Exp B 応答窓（μ/φ/δn）、Exp C 不均一性、Exp H の負の結果、厳密 T 行列の検証、全データセット

---

## 5. 非主張（貢献の後に 1 段落で）
本研究は 2D 単一バンド TB・オンサイト欠陥・弾性散乱・E_F 一点（＋Fermi 窓平均）の範囲で成立する。実 Cu の抵抗率の定量予測、fcc／s–d 混成、フォノン散乱・位相緩和・粒界構造の温度変化、製造履歴の生成は扱わない。Bloch 記述子の予測改善は離散周期（3〜4 水準）設計の制約下での Δ の符号と単調性に限る。N_eff·D_eff 分解は透過固有値の参加比として既知の量であり、ここでは分離の道具として用いる。

---

## 6. 不足している資格（より強い語を使うための最小追加）
- 「**機構**」（C4 の腹／節）：行を 1 本固定して y を掃引し、G と φ_n(y)² の対応を直接見せる 1 図（計算 1 分）。
- 「**レジーム交代**」を「補完」から区別：A_mn と Bloch 記述子の交互作用項、または周期数で層別した部分依存プロット。
- 「**ストップバンド**」の一般性：W=20 の多周期族で転送行列 Bloch 判定（実パターンセル）と G の直接相関（第9報 0.77）を本文に入れる──既にある。
- C2 の先行文献：透過固有値分布・参加比（Dorokhov, Beenakker 系）の引用を付け、「導入」と書かない。
- 連続周期（傾斜配列）で C8 の絶対 R² を上げる（任意、投稿後でも可）。

---

## 7. 落とした候補（記録）
| 候補 | 理由 |
|---|---|
| 「電流を最初の説明変数にしない観測窓」（μ, φ, δn：第2報） | 支持的だが核の述語に直結しない。μ 降下 ≈ 1−G は自明に近い。SI へ。壁⊥の自己無撞着非収束は未解決 → 本文から除外 |
| 「N_eff×D_eff ≈ const（固定資源下）」を法則として | n=10、クリップあり。C10 として「支持」に留め、法則名を付けない |
| 「S(q) 記述子」を貢献として | A_mn に吸収され重要度 0.03 以下。歴史的経路として本文で 1 文、貢献には入れない |
| 「T 行列スペクトル（σ_min）が grid 残差を説明」 | 証拠が逆（相関 ≤0.26、追加で悪化）。負の結果として SI |
| 「Bragg 条件で周期効果を予測」 | 証拠が逆（無相関）。述語を Bloch 条件に置換 |
| 「有限温度でも生き残る」 | Fermi 窓平均のみ。C7 の限定表現に縮小 |
| 「Fano 因子が均一型／選択型の判別子」 | 計算していない。仮説として展望 |
| 「SSOC との普遍分解」 | 式形の同型と支配項交代の類比まで。展望節 |
| 「実 Cu CSL：整った粒界 ≠ 透明な粒界」 | 2D トイに実 Cu の証拠なし。展望の仮説として 1 段落 |
| 「局所周期抽出記述子 H3/H4（平均場セル）」 | 一般配置では効かず（+0.004）。境界条件の定量化（周期数）に役割を変えて C8 に統合 |
| 「製造履歴 → 構造 → 伝導」 | 本研究では生成していない。非主張に明記 |

---

## 8. 公開可能テキストと要証拠テキストの区別
- **そのまま使える**：核の 1 文、台帳 C1〜C3, C5〜C7、アブストラクトの第 1〜3 文と最終文、非主張段落。
- **1 図追加で強くなる**：C4 の「腹／節」機構文（現状は解釈）。
- **表現を Δ に限定して使う**：C8 の数値群（絶対 R² を見出しに出さない）。
- **展望でのみ**：C11, C12。
