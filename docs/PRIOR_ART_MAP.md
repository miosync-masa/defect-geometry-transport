# 先行研究マップ（2026-11-06 時点、Web 調査）

目的：(1) 新規性の線引き、(2) 主張台帳の射程修正、(3) ATS の査読者候補（理論屋がいるか）。

## A. 粒界電子輸送の理論（主流＝特定 CSL 粒界の第一原理 NEGF）
| 文献 | 何をしたか | 僕らとの関係 |
|---|---|---|
| Zhou, Jog, Gall, *APL* 120, 241603 (2022) 「First-principles prediction of electron grain boundary scattering in fcc metals」 | 8 種 fcc 金属の Σ3/5/9/11 で反射確率 r を DFT 計算。**r は粒界面に平行な面間隔が小さいほど増える**。球に近い Fermi 面ほど透過しやすい（Cu が最良） | 実空間幾何記述子（面間隔）が CSL レベルでは効く、という先行。→ C3 の射程修正（下記） |
| César, Liu, Guo et al. (McGill, Nanodcal) 「Calculated resistances of single grain boundaries in copper」(2014); Ke, Xia, Guo *PRB* (2009) | Cu 単一粒界の DFT-NEGF。抵抗と粒界幾何の強い相関を報告。[001] 36.8° 傾角で 2SR=0.604×10⁻¹⁵ Ωm² | 絶対値の比較基準。手法系譜（NEGF）の親 |
| Valencia, Wilson, Jiang, Valencia-Zapata, Klimeck, Povolotskyi, *Phys. Rev. Applied* 9, 044005 (2018) 「Grain-boundary resistance in Cu interconnects: from an atomistic model to a neural network」 | EAM＋TB＋NEGF、10,000 原子級、**600 のランダム粒構造の統計集合**、NN によるコンパクトモデル。「粒界散乱は局所的でなく結晶方位に強く依存」 | **最も近い先行**（統計集合＋ML）。違い：彼らは方位角→抵抗の黒箱 NN、僕らは物理記述子の段階追加と族外転移、機構の二層分解。必ず引用し、差分を明示 |
| Feldman et al. (ORNL) 「Structural dependence of grain boundary resistance in Cu nanowires」(2010); Kim et al. *Nano Lett.* (2010) 「Large discrete resistance jump at GB in Cu nanowire」 | 4 探針 STM で個別粒界の抵抗を構造と 1 対 1 で測定。ランダム高角粒界 ≈25×10⁻¹² Ωcm²、CSL は大幅に低い | 実験側の基準。「同じ材料で粒界ごとに桁違い」の問いの出所 |
| Gall, *J. Appl. Phys.* 127, 050901 (2020) 「The search for the most conductive metal for narrow interconnect lines」（レビュー） | 粒界通過＝Fermi 面の回転、**同じ E, p の空き状態があれば透過**（Fermi 面整合の議論）。低抵抗粒界の条件：境界ポテンシャル変動が小、電子状態整合 | 「モード整合」の直感の標準的な言い方。A_mn はこれの横モード版 |
| Hegde, Bowen, Rodder (2016 頃) 「Lower limits to metal and GB resistivity」; Lanzillo (IBM) Pt/Rh/Ir/Pd | TB/DFT による粒界・細線の下限 | 周辺。必要なら引用 |

## B. 界面抵抗のモード整合理論（スピントロニクス系、1997〜）
| 文献 | 内容 | 関係 |
|---|---|---|
| Schep, van Hoof, Kelly, Bauer, Inglesfield, *JMMM* 177, 1166 (1998); Schep & Bauer | 透過行列 T_μν から界面抵抗 R = (h/e²)/ΣT − ½(1/G_A+1/G_B)。「汚れた界面」の透過固有値分布 | 透過固有値で界面を語る枠組みの原点。N_eff·D_eff の文脈に必須 |
| Stiles & Penn, *PRB* 61, 3200 (2000) | 上式を Boltzmann 解と比較して検証。「界面抵抗は欠陥散乱でなくバンド不整合が支配しうる」 | 同上 |
| Xia, Zwierzycki, Talanana, Kelly, Bauer, *PRB* 73, 064420 (2006)；cond-mat/0601367（金属対の界面透過の方位依存） | 第一原理散乱行列。**周期ポテンシャルが自由電子 Fermi 面に開くギャップ**が透過を変える | 「自由電子 Bragg では書けず、周期ポテンシャルのバンドで書く」の先行的直感 |
| Nikolić, cond-mat/0112020 | 乱れ界面の透過固有値分布（Schep–Bauer 分布）、多層の量子輸送 | T_n 分布の双峰性（第3報 C3）と比較可能 |
| Dorokhov / Beenakker（透過固有値分布の一般論） | 参加比・Fano 因子 | C2 の「導入でなく用いる」の根拠 |

## C. 周期的不純物列の導波路（**引用必須・新規性の線引き**）
| 文献 | 内容 | 関係 |
|---|---|---|
| Cserti et al., cond-mat/9710264 「Crossover from regular to chaotic behavior in the conductance of periodic quantum chains」; 同 9710265（レビュー） | 導波路に**有限個の周期配置された点状不純物**。不純物数 N が増えると**コンダクタンスが無限周期鎖のバンド数×2e²/h に指数的に収束**。強度を上げると可積分→量子カオス | **第7〜8報の G2・周期数依存はこれの再発見に近い。** 僕らの差分：(i) 2D 多チャネル細線で N_eff·D_eff に分解、(ii) これを*記述子*として未知配置の予測に組み込みレジーム境界（周期数 ≳4〜6）を欠陥数統制で示した、(iii) 自由電子 Bragg との無相関を明示。「ストップバンドの発見」とは絶対に書かない |
| Kaneko? / 周期不純物＋基板結合の TB（ResearchGate 2013） | 周期不純物で DOS にギャップ | 周辺 |

## D. 粒界の構造記述子・ML
| 文献 | 内容 | 関係 |
|---|---|---|
| Bishara, Lee, Brink, Liebscher, Dehm, *ACS Nano* 15, 16607 (2021) 「Understanding grain boundary electrical resistivity in Cu: the effect of boundary structure」 | 個別 Cu 傾角粒界の抵抗を STEM 構造と対応。**抵抗率は余剰体積にほぼ比例**（第一近似）。曲率で +80%。**同一 CSL 内の相変化・欠陥では大きく変わらない**。低角 14–18° が最も高抵抗 | 第1報の「v_f ⇔ 粒界余剰体積」の実験的裏付け。ただし「同一 Σ 内の相変化で変わらない」は A2 仮説（同一 Σ・異なる構造単位で変わる）への**反証候補**。展望では正直に併記 |
| Dehm グループ ERC GB-CORRELATE 報告（2023–24） | 「余剰体積が個別粒界の抵抗差の主因」、Fe 微量偏析で +1000% | 偏析＝「ピーク型を生む PCC 項」候補の実験根拠 |
| Homer et al., *npj Comput. Mater.* 11, 21 (2025) 「Feature engineering descriptors, transforms, and ML for grain boundaries」 | 7000 粒界、SOAP 等の記述子→粒界エネルギー。余剰体積との相関 | 記述子工学の文脈。彼らは**エネルギー**、僕らは**輸送**。実空間記述子の限界という主張の対照 |
| Rosenbrock et al. (SOAP 普遍記述子) | 粒界の ML 表現 | 同上 |
| Valencia 2018（A 欄） | 抵抗の NN | 同上 |

## E. 新規性の線引き（台帳への反映）
1. **C3（実空間記述子の転移限界）の射程修正**：CSL レベルでは面間隔（Gall）・余剰体積（Bishara）という実空間記述子が効くことが先行で示されている。僕らの主張は「**界面内の欠陥*配置*（同一欠陥数）に対する**局所相関記述子は、未学習の周期配列族へ転移しない」に限定する。CSL 分類レベルの記述子を否定しない。
2. **C5（有限＝無限ストップバンド）**：Cserti らの「有限周期鎖→無限鎖バンドへの収束」を先行として明記。僕らの新規は、(a) 多チャネルでの Bloch 割合と N_eff·D_eff の対応、(b) 自由電子 Bragg との無相関の明示、(c) 記述子としての周期数レジーム境界（欠陥数統制つき）。
3. **C2（分解）**：Schep–Bauer／Dorokhov–Beenakker の透過固有値分布の文脈に置き、「用いる」。
4. **C4（モード重なり A_mn）**：Fermi 面整合（Gall 2020、Stiles–Penn）の横モード版であることを述べたうえで、**配置依存の記述子として族外転移を回復する**という使い方が新規。
5. **Valencia 2018 との差分を導入で 1 段落**：統計集合＋ML は彼らが先。僕らは黒箱でなく物理量を段階追加し、どの情報が転移に効くかを問う（ご主人さまの M0→M3 設計の意義がここで立つ）。
6. **A2 仮説（同一 Σ・異なる構造単位）**：Bishara 2021 の「同一 CSL 内では大差なし」と緊張関係。展望では「余剰体積が第一近似、本研究の機構はその先の二次構造を与える候補」と書く。

## F. 理論屋はいるか？（ATS 査読者候補）
- **Daniel Gall**（RPI）— 実験＋DFT 両刀、粒界反射の第一人者。A_mn の「Fermi 面整合」読みを理解する
- **Hong Guo**（McGill）— DFT-NEGF 粒界抵抗の原典
- **Gerhard Klimeck / Michael Povolotskyi**（Purdue）— TB-NEGF 統計集合＋ML。最も近い
- **Paul J. Kelly**（Twente）、**Gerrit E. W. Bauer**（Tohoku）— 界面透過行列・モード整合理論
- **Mark D. Stiles**（NIST）— 同上
- **Branislav K. Nikolić**（Delaware）— 乱れ界面の透過固有値分布
- **József Cserti**（ELTE）— 周期不純物鎖の導波路
- **Eric R. Homer**（BYU）— 粒界記述子・ML
- 実験側で理論を読む人：**Gerhard Dehm / Christian Liebscher / Tobias Brink**（MPIE）
→ 理論屋は十分いる。ただし「粒界 × 記述子 × レジーム」を全部読める人は少ないので、カバーレターで Valencia・Cserti・Bishara の 3 本との位置関係を先に書くのが安全。

## G. 未調査（次回）
Hegde–Bowen–Rodder の正確な書誌、Lanzillo の Cu 粒界論文、Cserti 論文の著者・掲載誌の確定、Valencia 2018 の NN 入力変数の確認、Schep–Bauer 分布と第3報の T_n 双峰の定量比較。
