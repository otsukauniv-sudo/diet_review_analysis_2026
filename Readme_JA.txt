ダイエット口コミ研究：再現パッケージ

日本語版。英語版はReadme_EN.txtを開いてください。

自己申告の減量率と星評価の不一致を分析します。
論文・予稿の集計、本文表現の比較、商品固定効果回帰（N1・N1b・N2）と掲載図を作成します。

1. データを入れる

配布ZIPを展開してください。
利用許諾を得た「ダイエット口コミデータセット」の次の2ファイルを、
同梱のMaindataフォルダーに入れてください。

  Maindata/review.csv   ：口コミデータ
  Maindata/product.csv  ：商品データ

データのZIPも展開し、2つのCSVをMaindataの直下に置きます。
CSVのファイル名と列名は元のまま使ってください。
実データは配布パッケージに同梱していません。

2. 分析を実行する（Windows・VS Code）

PythonがインストールされているPCで使用します。
VS Codeで、Analysis.pyとrequirements.txtが見えるフォルダーを開いてください。
メニューの「ターミナル」から「新しいターミナル」を開き、PowerShellで実行します。

初回のみ、必要なライブラリを入れます。

  python -m pip install -r requirements.txt

続けて分析を実行します。

  python -u .\Analysis.py

Maindataから2つのCSVを読み込み、集計・回帰分析・作図まで行います。
元のCSVは変更しません。口コミを外部へ送信する処理はありません。

3. 出力フォルダーと各ファイル

以下は outputs_private/run_YYYYMMDD_HHMMSS_microseconds/ の中の場所です。
実行するたびに新しいフォルダーを作ります。
最初は次の3ファイルを確認してください。
  summary_tables/group_counts_and_weight_change.csv
  summary_tables/mismatch_counts_and_rates.csv
  regression_results/low_rating_associations_5_percent_loss.csv
CSVはExcel等、PNGは画像ビューア、JSONはメモ帳等で開けます。

実行フォルダー直下：実行全体

  run_status.json
    実行状態（COMPLETED／PARTIAL／FAILED）、分析設定、Python・ライブラリの版、各モデルの成否。

data_checks/：データの読込・変換・結合の確認

  input_data_summary.json
    元データの件数、重複、商品IDの結合状況、欠損・警告のまとめ。
  input_columns.json
    分析に使った元CSVの列名の対応表。
  sample_selection_counts.csv
    元データから対象カテゴリ・本文・評価付き標本までの件数。
  numeric_conversion_counts.csv
    体重・身長・年齢を数値に変換できた件数と、欠損・変換できなかった件数。
  PRIVATE_unparsed_numeric_formats.csv
    数値に変換できなかった入力表記の集計。数字を<num>に置き換えた形式を確認。
  PRIVATE_rating_conversion.csv
    元の評価表記、その件数、星1〜5への変換結果、欠損・無効の判定。
  product_category_mapping.csv
    元の親・子カテゴリを、分析用カテゴリへどう対応させたかと件数。
  otsuka_sample_count_comparison.csv
    大塚論文の4カテゴリの掲載件数と、今回の件数との差。

summary_tables/：減量成果と星評価の不一致の集計

  group_counts_and_weight_change.csv
    A〜D群と2つの中立群の件数・構成比、減量率・減量量・星評価の要約。
  mismatch_counts_and_rates.csv
    B群・C群の割合と、それぞれの分母。
  weight_rating_sample_selection.csv
    対象口コミから、星評価と前後体重が利用可能な標本までの除外件数。
  weight_loss_threshold_comparison.csv
    減量率5%・3%、減量量1kg・3kgの各固定基準による件数と割合。
  group_usage_duration.csv
    各群の使用期間別の件数・割合。
  group_product_categories.csv
    各群の商品カテゴリ別の件数・割合。
  group_text_expression_comparison.csv
    B対A・C対Dの本文表現の件数、出現率、その差。
  category_mismatch_rates.csv
    カテゴリ別のB群・C群の件数・割合と分母。
  weight_change_breakdown.csv
    C群・D群を、少量減量・体重変化なし・増量に分けた内訳。
  font_check.json
    Analysis.py側の日本語フォント検出記録。Figures.pyは作図時に別途フォントを確認。

regression_results/：商品固定効果回帰の結果

  low_rating_associations_5_percent_loss.csv
    主回帰。減量率5%以上の口コミで、低評価（星1・2）と本文指標の関連を推定。星3〜5は0。
  low_vs_high_ratings_5_percent_loss.csv
    補助回帰。減量率5%以上のB群とA群を比較。星3を除く。
  rating_weight_loss_cancellation_interaction.csv
    体重・評価・本文が有効な標本で、星評価と「減量5%以上×解約言及」の交互作用を推定。
  low_rating_associations_5_percent_loss_covariance.csv
    N1の係数の分散共分散行列。係数を組み合わせた標準誤差の計算等に使用。
  low_vs_high_ratings_5_percent_loss_covariance.csv
    N1bの係数の分散共分散行列。
  rating_weight_loss_cancellation_interaction_covariance.csv
    N2の係数の分散共分散行列。成功別の解約言及との関連の計算に使用。
  cancellation_associations_by_weight_loss.csv
    N2に基づく、減量5%以上・5%未満それぞれの解約言及との関連と、両者の差。
  regression_status.csv
    各回帰の成否、標本数・商品数、単一口コミ商品の除外数などの一覧。
  regression_details.json
    各回帰の詳細。除外した変数、調整変数の基準カテゴリ、警告、推定できない理由。

figures/en/：紀要用・英語表記の6図

  figure1_analysis_flow.png / figure1_analysis_flow.svg
    データの絞り込み、本文処理、群比較、回帰までの流れ。
  figure2_text_classification_rules.png / figure2_text_classification_rules.svg
    人工例文を使った本文の判定ルールと7つの回帰用指標。
  figure3_weight_loss_rating_groups.png / figure3_weight_loss_rating_groups.svg
    減量5%以上・5%未満と、低評価・中立・高評価のクロス集計。
  figure4_weight_change_comparison.png / figure4_weight_change_comparison.svg
    A群・B群の平均減量率と、C群・D群の体重変化の内訳。
  figure5_text_expression_comparison.png / figure5_text_expression_comparison.svg
    B対A・C対Dの本文表現の出現率の比較。
  figure6_regression_coefficients.png / figure6_regression_coefficients.svg
    主回帰N1の本文指標の係数と95%信頼区間。

figures/ja/：予稿用・日本語表記の4図

  figure1_analysis_flow.png / figure1_analysis_flow.svg
    分析対象の絞り込みと、体重・評価・本文を組み合わせる処理の流れ。
  figure2_text_classification_rules.png / figure2_text_classification_rules.svg
    人工例文による解約・症状の言及と具体的表現の判定例。
  figure3_weight_change_comparison.png / figure3_weight_change_comparison.svg
    A群・B群の平均減量率と、C群・D群の体重変化の内訳。
  figure4_text_expression_comparison.png / figure4_text_expression_comparison.svg
    B対A・C対Dの本文表現の出現率の比較。

回帰の係数表には、係数・標準誤差・95%信頼区間・p値・標本数・商品数を保存します。
N1・N1bの係数は確率差です。100倍するとパーセントポイントになります。
N2と交互作用の関連表は星の点数差です。
商品固定効果、投稿年、使用期間、文章長を調整し、標準誤差は商品単位でクラスタリングします。
推定できないモデルの係数表は出力されません。regression_status.csvまたはregression_details.jsonを確認してください。

図はPNG（300dpi）とSVG（拡大しても劣化しにくい形式）で保存します。
Figures.pyが集計CSVから自動で作図します。
紀要用と予稿用は、用途に合わせて図の内容・数・配置が異なります。
日本語フォントはメイリオ等を自動検出します。
PRIVATE_付きの確認表には元の入力表記が含まれるため、手元で確認してください。
本文分類の人手検証結果は、この版では出力しません。

4. 各ファイルの役割

  Analysis.py               ：分析の本体。通常はこのファイルを実行します。
  Figures.py                ：掲載図の作成。Analysis.pyから自動実行されます。
  requirements.txt          ：必要なPythonライブラリの一覧。
  Readme_EN.txt             ：英語の説明書。Windowsのメモ帳で開けます。
  Readme_JA.txt             ：日本語の説明書。Windowsのメモ帳で開けます。
  README.md                 ：GitHubで読む英語の利用案内。
  Maindata/README.md         ：CSVの置き場所の案内。
  LICENSE                   ：コードのMITライセンス。
  .gitignore                ：実データや分析結果をGitの管理対象から外す設定。

5. 必要な場合だけ

CSVがShift-JISの場合：
  python -u .\Analysis.py --encoding cp932

CSVを別の場所から読み込む場合（引用符の中を実際の場所に変更）：
  python -u .\Analysis.py --data-dir "D:\data\dietreview_data_20231026"

データなしで動作を試す場合（人工データを自動生成）：
  python -u .\Analysis.py --demo

保存済み結果から図だけ作り直す場合（run_...を実際の実行フォルダー名に置き換える）：
  python .\Figures.py ".\outputs_private\run_..."

「CSVが見つかりません」と表示されたら、Maindataの直下に
review.csvとproduct.csvの両方があるか確認してください。
標準の日本語列名と異なる場合は、Analysis.py冒頭のDEFAULT_CONFIGで対応を指定します。
体重・期間は「使用前体重」「使用後体重」「使用期間」の列を使用します。

再現の範囲とライセンス

論文の数値を再現するには、同じ版の実データが必要です。
人工データの実行では、そのデータに基づく結果が出力されます。
減量率5%以上を主要な暫定基準とし、星1・2を低評価、星4・5を高評価とします。
A群＝5%以上・高評価、B群＝5%以上・低評価、C群＝5%未満・高評価、D群＝5%未満・低評価です。
星3は中立として集計に残します。5%未満には、少量の減量・体重変化なし・増量を含みます。
本文指標の実口コミに対する精度は未確認です。回帰係数は条件付きの関連を表します。
人工データで動作確認しています。提供されたWindows・Python 3.13.2での
実データ実行結果でも、主要集計・回帰係数の一致を確認しています。

コードはMITライセンスです。データには提供元の利用条件が適用されます。
データセットの案内：https://www.nii.ac.jp/dsc/idr/diet-review/
