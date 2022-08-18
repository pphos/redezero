# ReDeZero: Re-implement DeZero framework
ReDeZeroは書籍[「ゼロから作る Deep Learning ❸」](https://www.amazon.co.jp/dp/4873119065/ref=cm_sw_r_tw_dp_U_x_KiA1Eb39SW14Q)で作成する
深層学習フレームワーク[「DeZero」](https://github.com/oreilly-japan/deep-learning-from-scratch-3)をベースに再実装したものです.


## 学習環境
WSL2 + Docker + Python3.9

## 環境のセットアップ
1. コンテナの起動
```bash
docker compose up -d
```

2. パッケージのインストール
```
docker compose run redezero poetry install
```

## コミットプレフィックス
以下の[Angularのコミットプレフィックス](https://github.com/angular/angular.js/blob/master/DEVELOPERS.md#type)を採用

- feat: A new feature
- fix: A bug fix
- docs: Documentation only changes
- style: Changes that do not affect the meaning of the code (white-space, formatting, missing semi-colons, etc)
- refactor: A code change that neither fixes a bug nor adds a feature
- perf: A code change that improves performance
- test: Adding missing or correcting existing tests
- chore: Changes to the build process or auxiliary tools and libraries such as documentation generation