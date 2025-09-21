# TechFeed
## セットアップ
1. リポジトリに本プロジェクトを配置
2. Settings → **Pages** → Branch: `main`, Folder: `/docs`
3. Settings → **Secrets and variables → Actions** で以下を登録
   - `DEEPL_API_KEY` : Google Translate API のキー
   - `USER_AGENT`    : `YourAppName/1.0 (contact: your_email@example.com)`（任意）
4. 公開URL：`https://<your-user>.github.io/<repo>/`

## ローカル実行（任意）
```bash
export GOOGLE_TRANSLATE_API_KEY=...
export USER_AGENT="YourAppName/1.0 (contact: your_email@example.com)"
python recommend.py