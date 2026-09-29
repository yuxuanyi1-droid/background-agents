# 交接文档 — feat/cli-agent-harnesses-custom-model-providers 分支工作与本地环境

> 生成于 2026-09-29 深夜。供新会话续接使用：读完本文即可完整接上上下文。本会话的 ZCode session
> id：`sess_bb1032c9-7ce3-4e33-bdba-c8bea63b1ec2`（可用 ReadSessionContext 拉取原始对话）。

## 一、分支状态（仓库 /home/yuxuanyi/background-agents）

分支 `feat/cli-agent-harnesses-custom-model-providers`，基于 `main`（a5cfc253），合并了：

- `origin/feat/cli-agent-harnesses`（5f1f8ca0：codex/pi/dsh/zcode 四个 CLI harness）
- `origin/feat/custom-model-providers`（d4d8b8bd：自定义模型网关 + OpenRouter 目录预填）

**本会话产出的提交（时间序）**：

| 提交                           | 内容                                                                                                                                                                                                                                                                                    |
| ------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 9fa3bb30 / 7efe805e / c3efed87 | 两次合并 + 迁移重编号（0085/0086，解决两个分支的 0084 撞号）                                                                                                                                                                                                                            |
| 90c27cd3                       | fix(web)：CustomProvidersSettings 在渲染期调用 load → 改 useEffect                                                                                                                                                                                                                      |
| ab51a07a                       | fix(sandbox-runtime)：`custom_anthropic_env` 剥离尾部 `/v1`（Claude Code 会自己拼 `/v1/messages`，智谱双重路径 404 → 被误报为 model not found）                                                                                                                                         |
| bdd1e490                       | fix(sandbox-images)：tools.sh 给 pnpm 传了 npm 专属 `--no-fund`，模板构建必失败                                                                                                                                                                                                         |
| 464c8d5b                       | **feat：Codex 接入自定义 provider（OpenAI Responses API）**——shared 协议枚举 + codex 模型族加 custom-openai + D1 迁移 0087（重建 custom_providers 放宽 protocol CHECK）+ 运行时写 `~/.codex/config.toml`（wire_api=responses/chat，幂等合并）+ `-c model_provider=` 路由 + web 协议下拉 |
| 9828582e                       | fix(web)：时间线虚拟化器 `useFlushSync: false`（measureElement ref 提交期 flushSync 警告）                                                                                                                                                                                              |

全部验证过：shared 1058 / control-plane 5317 / web
1982 测试通过，typecheck 干净，ruff 干净。sandbox-runtime 有 5 个**存量**测试失败（browser_desktop×3、git_signer、modal_image_build，改动前即失败，环境相关，与本分支无关）。

**分支尚未推送到远程。**

## 二、本地运行环境（WSL2）

| 服务                                          | 地址                                                | 状态                                                                     |
| --------------------------------------------- | --------------------------------------------------- | ------------------------------------------------------------------------ |
| control-plane（Node 模式，dist/node/main.js） | http://localhost:8788                               | 已跑，86 个迁移                                                          |
| web（next dev）                               | http://localhost:3000                               | 已跑                                                                     |
| SeaweedFS 对象存储（docker compose）          | 127.0.0.1:9000                                      | 已跑                                                                     |
| cloudflared 快速隧道（WORKER_URL 用）         | 见 `packages/control-plane/.env.node` 的 WORKER_URL | **不稳定，会挂**，挂了重启换地址并同步改 WORKER_URL + 重启 control-plane |

启动/重启方式（都在仓库内）：

- 对象存储：根目录 `docker compose up -d object-store`（读根目录 `.env`）
- control-plane：`cd packages/control-plane && node --env-file=.env.node dist/node/main.js`
- web：`cd packages/web && npx next dev -p 3000`
- 隧道：`cloudflared tunnel --url http://localhost:8788`，输出 URL 填进 .env.node 的 WORKER_URL

**密钥/配置文件（均已 gitignore，不随 git 同步）**：

- `packages/control-plane/.env.node`：全部运行配置（加密密钥、GitHub App/OAuth、E2B
  key、WORKER_URL、E2B_TEMPLATE_ID）
- `packages/web/.env.local`：CONTROL_PLANE_URL / NEXT_PUBLIC_WS_URL（指向 8788）/
  SERVICE_AUTH_SECRET（须等于 .env.node 的 SERVICE_AUTH_SECRET_WEB）
- 根目录 `.env`：compose 栈的对象存储口令

**本地数据**：`/tmp/open-inspect-local/control-plane-data/`（global.db +
sessions/\*.db，重启机器会清）。用户 `622b4eb0...`
已提权 owner（bootstrap 脚本只支持远程 D1，本地是手工 SQL + 审计事件）。已注册自定义 provider
`glm-code`（智谱 Anthropic 兼容端点，base URL 已修正为
`https://open.bigmodel.cn/api/anthropic`，去掉尾部 /v1）。

## 三、E2B 资产

- 账号内当前模板（API key 在 .env.node）：**在用
  `open-inspect-sandbox-6d354c1e2400-1790691850368658861`**（含全部合并代码 + responses 特性）
- 重建模板：`cd packages/e2b-infra && E2B_API_KEY=… E2B_TEMPLATE_ID=open-inspect-sandbox uv run python build-template.py`（本地 uv/ruff 已装在 ~/.local/bin；约 4 分钟，产出唯一候选名，需手动更新 .env.node 的 E2B_TEMPLATE_ID 并重启 control-plane）
- 沙箱回连原理：env 里 CONTROL_PLANE_URL=WORKER_URL，所以隧道挂 = 新沙箱连不上 = 会话卡"connecting"。四种 CLI
  harness 走同一 CliStager 路径，一个能起其他就能起。

## 四、Agent 现状

- **claude**：✅ 完整可用（glm-5.3 对话已验证）
- **codex**：✅ 沙箱链路通；静态 openai 模型可用；自定义网关需注册 Responses/OpenAI-compatible 协议 provider（设置页有下拉）
- **zcode**：✅ 链路通，用 `zai-coding-plan/glm-5.3`（已在用户启用列表）
- **pi**：沙箱通，但**尚不支持自定义 provider**（CLI 收到 `cpa-…` 前缀模型报 unknown
  model）——下一个待做的 feature（参考 codex 的接法：manifest → vendor prepare）
- **dsh**：需先在 Settings → Models 启用 DeepSeek 模型（默认关闭）

## 五、用户已提供的凭据（原 tar 包 /tmp/cf-config/open-inspect-cloudflare/）

GitHub App（xuanyi-background-agents，id 5086907，installation
166014231，已验证可发 token、见 9 仓库）；GitHub OAuth（已加 localhost 回调）；E2B key；Cloudflare
token（**权限不全**：缺 KV/D1/Queues 写，Workers 部署会被卡）；tfvars 里
`e2b_template_id=open-inspect-sandbox` 是错的（实际模板带哈希后缀）。

## 六、下一步候选（按用户意向）

1. **推送分支**（git push -u origin 分支名）
2. **服务器部署**（用户已表意向）：compose 栈 + Caddy
   TLS + 域名；数据库只能本地 SQLite+Litestream；对象存储本地或 R2；2C2G 起步；部署后隧道问题消失。web 不在 compose 里需另跑或加进去
3. **隧道根治**：named tunnel（需自有域名托管 CF）或 ngrok 静态域名
4. **pi 接自定义 provider**（仿 codex：cli_vendors prepare 写 pi 的 provider 配置）

## 七、新会话续接方法

- 同机新 ZCode 会话：直接说"读 HANDOFF.md 继续"，或用 `#sess_bb1032c9-7ce3-4e33-bdba-c8bea63b1ec2`
  引用本会话
- 服务器/新机器：先 `git push` 本分支 + `scp`
  本文档和（可选）三个 env 文件；DB 数据不同步（新环境重新注册 provider 即可）
