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

## 六、服务器部署（2026-09-29 已完成基础栈，等待用户凭据）

部署机：全新 Ubuntu VM（4C/3.8G/79G，公网 39.109.109.53 + IPv6，无域名）。仓库在
`/root/background-agents`，分支已与 origin 同步（推送完成，含本提交）。

**已就绪**：

- Docker 29 + Compose v5.5（官方源安装）；宿主机 Node v24.11.0（/usr/local，官方 tarball）
- 根目录 `.env`：全新生成的一套部署密钥（三把加密 key、BROWSER_AUTH_SECRET、四个
  SERVICE_AUTH_SECRET、对象存储口令），GitHub App 的 id/installation/bot 用户名已按已知值预填，
  凭据本体留 `TODO(user)` 标记。SANDBOX_PROVIDER 已设 e2b
- compose 栈运行中且全绿：app（127.0.0.1:8787）+ SeaweedFS（127.0.0.1:9000）+ Litestream
  （86 迁移、cron/闹钟/轮询 running、首快照已入备份桶）
- **web 已容器化加入栈**（新文件，未提交）：`packages/web/Dockerfile`（standalone 输出，
  NEXT_PUBLIC_WS_URL 作 build arg）+ `docker-compose.web.yml` overlay（web 走
  127.0.0.1:3000，服务端经 compose 网络访问 app:8787）。`.dockerignore` 白名单放行
  packages/web/ 并排除其 .env*/.next。启动：
  `docker compose -f docker-compose.yml -f docker-compose.web.yml up -d`
- Dockerfile 坑：`next build` 类型检查会扫到 import eslint 的 co-located
  测试，而 eslint 是仓库根 devDependency——workspace 定向安装需补 `npm install --no-save
  eslint@^9.18.0`
- compose smoke 端到端通过（会话往返/附件/WS 令牌/沙箱回复/cron/复制/排水/缺密钥报错全 ok）

**公网接入（2026-09-29 全链路已通）**：`https://sso.qinyuan.cloud:8443`（前置机
`43.161.245.204` 终止 TLS）路径分流——`/sessions/*` → 本机 `http://39.109.109.53:8787`（CP
浏览器 WS，已验证 404 行为与直连 CP 一致），其余 → 本机 `http://39.109.109.53:8123`（web，
已验证登录页 200）。`.env` 三 URL 就位、web 镜像已重建（wss 地址内联确认）、8787 公网可达
（沙箱直连，E2B 海外→国内连通性待真实会话验证）。
**防火墙：已按用户决定整体移除**（原 DOCKER-USER 限源规则在 iptables-nft 后端上源豁免
未生效导致前置机 SYN 被误杀，删除后即通；持久化已重存）。因此 8123 目前对公网明文可达，
如需收紧应去**阿里云安全组**把 8123/TCP 限源 43.161.245.204（云边缘层，无宿主机 netfilter
的怪异行为）。

**E2B 沙箱不回连的根因（2026-09-29 已定位，待用户加前置机端口后修复）**：沙箱运行时
`sandbox_runtime/runtime_config.py` 的 `_validate_control_plane_url` 只放行 `https://*` 或
`http://localhost|127.0.0.1|::1`——`WORKER_URL=http://39.109.109.53:8787` 导致 entrypoint
启动即抛 ValueError 崩溃（supervisor 日志 /tmp/oi-supervisor.log），bridge 永不连接，240s
`sandbox.connecting_timeout`。网络本身全通（已验证：沙箱→本机 8787 直连 RTT ~135ms；
沙箱→前置机 TLS→8787/8123 双路径 marker 均到达）。WSL2 时代能用是因为 WORKER_URL 是
cloudflared 的 https 隧道。**修复**：前置机加 8444(TLS) 全路径裸转发到
`http://39.109.109.53:8787`，然后本机 .env 改 `WORKER_URL=https://sso.qinyuan.cloud:8444`
并 `up -d --force-recreate`（web 镜像无需重建）。诊断手法记录：envd 在
`https://49983-<sandboxID>.e2b.app/process.Process/Start`（Connect 信封：1 flag 字节+4 字节
BE 长度+JSON，头 X-Access-Token；create 时传 secure:true 拿 token、envVars 随 create 下发）；
进程输出读不回，用"结果编码进 URL + 本机 tcpdump -A"当回传通道。

**自定义模型思考等级修复（2026-09-29 已上线）**：根因是推理配置只查静态目录
`MODEL_REASONING_CONFIG`，cpa-/cpo- 模型查不到 → web 无 Effort 子菜单、控制面
`validateReasoningEffort` 把值当无效丢弃。修复三层：shared 新增
`customModelReasoningConfig(efforts)`（从注册表 efforts 建配置，过滤 "none"、按严重度排序、
默认优先 high）+ `isValidReasoningEffort` 增加注册表参数 + `ModelDisplayInfo.reasoningEfforts`；
web 在 `use-enabled-models` 的 custom items 里带 efforts，选择器/`resolveModelPreference`/
`defaultReasoningEffort` 回退到它；控制面 `validateReasoningEffort` 加 customEfforts 参数，
`SessionMessageQueue`/`SessionInitHandler` 注入 `getCustomModelReasoningEfforts`（惰性查
`CustomProviderStore.resolveCustomModel`，components.ts 一处闭包两处复用）。沙箱侧本就把
effort 透传给 harness，无需重建模板。测试：shared 1061 / web 1990 / CP 5317 全过，镜像已重建
上线。注意：菜单里显示的等级来自导入模型时存的 reasoningEfforts（如 glm-5.3 =
["xhigh","max","high"]），在设置页可改。

**pi/dsh 自定义 provider 支持（2026-09-29 已上线，E2B 模板 f3852b09ddf1）**：
- **pi**：`prepare()` 把 provider 写进 `~/.pi/agent/models.json`（`{providers:{key:{baseUrl,api,apiKey:"$ENV",models:[{id}]}}}`；api 映射 anthropic→anthropic-messages / openai_compatible→openai-completions / openai_responses→openai-responses；URL 约定与 codex 相同）。模型串 `{key}/{model}` 与既有 `--model` 直通；`--thinking` 的 none→off 映射。已在新模板验证 `pi --list-models` 列出 cpo 模型。
- **dsh**：`prepare()` 写 `~/.dsh/profiles/headless/cordis.patch.yml`（`- id: llm-pi-ai` config.providers.{key}：displayName/api/baseURL/apiKeyEnv/models{id,name,contextWindow,maxTokens,input,reasoningEfforts}）；每回合 `build_argv` 写 `/tmp/oi-dsh-model-selection.yml`（`- id: agent-default-model` 的 provider/model/reasoningEffort，none→off）并以 `--patch` 注入。凭据：apiKeyEnv 直接解析同名环境变量（"store through credentials service or export it"），无需写 .credentials.yaml。
- **门控**：shared 新增 `customProviderProtocols` 能力 + `harnessSupportsCustomModel(harness, model, protocol?)`（模型串不含协议——两个 OpenAI 线协议共用 cpo- 前缀，需注册表元数据）；**codex 仅 openai_responses**（用户定），pi 保持 any，**dsh 族放开 custom-anthropic/custom-openai**，zcode 暂不动（族仍 zai-only）。web `ModelDisplayInfo.protocol` 随选项传递，`filterModelOptionsForHarness` 走新 helper；CP 侧 checkHarnessCompatibility 保持串级（无协议信息，宽松）。
- 测试：python 1447 / shared 1063 / web 1986 全过；E2B 模板重建（uv 装在 ~/.local/bin，构建需 PATH 含它）并切换 `open-inspect-sandbox-f3852b09ddf1-1790705766787503573`。
- **zcode 未完成（下一轮）**：研究到 80%——支持 anthropic-messages/openai-chat-completions/openai-responses 三协议，provider 声明在 `~/.zcode/v2/provider_config.json`（access:{type:"api-key",apiKey} + api:{type,baseUrl,headers}，源码 /opt/openinspect/zcode/packages/provider/src/config/provider-data-schema.ts），模型选择 defaultModelSelection（modelSelectionSchema 在 packages/shared/src/model-selection.ts）。剩余：文件顶层 JSON 结构、--prompt 怎么选模型、凭据文件格式（resolveCredentialFilePath）。完成后放开 zcode 族即可。
- **诊断通道（复用）**：envd Connect 信封流**直接带 stdout**（base64 event.data.stdout），`49983-{id}.e2b.app/process.Process/Start` 起 `/opt/openinspect/python/bin/python -c` 即得输出，不再需要 URL 回传。

**会话创建丢失思考等级修复（2026-09-29 二次上线）**：症状=创建时选 max，进会话变 high。
根因：`routes/session-create.ts` 建会话路由层裸调 `isValidReasoningEffort(model, effort)`（无注册表
参数）→ 自定义模型一律 false → 存 null → 进会话回退到模型默认（启发式优先 high）。上次只修了
session-init/message-queue，漏了这第三道闸门。本轮以 `grep isValidReasoningEffort` 拉全所有闸门
并全部修掉：新增 `routes/custom-model-efforts.ts` helper（惰性查注册表，失败降级静态判定），
接入 session-create、session-child-spawn（拒绝信息也带上注册表等级）、automation-crud（create+update，
resolveReasoningEffort 加 customEfforts 参数）、integration-settings（github/linear 读取路径）、web
automations 表单（automation-form-policy + customModelEfforts helper）。未动的：slack/linear-bot 包
（CF 侧独立部署，本栈不跑）。测试 CP 5317 / web 1986 全过，镜像已重建上线。**回归验证方法**：建会话
选 max → 重进会话应仍显示 max；查库 `sessions/<id>.db` 的 session 行 reasoning_effort 应为 max。

**dsh reasoningEfforts 格式修复（2026-09-30 已上线，模板 0ebe7efa6091）**：dsh 的模型
`reasoningEfforts` 不是等级数组而是 `{等级: 线上值}` 映射（仅 "off" 可留空、必须至少一个非 off
等级）——数组格式使整个 llm-pi-ai 服务校验失败不激活（NO_ADAPTER）。已改为恒等映射
`{"high": "high", ...}`，新模板实测：服务激活、`--patch` 选模型后回合真实到达自定义端点。
注意实证细节：`cordis.patch.yml` 里 llm-pi-ai 的 config 覆盖生效，但 agent-default-model 的
config 覆盖在用户层不生效（原因未深究）——每回合选模型必须走 `--patch` 叠加（vendor 已如此）。

**zcode 结论（2026-09-30 研究完毕，实现待做）**：zcode 0.16.9 的 `--prompt` 对新会话**不做模型
种子**——defaultModelSelection 只进 TUI 路径（tui-prompt-handler-runtime 消费），runtimeConfig
无 modelSelection 项、CLI 无 --model 参数（AgentRuntime 从 config.modelSelection 初始化，
prompt-command 不传）。内置 provider 做默认选择同样报 "Select a model before continuing"（已实证），
即 vendor docstring 所指 "v3.14.3 源码构建" 的行为已随版本漂移失效。**正确路径：把 ZcodeVendor
改造为 `zcode app-server`（ZCode Protocol stdio）驱动**——常驻进程、setModel 指令 + 回合提交，
这也是 ZCode 官方的无头集成方式。provider 注册侧已全部探明：`~/.zcode/v2/provider_config.json`
（`{schemaVersion:1, config:{providerConfigRules:{providerRules:{<id>:{access:{type:"api-key",apiKey},
api:{type:"anthropic-messages"|"openai-chat-completions"|"openai-responses", baseUrl, headers?},
personalModelIds}}}, modelConfigRules:{providerModelRules:[], manualProviderModelRules:[{providerId,
modelId, config:{enabled, properties:{...全字段必填}, optionSpecs:{reasoningLevel:{values,map},
maxOutputTokens:{max,map}}}}]}, defaultModelSelection}}`；map 是编译表达式字符串，恒等式
`{"reasoning_effort": reasoningLevel}`（OpenAI 系）；文件 schema 见
packages/provider-node/src/provider-config-file-codec.ts，字段 schema 见
packages/provider/src/config/{provider-data-schema,model-config 由 shared/model-config}.ts，
内置范例 /opt/openinspect/zcode/config/provider/zcode-builtin.json。

**剩余收尾**：

1. 用户实测 dsh（修后）与 pi；codex 菜单只剩 Responses 协议网关
2. zcode：app-server 协议驱动重写 + provider_config.json staging + 放开 zcode 模型族

**owner 提权（2026-09-29 已完成）**：用户首次登录后（canonical id
`462d0b1645a0e34d9a6eb55b16a860e3`）用仓库 `scripts/bootstrap-workspace-owner.ts` 导出的
`buildBootstrapSql`（与正式脚本同语义：preflight → 审计+角色变更同事务 → postcondition 验证）
对容器内 `/data/global.db` 执行（`docker exec` + `node:sqlite`，D1-only 的 CLI 走不了）。
结果 executed、`role_builtin_owner`、audit_written=1。重装/重建后重做此步骤。

**凭据已配（2026-09-29，来源 /tmp/cf-config tar，已配后删除）**：GITHUB_CLIENT_ID/SECRET、
GITHUB_APP_PRIVATE_KEY（PKCS#8，\n 单行转义；已用 JWT+GET /app 只读验证：app slug
xuanyi-background-agents、installation 166014231 @ yuxuanyi1-droid）、E2B_API_KEY（已验证）、
E2B_TEMPLATE_ID=`open-inspect-sandbox-6d354c1e2400-1790691850368658861`（E2B API 确认 ready；
tar 里的裸名和 aaabdeb2 都是旧值，勿用）、E2B 超时 3300s/自动暂停 true、
ALLOWED_USERS=yuxuanyi1-droid、DEPLOYMENT_NAME=yuxuanyi1。加密密钥类未复用旧值（本机
全新生成，数据库与新人绑定）。踩坑记录：把 `\n` 转义 PEM 写进 .env 时不能用 `re.sub`
（替换串里的 `\n` 会被展开成真换行导致 compose 解析失败），需逐行拼接。

WSL2 侧原环境（8788 端口 node 模式、3000 next dev、SeaweedFS、隧道）不受影响，仍在。

## 七、下一步候选（按用户意向）

1. ~~推送分支~~（已完成）
2. ~~服务器部署基础栈~~（已完成，见上节；余下为填凭据 + 域名接入）
3. **隧道根治**（仅 WSL2 本地开发需要）：named tunnel（需自有域名托管 CF）或 ngrok 静态域名
4. **pi 接自定义 provider**（仿 codex：cli_vendors prepare 写 pi 的 provider 配置）

## 八、新会话续接方法

- WSL2 原机：`#sess_bb1032c9-7ce3-4e33-bdba-c8bea63b1ec2` 引用原会话；服务器新会话：读本文件即可
- 服务器栈日常操作：`cd /root/background-agents && docker compose -f docker-compose.yml -f
  docker-compose.web.yml ps/logs/up -d`；数据在 `background-agents_control-plane-data` 卷
  （Litestream 每秒复制 global.db 到本地 backups 桶）
- 服务器上未提交的新文件：`packages/web/Dockerfile`、`docker-compose.web.yml`、
  `.dockerignore` 修改、本文件更新——待 review 后随分支提交
