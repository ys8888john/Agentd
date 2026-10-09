# 内置技能（Agent Skills / SKILL.md 标准）

这个目录下的每个子目录都是一个随包捆绑的技能，由 `agentd/kernel/skills.py`
的发现器扫描、经 `load_skill` 工具按需加载完整工作流。

## 来源与许可

| 技能 | 来源 | 许可 |
|---|---|---|
| systematic-debugging | https://github.com/obra/superpowers （skills/） | MIT（见上游仓库 LICENSE） |
| verification-before-completion | 同上 | MIT |
| test-driven-development | 同上 | MIT |
| writing-plans | 同上 | MIT |
| skill-creator | https://github.com/anthropics/skills （skills/skill-creator） | Apache-2.0（目录内 LICENSE.txt） |

拷贝时的改动：
- superpowers 四个技能去掉了作者的 `CREATION-LOG.md` 与 `test-*.md`
  自测夹具，其余原样保留；
- skill-creator 整目录原样保留（含 references/ 与 scripts/，脚本可经
  run_command 执行）。

**注意**：anthropics/skills 里的 docx/pdf/pptx/xlsx 文档技能是
source-available（专有），**不能**拷进本仓库 —— Excel/文档能力走 agentd
自己的原生工具（make_xlsx 等）。

## 怎么加新技能

- 用户级：把技能目录放进 `~/.agentd/skills/`（优先级高于内置，同名覆盖）；
- 项目级：放进工作目录 `.agentd/skills/`（跟着仓库走，优先级最高）；
- 内置级：放进本目录并随包发布（改动需要过测试）。

技能格式：目录内一个 `SKILL.md`，YAML frontmatter 至少有 `name` 和
`description`（一句话说清"做什么、什么时候用"——它是路由依据），正文是
完整工作流；可带 `scripts/`、`references/`。
