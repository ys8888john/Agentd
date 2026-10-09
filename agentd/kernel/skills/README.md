# 技能目录（Agent Skills / SKILL.md 标准）

agentd 的技能分四层，同名时**靠近用户的覆盖靠里的**：

| 层 | 位置 | 说明 |
|---|---|---|
| builtin | `agentd/kernel/skills/<name>/`（本目录） | 随包手写的技能 |
| upstream | `vendor/<repo>`（**git submodule**） | 上游开源技能库，按 `skills_sources.json` 白名单暴露 |
| user | `~/.agentd/skills/` | 用户自装 |
| project | `<工作目录>/.agentd/skills/` | 项目级，跟仓库走 |

## upstream 层：submodule

上游仓库以 submodule 引用，内容不再拷贝进本仓库，`git submodule
update --remote` 即可跟进上游。当前引入：

| submodule | 来源 | 许可 | 暴露的白名单 |
|---|---|---|---|
| `vendor/superpowers` | https://github.com/obra/superpowers | MIT | systematic-debugging、verification-before-completion、test-driven-development、writing-plans |
| `vendor/anthropic-skills` | https://github.com/anthropics/skills | 各目录自明（示例为 Apache-2.0） | skill-creator |

**克隆必须带 submodule**，否则 upstream 层整层缺席（不报错，但技能没了）：

```bash
git clone --recurse-submodules https://github.com/ys8888john/Agentd.git
# 已克隆过的：
git submodule update --init
```

白名单在 `agentd/kernel/skills_sources.json`。**为什么要有白名单**：
上游库不是所有内容都适合暴露 —— superpowers 里有依赖 Claude Code 子代理
机制的技能（dispatching-parallel-agents 等），anthropics/skills 里的
docx/pdf/pptx/xlsx 是 **source-available 专有许可**（只能引用、不能纳入
我们的分发，且与原生 make_xlsx 重叠）。往白名单加名字 = 同意把该技能
暴露给模型，加之前先看清它的许可和依赖。

## 怎么加新技能

- 用户级：技能目录放进 `~/.agentd/skills/`；
- 项目级：放进工作目录 `.agentd/skills/`；
- 内置级：放进本目录并随包发布；
- 上游级：`git submodule add <repo> vendor/<name>`，然后在
  `skills_sources.json` 的 allow 里点名。

技能格式：目录内一个 `SKILL.md`，YAML frontmatter 至少有 `name` 和
`description`（一句话说清"做什么、什么时候用"——它是路由依据），正文是
完整工作流；可带 `scripts/`、`references/`。
