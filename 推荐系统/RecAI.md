# RecAI

### InteRecAgent

LLM：对话

### RecLM-emb

？LLM知识的滞后性

### RecLM-gen

？为什么不直接对教师模型进行微调

### RecLM-cgen

？既然RecLM-cgen才适用于工业场景，那么RecLM-gen一般用于什么场景？有什么用途？

在实际的 InteRecAgent 运行中： 系统通常先用 RecLM-gen 理解用户的“灵魂需求”，再让它调用 RecLM-cgen/ret 锁死“合法候选池”确保不出错，最后再由 RecLM-gen 组织语言把结果优雅地推销出去。

### RecExplainer

？怎么保证解释LLM的知识/经验是高于推荐模型的