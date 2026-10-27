r"""交互式引擎选择向导：用户通过状态机逐级选择 后端 -> 模型 -> 权重类型。

状态流转：

    SELECT_BACKEND -> SELECT_MODEL -> SELECT_WEIGHT -> CONFIRM -> DONE
         \               \               \               \
          ---------------->------------>-------> CANCELLED (输入 q)

用法：
    from embedding_service.wizard import EngineWizard
    engine = EngineWizard().run()   # 返回就绪的 EmbeddingEngine，取消返回 None
"""

from enum import Enum, auto
from pathlib import Path

from .core.engine import MODEL_ROOT, EmbeddingEngine


class WizardState(Enum):
    SELECT_BACKEND = auto()
    SELECT_MODEL = auto()
    SELECT_WEIGHT = auto()
    CONFIRM = auto()
    DONE = auto()
    CANCELLED = auto()


_TRANSITIONS = {
    WizardState.SELECT_BACKEND: {WizardState.SELECT_MODEL, WizardState.CANCELLED},
    WizardState.SELECT_MODEL: {WizardState.SELECT_WEIGHT, WizardState.CANCELLED},
    WizardState.SELECT_WEIGHT: {WizardState.CONFIRM, WizardState.CANCELLED},
    WizardState.CONFIRM: {WizardState.DONE, WizardState.SELECT_BACKEND, WizardState.CANCELLED},
    WizardState.DONE: set(),
    WizardState.CANCELLED: set(),
}


def discover_options(model_root: Path) -> list[dict]:
    """扫描 embedding_service/model，列出所有后端/模型/权重组合。"""
    options = []
    for sub in sorted(model_root.iterdir()):
        if not sub.is_dir():
            continue
        for f in sub.glob("*.onnx"):
            if "text" not in f.stem:  # 跳过 vision/audio/lm 等多模态组件
                continue
            options.append(
                {"backend": "onnx", "model": f.stem, "weight": "fp32", "dir": sub}
            )
        for f in sub.glob("*.xml"):
            if "text" not in f.stem:
                continue
            weight = "int8-w8" if "w8" in sub.name else "fp16"
            options.append(
                {"backend": "openvino", "model": f.stem, "weight": weight, "dir": sub}
            )
    return options


class EngineWizard:
    def __init__(self, model_root: str | Path = MODEL_ROOT):
        self.options = discover_options(Path(model_root))
        if not self.options:
            raise RuntimeError(f"no models found under {model_root}, run tools/ scripts first")
        self.choice: dict = {}
        self.state = WizardState.SELECT_BACKEND
        print(f"[wizard] -> {self.state.name}")

    def _transition(self, to: WizardState):
        if to not in _TRANSITIONS[self.state]:
            raise RuntimeError(f"illegal transition: {self.state.name} -> {to.name}")
        self.state = to
        print(f"[wizard] -> {to.name}")

    def _ask(self, title: str, items: list, fmt) -> int | None:
        """编号菜单；返回选中下标，q 取消。非法输入停留在当前状态重问。"""
        print(f"\n{title}")
        for i, item in enumerate(items, 1):
            print(f"  {i}) {fmt(item)}")
        while True:
            raw = input("> ").strip().lower()
            if raw == "q":
                return None
            if raw.isdigit() and 1 <= int(raw) <= len(items):
                return int(raw) - 1
            print("invalid input, try again (q to cancel)")

    def run(self) -> EmbeddingEngine | None:
        # 1) 后端
        backends = sorted({o["backend"] for o in self.options})
        idx = self._ask("选择推理后端:", backends, str)
        if idx is None:
            self._transition(WizardState.CANCELLED)
            return None
        self.choice["backend"] = backends[idx]
        pool = [o for o in self.options if o["backend"] == self.choice["backend"]]
        self._transition(WizardState.SELECT_MODEL)

        # 2) 模型
        models = sorted({o["model"] for o in pool})
        idx = self._ask("选择模型:", models, str)
        if idx is None:
            self._transition(WizardState.CANCELLED)
            return None
        self.choice["model"] = models[idx]
        pool = [o for o in pool if o["model"] == self.choice["model"]]
        self._transition(WizardState.SELECT_WEIGHT)

        # 3) 权重类型
        idx = self._ask("选择权重类型:", pool, lambda o: f"{o['weight']}  ({o['dir']})")
        if idx is None:
            self._transition(WizardState.CANCELLED)
            return None
        self.choice.update(pool[idx])
        self._transition(WizardState.CONFIRM)

        # 4) 确认
        print(
            f"\n确认选择: backend={self.choice['backend']}  model={self.choice['model']}"
            f"  weight={self.choice['weight']}  dir={self.choice['dir']}"
        )
        while True:
            raw = input("确认? [y/n/r=重新选择] ").strip().lower()
            if raw == "y":
                break
            if raw == "r":
                self.choice = {}
                self._transition(WizardState.SELECT_BACKEND)
                return self.run()
            if raw in ("n", "q"):
                self._transition(WizardState.CANCELLED)
                return None

        self._transition(WizardState.DONE)
        return EmbeddingEngine(
            model_dir=self.choice["dir"], prefer=[self.choice["backend"]]
        )
