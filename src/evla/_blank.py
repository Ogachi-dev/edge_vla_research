"""穴埋め用のプレースホルダ `___`。

穴埋めファイルでは `x = ___` のように書いてあり、ファイル自体は import できる。
埋め忘れた `___` が実行時に使われると、NotImplementedError で場所が分かる。
残っている穴の一覧を出すコマンドは docs/03_fill_in_guide.md の1節にある。
"""

from __future__ import annotations

from typing import Any, NoReturn


class _Blank:
    def _fail(self, *_: Any, **__: Any) -> NoReturn:
        raise NotImplementedError("穴埋め（___）が残っています。トレースバックの行を埋めてください")

    __call__ = __getattr__ = __getitem__ = __iter__ = __len__ = __bool__ = _fail
    __add__ = __radd__ = __sub__ = __rsub__ = __mul__ = __rmul__ = _fail
    __truediv__ = __rtruediv__ = __floordiv__ = __mod__ = __pow__ = __neg__ = _fail
    __lt__ = __le__ = __gt__ = __ge__ = __float__ = __int__ = __index__ = _fail
    __matmul__ = __rmatmul__ = __and__ = __or__ = __enter__ = __exit__ = _fail
    __array__ = __setitem__ = _fail

    def __repr__(self) -> str:
        return "___"


___: Any = _Blank()
