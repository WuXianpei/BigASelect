"""网络环境：本进程内强制直连（不修改系统代理设置）"""

from __future__ import annotations

from typing import Any

# 本程序全部数据源均为国内站点，无需走代理：
# 东财、新浪、同花顺、腾讯 qt.gtimg.cn 等
_DIRECT_CONFIGURED = False


def setup_network(settings: dict[str, Any] | None = None) -> None:
    """
    启动时为本 Python 进程启用直连。

    仅 patch 进程内的 requests / urllib，不读写、不修改：
    - Windows 系统代理设置
    - 环境变量 HTTP_PROXY / HTTPS_PROXY 等
    - 其他程序的网络配置
    """
    if settings is not None and not settings.get("process_direct", True):
        return
    _enable_process_direct(verbose=True)


def _enable_process_direct(*, verbose: bool = False) -> None:
    """进程级直连：忽略系统代理，仅对本程序生效"""
    global _DIRECT_CONFIGURED
    if _DIRECT_CONFIGURED:
        return

    _patch_requests()
    _patch_urllib()

    _DIRECT_CONFIGURED = True
    if verbose:
        print(
            "[网络] 本程序使用进程级直连（数据源均为国内站点，无需代理；"
            "不影响系统及其他程序的网络设置）"
        )


def _patch_requests() -> None:
    """新建 requests.Session 时不读取系统/环境代理"""
    try:
        from requests.sessions import Session

        if getattr(Session, "_bigaselect_direct_patched", False):
            return

        _orig_init = Session.__init__

        def _init(self, *args: Any, **kwargs: Any) -> None:
            _orig_init(self, *args, **kwargs)
            self.trust_env = False
            self.proxies = {"http": None, "https": None}

        Session.__init__ = _init  # type: ignore[method-assign]
        Session._bigaselect_direct_patched = True  # type: ignore[attr-defined]
    except ImportError:
        pass


def _patch_urllib() -> None:
    """本进程内 urllib 不读取系统代理（akshare 部分接口使用 urllib）"""
    try:
        import urllib.request

        if getattr(urllib.request, "_bigaselect_direct_patched", False):
            return

        _orig_getproxies = urllib.request.getproxies

        def _no_proxies() -> dict[str, str]:
            return {}

        urllib.request.getproxies = _no_proxies  # type: ignore[method-assign, assignment]
        urllib.request._bigaselect_orig_getproxies = _orig_getproxies  # type: ignore[attr-defined]
        urllib.request._bigaselect_direct_patched = True  # type: ignore[attr-defined]
    except ImportError:
        pass
