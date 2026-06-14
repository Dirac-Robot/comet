"""Per-call MCP bridge for ``claude -p`` so it can call bound LangChain tools
natively (as ``mcp__cobra__<tool>``) instead of going through a system-prompt
envelope.

Lifecycle: one bridge per ``_generate()`` call. Spin up a streamable-HTTP MCP
server on a free localhost port, hand the URL to ``claude -p`` via
``--mcp-config``, wait for ``claude`` to finish, then tear the server down.

The bridge stays free of any host-side knowledge — it only sees LangChain
``BaseTool`` objects and two opaque callbacks (``ctx_capture`` to grab whatever
threadlocal/contextvar state the host wants the tool handler to see, and
``ctx_apply`` to restore it on the MCP handler thread before invoking the
tool). Hosts that don't need context propagation can leave both at ``None``.
"""
from __future__ import annotations

import asyncio
import contextlib
import json
import socket
import threading
import uuid
from collections import deque
from typing import Any, Callable

import uvicorn
from loguru import logger
from mcp.server import Server
from mcp.server.streamable_http_manager import StreamableHTTPSessionManager
from mcp.types import ImageContent, TextContent, Tool
from starlette.applications import Starlette
from starlette.routing import Mount


MCP_SERVER_NAME = 'cobra'
MCP_TOOL_PREFIX = f'mcp__{MCP_SERVER_NAME}__'


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(('127.0.0.1', 0))
        return s.getsockname()[1]


def _tool_input_schema(tool: Any) -> dict[str, Any]:
    args_schema = getattr(tool, 'args_schema', None)
    if args_schema is None:
        return {'type': 'object', 'properties': {}}
    for accessor in ('model_json_schema', 'schema'):
        fn = getattr(args_schema, accessor, None)
        if callable(fn):
            try:
                schema = fn()
            except Exception:
                continue
            if isinstance(schema, dict):
                return schema
    args = getattr(tool, 'args', None)
    if isinstance(args, dict):
        return {'type': 'object', 'properties': args}
    return {'type': 'object', 'properties': {}}


def _invoke_tool_safely(tool: Any, arguments: dict[str, Any]) -> str:
    try:
        result = tool.invoke(arguments)
    except Exception as e:
        logger.warning(f'MCP bridge: tool {getattr(tool, "name", "?")} raised: {e}')
        return f'[error] tool {getattr(tool, "name", "?")} raised: {e}'
    if isinstance(result, str):
        return result
    return str(result)


def _to_mcp_content(content: Any) -> list[Any]:
    """Convert a tool result (a plain string, or a LangChain multimodal block
    list with ``image_url`` data URLs) into MCP content blocks. Image blocks
    become ``ImageContent`` so the model actually SEES the image — this is the
    only path by which a CoBrA tool's image output (e.g. read_file_tool on an
    image, via the chat-engine image intercept) reaches an oauth:claude model.
    Without it the multimodal result is flattened to a useless base64 string."""
    blocks: list[Any] = []
    if isinstance(content, str):
        if content:
            blocks.append(TextContent(type='text', text=content))
    elif isinstance(content, list):
        for part in content:
            if isinstance(part, str):
                if part:
                    blocks.append(TextContent(type='text', text=part))
                continue
            if not isinstance(part, dict):
                continue
            ptype = part.get('type')
            if ptype == 'text':
                text = part.get('text', '')
                if text:
                    blocks.append(TextContent(type='text', text=text))
            elif ptype == 'image_url':
                url = (part.get('image_url') or {}).get('url', '')
                if url.startswith('data:'):
                    header, _, b64 = url.partition(',')
                    # header looks like 'data:image/png;base64'
                    mime = header[5:].split(';', 1)[0] or 'image/png'
                    if b64:
                        blocks.append(ImageContent(type='image', data=b64, mimeType=mime))
    elif content is not None:
        blocks.append(TextContent(type='text', text=str(content)))
    # MCP requires a non-empty content list.
    if not blocks:
        blocks.append(TextContent(type='text', text=''))
    return blocks


class ClaudeOAuthMcpBridge:
    """Background MCP server that exposes ``tools`` to a single ``claude -p``
    invocation. Starts on construction-via-:meth:`start`, blocks until ready,
    and tears down via :meth:`stop`.

    Both lifecycle methods are sync — designed to be called from the same
    thread that drives the synchronous ``subprocess.run(claude_bin, ...)``
    call. The server itself runs on a daemon thread with its own asyncio
    loop so it can serve requests in parallel with the subprocess.
    """

    def __init__(
        self,
        tools: tuple[Any, ...],
        *,
        ctx_capture: Callable[[], dict[str, Any]] | None = None,
        ctx_apply: Callable[[dict[str, Any]], None] | None = None,
        ready_timeout: float = 10.0,
        holding: bool = False,
        on_call: Callable[[str, str, dict[str, Any]], None] | None = None,
    ):
        """``holding=True`` switches the bridge into a synchronization mode
        where ``call_tool`` blocks until the host resolves it via
        :meth:`respond`. Combined with ``on_call`` (invoked with
        ``(call_id, name, arguments)``) this lets the host surface tool
        calls to its own tool loop and forward the result back to claude,
        keeping a single ``claude -p`` process alive for the whole turn
        instead of paying cold-start per iteration. ``on_call`` runs on
        the asyncio loop thread — keep it lock-free and fast."""
        self._tools = tuple(t for t in tools if getattr(t, 'name', None))
        self._tools_by_name = {t.name: t for t in self._tools}
        self._ctx_snapshot: dict[str, Any] = ctx_capture() if ctx_capture else {}
        self._ctx_apply = ctx_apply
        self._ready_timeout = ready_timeout
        self._holding = holding
        self._on_call = on_call

        self._port: int | None = None
        self._uvicorn_server: uvicorn.Server | None = None
        self._thread: threading.Thread | None = None
        self._ready = threading.Event()
        self._startup_error: BaseException | None = None

        self._loop: asyncio.AbstractEventLoop | None = None
        self._pending: 'deque[tuple[str, asyncio.Future[str]]]' = deque()
        self._pending_lock = threading.Lock()

    @property
    def url(self) -> str:
        if self._port is None:
            raise RuntimeError('MCP bridge not started')
        return f'http://127.0.0.1:{self._port}/mcp'

    @property
    def allowed_tool_glob(self) -> str:
        return f'{MCP_TOOL_PREFIX}*'

    def mcp_config_json(self) -> str:
        """JSON value suitable for ``claude -p --mcp-config <value>``."""
        return json.dumps({
            'mcpServers': {
                MCP_SERVER_NAME: {
                    'type': 'http',
                    'url': self.url,
                },
            },
        })

    def start(self) -> None:
        if self._thread is not None:
            return
        if not self._tools:
            # No tools to expose — nothing to start. Caller should skip MCP.
            self._ready.set()
            return
        self._port = _free_port()
        self._thread = threading.Thread(
            target=self._run_thread,
            daemon=True,
            name='claude-oauth-mcp',
        )
        self._thread.start()
        if not self._ready.wait(timeout=self._ready_timeout):
            raise RuntimeError(
                f'Claude OAuth MCP bridge failed to become ready within '
                f'{self._ready_timeout:g}s'
            )
        if self._startup_error is not None:
            raise RuntimeError(
                f'Claude OAuth MCP bridge startup failed: {self._startup_error}'
            )

    def stop(self) -> None:
        # Release any holding-mode futures so claude isn't left waiting on
        # a torn-down MCP socket. This is a no-op in schema-only mode.
        self.cancel_pending(error='bridge stopped')
        srv = self._uvicorn_server
        if srv is not None:
            srv.should_exit = True
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=5.0)
        self._thread = None
        self._uvicorn_server = None
        self._loop = None

    def __enter__(self) -> 'ClaudeOAuthMcpBridge':
        self.start()
        return self

    def __exit__(self, *exc: Any) -> None:
        self.stop()

    def _run_thread(self) -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(self._async_main())
        except BaseException as e:
            self._startup_error = e
            self._ready.set()
            logger.warning(f'MCP bridge thread exited with error: {e}')
        finally:
            # Cancel and drain anything still pending — uvicorn's graceful
            # shutdown doesn't always reach SSE/streamable-HTTP background
            # tasks (sse_starlette's _shutdown_watcher, in particular), and
            # leaving them on a closing loop produces noisy "Task was
            # destroyed but it is pending!" warnings at exit. Cleaning the
            # task list here keeps stdout quiet without changing behavior.
            try:
                pending = [t for t in asyncio.all_tasks(loop) if not t.done()]
                for task in pending:
                    task.cancel()
                if pending:
                    loop.run_until_complete(
                        asyncio.gather(*pending, return_exceptions=True)
                    )
            except Exception:
                pass
            try:
                loop.close()
            except Exception:
                pass

    def respond(self, call_id: str, content: Any) -> None:
        """Resolve a pending holding-mode tool call. ``content`` is normally a
        string but may be a LangChain multimodal block list (image reads),
        converted to MCP content blocks when the call returns. Safe to invoke
        from any
        thread — the future itself is set via ``call_soon_threadsafe`` so it
        always lands on the bridge's asyncio loop.

        ``call_id`` matches the id surfaced through ``on_call``; if it's
        unknown (e.g. the host pushed a stale response after the bridge
        torn down), the call is silently ignored."""
        loop = self._loop
        if loop is None or loop.is_closed():
            return
        with self._pending_lock:
            future: asyncio.Future[str] | None = None
            kept: deque = deque()
            while self._pending:
                cid, fut = self._pending.popleft()
                if cid == call_id and future is None:
                    future = fut
                    continue
                kept.append((cid, fut))
            self._pending = kept
        if future is None:
            return
        loop.call_soon_threadsafe(future.set_result, content)

    def cancel_pending(self, error: str = 'cancelled') -> None:
        """Resolve every outstanding holding-mode call with an error string,
        used during teardown so claude doesn't sit on a stuck MCP request."""
        loop = self._loop
        if loop is None or loop.is_closed():
            return
        with self._pending_lock:
            pending = list(self._pending)
            self._pending.clear()
        for _, fut in pending:
            loop.call_soon_threadsafe(
                lambda f=fut: f.set_result(f'[error] {error}') if not f.done() else None
            )

    async def _async_main(self) -> None:
        self._loop = asyncio.get_running_loop()
        server = Server(MCP_SERVER_NAME)

        @server.list_tools()
        async def _list_tools() -> list[Tool]:
            return [
                Tool(
                    name=tool.name,
                    description=(getattr(tool, 'description', '') or '')[:1024],
                    inputSchema=_tool_input_schema(tool),
                )
                for tool in self._tools
            ]

        @server.call_tool()
        async def _call_tool(name: str, arguments: dict[str, Any] | None):
            args = arguments or {}
            if self._holding:
                # Hand the call to the host (its tool loop runs the tool
                # through its normal per-tool side-effect pipeline) and
                # block this MCP response until the host pushes a result
                # back through :meth:`respond`.
                loop = asyncio.get_running_loop()
                future: asyncio.Future[str] = loop.create_future()
                call_id = f'oauth_call_{uuid.uuid4().hex[:12]}'
                with self._pending_lock:
                    self._pending.append((call_id, future))
                if self._on_call is not None:
                    try:
                        self._on_call(call_id, name, args)
                    except Exception as e:
                        future.set_exception(e)
                try:
                    result = await future
                except Exception as e:
                    return [TextContent(type='text', text=f'[error] {e}')]
                # result may be a plain string OR a LangChain multimodal block
                # list (image reads) — convert so images ride as ImageContent.
                return _to_mcp_content(result)

            tool = self._tools_by_name.get(name)
            if tool is None:
                return [TextContent(type='text', text=f'[error] unknown tool: {name}')]
            apply_cb = self._ctx_apply
            snapshot = self._ctx_snapshot
            # tool.invoke is sync; run it in the default executor so we don't
            # block the asyncio loop while the tool does I/O. Apply the host's
            # context snapshot on whatever thread actually runs the tool.
            def _run() -> str:
                if apply_cb is not None and snapshot:
                    try:
                        apply_cb(snapshot)
                    except Exception as e:
                        logger.warning(f'MCP bridge: ctx_apply raised: {e}')
                return _invoke_tool_safely(tool, args)

            result_text = await asyncio.get_running_loop().run_in_executor(None, _run)
            return [TextContent(type='text', text=result_text)]

        manager = StreamableHTTPSessionManager(app=server, stateless=True)

        async def handle_mcp(scope, receive, send):  # type: ignore[no-untyped-def]
            await manager.handle_request(scope, receive, send)

        @contextlib.asynccontextmanager
        async def lifespan(app):  # type: ignore[no-untyped-def]
            async with manager.run():
                yield

        app = Starlette(routes=[Mount('/mcp', app=handle_mcp)], lifespan=lifespan)
        config = uvicorn.Config(
            app,
            host='127.0.0.1',
            port=self._port or 0,
            log_level='warning',
            access_log=False,
        )
        self._uvicorn_server = uvicorn.Server(config)
        # uvicorn.Server.serve() flips started=True after the lifespan has
        # entered and the socket is bound. Watch for that instead of guessing
        # at a sleep delay so the caller can hand the URL to claude as soon
        # as it's actually accepting connections.
        async def _watch_ready():
            while not self._uvicorn_server.started:  # type: ignore[union-attr]
                await asyncio.sleep(0.02)
            self._ready.set()

        watcher = asyncio.create_task(_watch_ready())
        try:
            await self._uvicorn_server.serve()
        finally:
            watcher.cancel()
            with contextlib.suppress(BaseException):
                await watcher
