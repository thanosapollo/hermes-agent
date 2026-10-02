"""Native final receipts survive reconciliation, replacement and cancellation."""
import asyncio
import concurrent.futures
import threading
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway import delivery_ledger as ledger
from gateway.config import PlatformConfig
from gateway.platforms.base import MessageEvent
from gateway.run import GatewayRunner
from gateway.turn_context import TurnContext
from gateway.stream_consumer import GatewayStreamConsumer, StreamConsumerConfig
from plugins.platforms.discord.adapter import DiscordAdapter
from plugins.platforms.telegram.adapter import TelegramAdapter

FINAL = ("| " + "H" * 55 + " | " + "J" * 55 + " |\n| --- | --- |\n" + "| a | b |\n" * 40).strip()
NEXT = "independent complete reply"


class OfflineMessage:
    def __init__(self, message_id, content):
        self.id, self.content = message_id, content

    async def edit(self, **kwargs):
        self.content = kwargs["content"]

    def to_reference(self, **kwargs):
        return SimpleNamespace(message_id=self.id)


class OfflineChannel:
    id = 1
    guild = None

    def __init__(self):
        self.visible = {}
        self.calls = []
        self.reject_continuations = False
        self.ready = asyncio.Event()

    def get_partial_message(self, message_id):
        return self.visible[message_id]

    async def send(self, **kwargs):
        self.calls.append(kwargs)
        if self.reject_continuations:
            raise RuntimeError("definite offline continuation rejection")
        message_id = 101 + len(self.visible)
        message = OfflineMessage(message_id, kwargs["content"])
        self.visible[message_id] = message
        self.ready.set()
        return message


@pytest.fixture
def native(tmp_path, monkeypatch):
    # Exercise real configuration/enablement and SQLite, not forced-enabled helpers.
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "config.yaml").write_text("gateway:\n  delivery_ledger: true\n")
    adapter = DiscordAdapter(PlatformConfig(enabled=True, token="offline", typing_indicator=False))
    adapter._client = object()
    channel = OfflineChannel()
    adapter._resolve_channel = AsyncMock(return_value=channel)
    adapter._record_response_async = AsyncMock(side_effect=lambda *a, **k: a[1])
    adapter._media_delivery_scope = lambda _: nullcontext()
    adapter._flush_text_debounce_now = AsyncMock()
    adapter._stop_typing_refresh = AsyncMock()
    adapter._fire_post_delivery_callback = AsyncMock()
    adapter._finish_session_task = lambda *a: None
    source = adapter.build_source(chat_id="1", user_id="1", thread_id="42")
    runner = GatewayRunner.__new__(GatewayRunner)
    runner._voice_mode = {}
    runner._delivery_adapter_for = lambda _: adapter
    runner._intake_adapter_for = lambda _: adapter
    runner._pop_post_delivery_callback = lambda *a: None
    runner._deliver_media_from_response = AsyncMock()
    runner._should_send_voice_reply = lambda *a, **k: False
    return adapter, channel, source, runner


def rows():
    with ledger._connect() as conn:
        return conn.execute(
            "SELECT obligation_id, state, content, thread_id FROM delivery_obligations ORDER BY content"
        ).fetchall()


async def preview(native):
    adapter, channel, source, runner = native
    stream = GatewayStreamConsumer(adapter, "1", StreamConsumerConfig(
        cursor="", edit_interval=0, buffer_threshold=1), metadata={"thread_id": "42"})
    task = asyncio.create_task(stream.run())
    try:
        stream.on_delta("initial accepted preview")
        await asyncio.wait_for(channel.ready.wait(), 5)
        stream.finish(final_text="initial accepted preview")
        await asyncio.wait_for(task, 5)
        assert stream.message_id == "101"
        return stream
    finally:
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)


async def queued_reconcile(native):
    adapter, channel, source, runner = native
    stream = await preview(native)
    channel.reject_continuations = True
    ctx = TurnContext(source=source, session_key="receipt-session", session_id="offline",
                      inbound_message_id="first-inbound", stream_consumer_holder=[stream])
    result = {"final_response": FINAL, "messages": []}
    ctx.result_holder[0] = result
    await runner._run_agent_deliver_first_response(ctx, adapter, result, result, None)
    return ctx, result



import sys
import types
from tests.gateway.test_queued_followup_processing_hooks import _make_runner
from tests.gateway.test_telegram_split_send_flood import _adapter, _three_chunk_text, _FloodError

async def _provider_settlement(native, monkeypatch, cancel_count, boundary):
    adapter, channel, source, oldrunner = native
    runner = _make_runner(adapter)
    runner._draining = False
    runner._profile_scope_for_source = lambda _: nullcontext()
    runner._get_proxy_url = lambda: None
    runner._pop_post_delivery_callback = lambda *a: None
    runner._is_goal_continuation_event = lambda _: False
    runner._prepare_profile_scoped_inbound_message_text = AsyncMock(side_effect=lambda event, **k: event.text)
    runner._pinned_channel_inputs = lambda key, prompt, source, **k: (prompt, source)
    runner._persist_prompt_pins = AsyncMock()
    runner._refresh_agent_cache_message_count = AsyncMock()
    runner._deliver_media_from_response = AsyncMock()
    fake = types.ModuleType('run_agent'); fake.AIAgent=object
    monkeypatch.setitem(sys.modules,'run_agent',fake)
    # The only replaced provider seam is synchronous conversation work.
    # Native context construction, scheduling, draining, recursion and finally remain intact.
    contexts=[]; provider_calls=[]; cleanup=[]
    original_build=runner._run_agent_build_turn_context
    loop=asyncio.get_running_loop()
    def build(*args, **kwargs):
        ctx,tr,ca=original_build(*args,**kwargs)
        contexts.append(ctx)
        first=len(contexts)==1
        if first and boundary!='complete':
            stream=GatewayStreamConsumer(adapter,'1',StreamConsumerConfig(
                cursor='', edit_interval=0,buffer_threshold=1),metadata={'thread_id':'42'})
            ctx.stream_consumer_holder[0]=stream
        def offline_provider():
            provider_calls.append(ctx.message)
            ctx.agent_holder[0]=SimpleNamespace(tools=[])
            if first and boundary!='complete':
                stream.on_delta('initial accepted preview')
                asyncio.run_coroutine_threadsafe(channel.ready.wait(),loop).result(5)
                channel.reject_continuations=True
                stream.finish(final_text='initial accepted preview')
            result={'final_response':FINAL if first and boundary!='complete' else NEXT,'messages':[],
                    'response_transformed':first and boundary!='complete'}
            ctx.result_holder[0]=result
            return result
        tr.run_sync=offline_provider
        return ctx,tr,ca
    runner._run_agent_build_turn_context=build
    real_cleanup=runner._run_agent_cleanup_turn_tasks
    async def observed_cleanup(ctx,**kwargs):
        cleanup.append(('enter',ctx.inbound_message_id))
        try: return await real_cleanup(ctx,**kwargs)
        finally: cleanup.append(('exit',ctx.inbound_message_id))
    runner._run_agent_cleanup_turn_tasks=observed_cleanup
    adapter._pending_messages['provider-session']=MessageEvent(text='follow-up',source=source,message_id='second-inbound')
    # Native final partial comes from overflow editing, not an injected receipt.
    event=MessageEvent(text='first request',source=source,message_id='first-inbound')
    outcomes=[]
    async def hook(name,event,outcome=None):
        if name=='on_processing_complete': outcomes.append((event.message_id,outcome.value))
    adapter._run_processing_hook=hook
    async def handler(event):
        result=await runner._run_agent(message=event.text,context_prompt='',history=[],source=source,
            session_id='offline-provider',session_key='provider-session',inbound_message_id=event.message_id)
        # No-cancel positive: second inbound remains independent.
        channel.reject_continuations=False
        if result.get('queued_delivery_incomplete'): event._queued_delivery_incomplete=True
        return None
    adapter._message_handler=handler
    executor=concurrent.futures.ThreadPoolExecutor(max_workers=1)
    old_executor=loop._default_executor
    loop.set_default_executor(executor)
    release=threading.Event(); blocked=threading.Event(); admitted=asyncio.Event()
    futures=[]; children=[]
    actual_submit=executor.submit
    def submit(fn,*a,**k):
        f=actual_submit(fn,*a,**k); futures.append(f); return f
    executor.submit=submit
    target={'enabled':ledger.ledger_enabled,'record':ledger.record_obligation,'complete':ledger.mark_delivered}[boundary]
    actual_to_thread=asyncio.to_thread
    blocker=None
    def occupy():
        blocked.set(); assert release.wait(15)
    async def observed_to_thread(func,*a,**k):
        nonlocal blocker
        # Initial sender enabled/record does not run for partial reconciliation.
        if func is target and blocker is None:
            blocker=actual_submit(occupy)
            while not blocked.is_set(): await asyncio.sleep(0)
            children.append(asyncio.current_task())
            admitted.set()
        return await actual_to_thread(func,*a,**k)
    monkeypatch.setattr(asyncio,'to_thread',observed_to_thread)
    task=asyncio.create_task(adapter._process_message_background(event,'provider-session'))
    observed=0
    try:
        await asyncio.wait_for(admitted.wait(),5)
        assert futures[-1].running() is False and futures[-1].done() is False
        assert channel.visible
        assert len(contexts)==1 and provider_calls==['first request']
        for i in range(cancel_count):
            previous=task._fut_waiter
            task.cancel('provider-cancel-'+str(i))
            for _ in range(100):
                await asyncio.sleep(0)
                if task.done() or (task._fut_waiter is not None and task._fut_waiter is not previous and not task._fut_waiter.done()): break
            assert not task.done() and task._fut_waiter is not previous
            assert not children[0].done() and not futures[-1].cancelled()
            observed+=1
        assert observed==cancel_count
        release.set()
        if cancel_count:
            with pytest.raises(asyncio.CancelledError,match='provider-cancel-0'):
                await asyncio.wait_for(task,8)
            assert provider_calls==['first request']
            assert cleanup==[('enter','first-inbound'),('exit','first-inbound')]
            assert outcomes==[('first-inbound','failure')]
        else:
            await asyncio.wait_for(task,8)
            assert provider_calls==['first request','follow-up']
        expected_content=NEXT if boundary=='complete' else FINAL
        expected_state='delivered' if boundary=='complete' else 'incomplete'
        oid=ledger.compute_obligation_id('provider-session','first-inbound',expected_content)
        assert (oid,expected_state,expected_content,'42') in rows()
        assert ledger.pending_retries(now=ledger.time.time()+1000)==[]
        assert all(c.done() for c in children)
        assert all(f.done() and not f.cancelled() for f in futures)
        assert not runner._running_agents
    finally:
        release.set()
        await asyncio.gather(task,return_exceptions=True)
        loop.set_default_executor(old_executor or concurrent.futures.ThreadPoolExecutor())
        executor.shutdown(wait=True)

@pytest.mark.asyncio
@pytest.mark.parametrize("scenario", ['ordinary-complete', 'ordinary-partial', 'queued', 'owner', 'transformed', 'transformed-split', 'telegram', 'exact-admission', 'recursive', 'storage-error', 'disabled', 'inline-tail', 'retention', 'nonterminal', 'over-cap-tail'])
async def test_post_stream_receipt_conservation(native, monkeypatch, scenario):
    """Acquired reconciliation debt survives replacement; ordinary recovery is unchanged."""
    if scenario.startswith("ordinary-"):
        partial = scenario == "ordinary-partial"
        adapter, channel, source, runner = native
        stream = await preview(native)
        channel.reject_continuations = partial
        ctx = TurnContext(source=source, session_key="receipt-session", inbound_message_id="first-inbound",
                          stream_consumer_holder=[stream])
        response = {"final_response": FINAL, "response_transformed": True}
        outcomes = []
        async def hook(name, event, outcome=None):
            if name == "on_processing_complete":
                outcomes.append(outcome.value)
        async def handler(event):
            await runner._run_agent_mark_streamed_delivery(response, ctx)
            return await runner._hmwa_deliver_turn_response(
                event, source, SimpleNamespace(session_id="offline"), ctx.session_key, 1,
                response, [], FINAL, "", False)
        adapter._message_handler = handler
        adapter._run_processing_hook = hook
        event = MessageEvent(text="first request", source=source, message_id="first-inbound")
        await adapter._process_message_background(event, ctx.session_key)
        if partial:
            assert len(channel.visible) == 1
            assert outcomes == ["failure"]
            assert rows() == [(ledger.compute_obligation_id(ctx.session_key, "first-inbound", FINAL),
                               "incomplete", FINAL, "42")]
            assert ledger.pending_retries(now=ledger.time.time() + 1000) == []
        else:
            assert len(channel.visible) == 2
            assert outcomes == ["success"]
            assert rows() == []  # complete streamed final remains unledgered
    elif scenario == 'queued':
        adapter, channel, source, runner = native
        ctx, result = await queued_reconcile(native)
        assert not result.get("already_sent")
        assert adapter._is_partial_delivery(result["delivery_incomplete"])
        assert rows() == [(ledger.compute_obligation_id(ctx.session_key, "first-inbound", FINAL),
                           "incomplete", FINAL, "42")]
        calls = list(channel.calls)
        event = MessageEvent(text="request", source=source, message_id="first-inbound")
        assert await runner._hmwa_deliver_turn_response(
            event, source, SimpleNamespace(session_id="offline"), ctx.session_key, 1,
            result, [], FINAL, "", False) is None
        assert channel.calls == calls
        # A new authorized obligation must still succeed; incomplete does not poison the adapter.
        channel.reject_continuations = False
        next_event = MessageEvent(text="next", source=source, message_id="second-inbound")
        sent, _ = await adapter.send_final_ledgered(next_event, ctx.session_key, NEXT, {}, reply_to=None)
        assert sent.success
        assert {(state, content) for _, state, content, _ in rows()} == {
            ("incomplete", FINAL), ("delivered", NEXT)}
        runner._deliver_media_from_response.assert_not_awaited()
    elif scenario == 'owner':
        adapter, channel, source, runner = native
        adapter._owner_profile = "preview-owner"
        stream = await preview(native)
        channel.reject_continuations = True
        replacement = DiscordAdapter(PlatformConfig(enabled=True, token="offline"))
        replacement._owner_profile = "replacement-owner"
        replacement._resolve_channel = AsyncMock(side_effect=AssertionError("must not replay via replacement"))
        ctx = TurnContext(source=source, session_key="receipt-session", session_id="offline",
                          inbound_message_id="first-inbound", stream_consumer_holder=[stream])
        result = {"final_response": FINAL, "messages": []}
        ctx.result_holder[0] = result
        await runner._run_agent_deliver_first_response(ctx, replacement, result, result, None)
        assert adapter._is_partial_delivery(result["delivery_incomplete"])
        with ledger._connect() as conn:
            assert conn.execute("SELECT state, adapter_profile FROM delivery_obligations").fetchall() == [
                ("incomplete", "preview-owner")]
        replacement._resolve_channel.assert_not_awaited()
    elif scenario == 'transformed':
        adapter, channel, source, runner = native
        stream = await preview(native)
        channel.reject_continuations = True
        ctx = TurnContext(source=source, session_key="receipt-session", inbound_message_id="first-inbound",
                          stream_consumer_holder=[stream])
        result = {"final_response": FINAL, "response_transformed": True}
        await runner._run_agent_mark_streamed_delivery(result, ctx)
        assert not result.get("already_sent")
        assert adapter._is_partial_delivery(result["delivery_incomplete"])
        assert rows()[0][1:] == ("incomplete", FINAL, "42")
    elif scenario == 'transformed-split':
        adapter, channel, source, runner = native
        stream = await preview(native)
        # A split stream's last message is only a tail, never a whole-turn edit handle.
        stream._turn_split_delivery = True
        before = channel.visible[101].content
        ctx = TurnContext(source=source, session_key="receipt-session", inbound_message_id="first-inbound",
                          stream_consumer_holder=[stream])
        result = {"final_response": FINAL, "response_transformed": True}
        await runner._run_agent_mark_streamed_delivery(result, ctx)
        assert not result.get("already_sent") and result.get("delivery_incomplete") is None
        assert channel.visible[101].content == before and len(channel.visible) == 1
        assert rows() == []
    elif scenario == 'telegram':
        _, _, _, runner = native
        adapter = TelegramAdapter(PlatformConfig(enabled=True, token="offline"))
        object.__setattr__(adapter, "MAX_MESSAGE_LENGTH", 160)
        adapter._bot = MagicMock()
        accepted = asyncio.Event()
        sends = []
        async def send_message(**kwargs):
            sends.append(kwargs)
            if len(sends) > 1:
                raise RuntimeError("definite offline continuation rejection")
            accepted.set()
            return SimpleNamespace(message_id=101)
        adapter._bot.send_message = send_message
        adapter._bot.edit_message_text = AsyncMock(return_value=True)
        source = adapter.build_source(chat_id="1", user_id="1", thread_id="42")
        stream = GatewayStreamConsumer(adapter, "1", StreamConsumerConfig(
            cursor="", edit_interval=0, buffer_threshold=1), metadata={"thread_id": "42"})
        task = asyncio.create_task(stream.run())
        try:
            stream.on_delta("accepted preview")
            await asyncio.wait_for(accepted.wait(), 5)
            stream.finish(final_text="accepted preview")
            await asyncio.wait_for(task, 5)
            final = ("word " * 120).strip()
            result = {"final_response": final}
            ctx = TurnContext(source=source, session_key="telegram-receipt", session_id="offline",
                              inbound_message_id="first-inbound", stream_consumer_holder=[stream])
            ctx.result_holder[0] = result
            await runner._run_agent_deliver_first_response(ctx, adapter, result, result, None)
            assert not result.get("already_sent")
            receipt = result["delivery_incomplete"]
            assert not receipt.success and adapter._is_partial_delivery(receipt)
            assert receipt.raw_response["delivered_chunks"] == 1
            assert receipt.raw_response["delivered_prefix"]
            assert not receipt.raw_response.get("undelivered_chunks")
            assert rows() == [(ledger.compute_obligation_id("telegram-receipt", "first-inbound", final),
                               "incomplete", final, "42")]
            assert ledger.pending_retries(now=ledger.time.time() + 1000) == []
            runner._deliver_media_from_response.assert_not_awaited()
        finally:
            if not task.done():
                task.cancel()
            await asyncio.gather(task, return_exceptions=True)
    elif scenario == 'exact-admission':
        adapter, channel, source, runner = native
        ctx, result = await queued_reconcile(native)
        oid = rows()[0][0]
        ledger.mark_delivered(oid)
        ledger.mark_failed(oid, "late failure")
        ledger.mark_attempting(oid)
        ledger.record_obligation(obligation_id=oid, session_key=ctx.session_key, platform="discord",
                                 chat_id="1", thread_id="42", content=FINAL)
        ledger.record_crash_left_reply(obligation_id=oid, session_key=ctx.session_key, platform="discord",
                                       chat_id="1", thread_id="42", content=FINAL, since=0)
        assert rows()[0][1:] == ("incomplete", FINAL, "42")
        with ledger._connect() as conn:
            conn.execute("UPDATE delivery_obligations SET owner_pid=NULL, owner_started_at=NULL")
            conn.commit()
        assert ledger.sweep_recoverable() == []
        before = list(channel.calls)
        event = MessageEvent(text="request", source=source, message_id="first-inbound")
        result, _ = await adapter.send_final_ledgered(event, ctx.session_key, FINAL, {}, reply_to=None)
        assert adapter._is_partial_delivery(result)
        assert channel.calls == before  # exact-id readmission cannot replay the accepted head
    elif scenario == 'recursive':
        adapter, channel, source, runner = native
        stream = await preview(native)
        channel.reject_continuations = True
        ctx = TurnContext(source=source, session_key="receipt-session", session_id="offline",
                          inbound_message_id="first-inbound", stream_consumer_holder=[stream])
        first = {"final_response": FINAL, "messages": []}
        ctx.result_holder[0] = first
        pending = MessageEvent(text="next request", source=source, message_id="second-inbound")
        runner._is_goal_continuation_event = lambda event: False
        runner._session_key_for_source = lambda source: ctx.session_key
        runner._prepare_profile_scoped_inbound_message_text = AsyncMock(return_value=pending.text)
        runner._pinned_channel_inputs = lambda key, prompt, source, **kwargs: (prompt, source)
        runner._persist_prompt_pins = AsyncMock()
        runner._refresh_agent_cache_message_count = AsyncMock()
        outcomes = []
        async def hook(name, event, outcome=None):
            if name == "on_processing_complete":
                outcomes.append((event.message_id, outcome.value))
        adapter._run_processing_hook = hook
        async def next_provider(**kwargs):
            assert kwargs["inbound_message_id"] == pending.message_id
            assert rows()[0][1:] == ("incomplete", FINAL, "42")
            channel.reject_continuations = False
            return {"final_response": NEXT, "messages": []}
        runner._run_agent = next_provider
        original = MessageEvent(text="first request", source=source, message_id="first-inbound")
        async def handler(event):
            merged = await runner._run_agent_queued_followup(
                ctx, adapter, pending.text, pending, first, first, None)
            assert merged["final_response"] == NEXT
            assert merged.get("delivery_incomplete") is None
            assert merged["queued_delivery_incomplete"]
            event.ledger_message_id = merged["queued_terminal_inbound_id"]
            return await runner._hmwa_deliver_turn_response(
                event, source, SimpleNamespace(session_id="offline"), ctx.session_key, 1,
                merged, [], NEXT, "", False)
        adapter._message_handler = handler
        await adapter._process_message_background(original, ctx.session_key)
        assert outcomes == [("second-inbound", "success"), ("first-inbound", "failure")]
        assert {(state, content) for _, state, content, _ in rows()} == {
            ("incomplete", FINAL), ("delivered", NEXT)}
        assert len(channel.visible) == 2
    elif scenario == 'storage-error':
        disabled = False
        adapter, channel, source, runner = native
        if disabled:
            from pathlib import Path
            import os
            (Path(os.environ["HERMES_HOME"]) / "config.yaml").write_text(
                "gateway:\n  delivery_ledger: false\n")
        else:
            def fail(**kwargs):
                raise OSError("offline storage unavailable")
            monkeypatch.setattr(ledger, "record_obligation", fail)
        ctx, result = await queued_reconcile(native)
        assert not result.get("already_sent")
        assert adapter._is_partial_delivery(result["delivery_incomplete"])
        assert rows() == []
        assert len(channel.visible) == 1
    elif scenario == 'disabled':
        disabled = True
        adapter, channel, source, runner = native
        if disabled:
            from pathlib import Path
            import os
            (Path(os.environ["HERMES_HOME"]) / "config.yaml").write_text(
                "gateway:\n  delivery_ledger: false\n")
        else:
            def fail(**kwargs):
                raise OSError("offline storage unavailable")
            monkeypatch.setattr(ledger, "record_obligation", fail)
        ctx, result = await queued_reconcile(native)
        assert not result.get("already_sent")
        assert adapter._is_partial_delivery(result["delivery_incomplete"])
        assert rows() == []
        assert len(channel.visible) == 1
    elif scenario == 'inline-tail':
        sends=[]; attempts=0
        async def send(text,**kw):
            nonlocal attempts
            attempts+=1
            if attempts==2: raise _FloodError(7.0)
            from types import SimpleNamespace
            sends.append(text)
            return SimpleNamespace(message_id=1000+attempts)
        adapter=_adapter(AsyncMock(side_effect=send))
        async def penalty_passed(delay): adapter._telegram_send_cooldown_until.clear()
        monkeypatch.setattr(asyncio,'sleep',penalty_passed)
        source=adapter.build_source(chat_id='4242',user_id='1',thread_id='42')
        event=MessageEvent(text='reply please',source=source,message_id='tail-input')
        content=_three_chunk_text()
        from plugins.platforms.telegram.adapter import utf16_len, _separate_chunk_indicator_from_fence
        import re
        expected=adapter.truncate_message(adapter.format_message(content),adapter.MAX_MESSAGE_LENGTH,len_fn=utf16_len)
        expected=[_separate_chunk_indicator_from_fence(re.sub(r" \((\d+)/(\d+)\)$", r" \\(\1/\2\\)", c)) for c in expected]
        result,_=await adapter.send_final_ledgered(event,'tail-session',content,{'thread_id':'42'},reply_to=None)
        assert result.success and not adapter._is_partial_delivery(result)
        assert sends==expected and attempts==len(expected)+1
        assert rows()==[(ledger.compute_obligation_id('tail-session','tail-input',content),'delivered',content,'42')]
    elif scenario == 'retention':
        ctx,result=await queued_reconcile(native)
        oid=rows()[0][0]
        def full():
            with ledger._connect() as c: return c.execute('SELECT * FROM delivery_obligations WHERE obligation_id=?',(oid,)).fetchone()
        original=full()
        for mark in (ledger.mark_delivered,ledger.mark_failed,ledger.mark_attempting): mark(oid)
        ledger._update_state(oid, "incomplete")
        ledger.record_obligation(obligation_id=oid,session_key='foreign',platform='telegram',chat_id='foreign',thread_id='other',content='replacement',adapter_profile='other')
        assert full()==original
        with ledger._transaction() as c: ledger._prune_unlocked(c,ledger.time.time()+ledger._RETENTION_SECONDS+1)
        assert full() is None
        # Terminal retention is bounded, not an eternal idempotence tombstone.
        assert ledger.pending_retries()==[]
    elif scenario == 'nonterminal':
        kw=dict(obligation_id='replaceable',session_key='s',platform='discord',chat_id='1',thread_id='42',content='old')
        ledger.record_obligation(**kw)
        ledger.mark_failed('replaceable','old error')
        kw.update(content='new',chat_id='2',thread_id='43',adapter_profile='successor')
        ledger.record_obligation(**kw)
        with ledger._connect() as c:
            assert c.execute("SELECT state,attempts,last_error,content,chat_id,thread_id,adapter_profile FROM delivery_obligations WHERE obligation_id='replaceable'").fetchone()==('pending',0,None,'new','2','43','successor')
    else:
        adapter = TelegramAdapter(PlatformConfig(enabled=True, token="offline"))
        adapter._rich_send_disabled = True
        adapter._bot = MagicMock()
        adapter._retrigger_typing = AsyncMock()
        calls, accepted = [], []
        async def send_message(text, **kwargs):
            calls.append(text)
            if len(calls) == 2:
                raise _FloodError(120.0)
            accepted.append(text)
            return SimpleNamespace(message_id=1000 + len(calls))
        adapter._bot.send_message = send_message
        scheduler = MagicMock()
        source = adapter.build_source(chat_id="4242", user_id="1")
        adapter.gateway_runner = SimpleNamespace(_schedule_flood_redelivery=scheduler)
        event = MessageEvent(text="ordinary request", source=source, message_id="inbound")
        content = "\n".join(" ".join(f"w{i * 20 + j}" for j in range(20)) for i in range(80))
        result, owner = await adapter.send_final_ledgered(event, "known-tail-session", content, {}, reply_to=None)
        assert owner is adapter and adapter._is_partial_delivery(result)
        assert not result.success and len(result.raw_response["undelivered_chunks"]) == 2
        assert len(accepted) == 1 and len(calls) == 2
        assert rows() == [(ledger.compute_obligation_id("known-tail-session", "inbound", content), "failed", content, None)]
        scheduler.assert_called_once_with(source.platform, profile=None)
        targets = ledger.pending_retries(now=ledger.time.time() + 1000)
        assert len(targets) == 1
        assert targets[0]["platform"] == "telegram" and targets[0]["profile"] == "default"
        assert targets[0]["not_before"] > ledger.time.time()
        # This explicit diagnostic continuation is NOT the existing ledger's delayed replay.
        adapter._telegram_send_cooldown_until.clear()
        resumed = await adapter._resume_partial_send("4242", result, reply_to=None, metadata={})
        assert resumed.success and len(accepted) == 3
        assert len(set(accepted)) == 3
        assert accepted == [calls[0], calls[2], calls[3]]


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel_count", [0, 1, 3])
@pytest.mark.parametrize("caller,boundary", [("queued", "enabled"), ("queued", "record"), ("provider", "enabled"), ("provider", "record")])
async def test_reconciliation_admission_survives_cancellation(native, monkeypatch, cancel_count, caller, boundary):
    """Own acquired partial admission through real executor and provider cleanup."""
    if caller == "provider":
        await _provider_settlement(native, monkeypatch, cancel_count, boundary)
        return
    adapter, channel, source, runner = native
    loop = asyncio.get_running_loop()
    old_executor = loop._default_executor
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    loop.set_default_executor(executor)
    release = threading.Event()
    blocked = threading.Event()
    def occupy():
        blocked.set()
        assert release.wait(10), "driver failed to release owned executor"
    blocker = None
    queued = asyncio.Event()
    actual_to_thread = asyncio.to_thread
    target = ledger.ledger_enabled if boundary == "enabled" else ledger.record_obligation
    async def observed_to_thread(func, *args, **kwargs):
        nonlocal blocker
        if func is target and blocker is None:
            blocker = executor.submit(occupy)
            while not blocked.is_set():
                await asyncio.sleep(0)
            queued.set()
        return await actual_to_thread(func, *args, **kwargs)
    monkeypatch.setattr(asyncio, "to_thread", observed_to_thread)
    task = asyncio.create_task(queued_reconcile(native))
    try:
        await asyncio.wait_for(queued.wait(), 5)
        for i in range(cancel_count):
            previous_waiter = task._fut_waiter
            task.cancel("receipt-cancel-" + str(i))
            # Observe each cancellation at a live wait; not merely scheduled cancel calls.
            for _ in range(100):
                await asyncio.sleep(0)
                if task.done() or (task._fut_waiter is not None
                                   and task._fut_waiter is not previous_waiter
                                   and not task._fut_waiter.done()):
                    break
            assert not task.done(), "caller must retain settlement while admission cannot start"
            assert task._fut_waiter is not previous_waiter
        release.set()
        if cancel_count:
            with pytest.raises(asyncio.CancelledError, match="receipt-cancel-0"):
                await asyncio.wait_for(task, 5)
        else:
            await asyncio.wait_for(task, 5)
        assert rows() == [(ledger.compute_obligation_id("receipt-session", "first-inbound", FINAL),
                           "incomplete", FINAL, "42")]
        assert len(channel.visible) == 1
        assert ledger.pending_retries(now=ledger.time.time() + 1000) == []
    finally:
        release.set()
        await asyncio.gather(task, return_exceptions=True)
        loop.set_default_executor(old_executor or concurrent.futures.ThreadPoolExecutor())
        executor.shutdown(wait=True)

@pytest.mark.asyncio
@pytest.mark.parametrize("scenario,cancel_count", [("ordinary", 0), ("ordinary", 1), ("ordinary", 3), ("queued", 0), ("queued", 1), ("queued", 3), ("provider", 0), ("provider", 1), ("provider", 3), ("child-success", 0), ("child-error", 0), ("child-cancel", 0)])
async def test_acquired_complete_settlement_survives_cancellation(native, monkeypatch, scenario, cancel_count):
    """Complete ACK settles before cancellation; terminal child controls cannot spin."""
    if scenario == "provider":
        await _provider_settlement(native, monkeypatch, cancel_count, "complete")
    elif scenario.startswith("child-"):
        kind = scenario.removeprefix("child-")
        adapter, channel, source, runner = native
        async def child():
            if kind == "error":
                raise RuntimeError("child failed")
            if kind == "cancel":
                raise asyncio.CancelledError("child self-cancelled")
            return "settled"
        if kind == "success":
            assert await asyncio.wait_for(adapter._await_receipt_settlement(child()), 1) == "settled"
        else:
            expected = RuntimeError if kind == "error" else asyncio.CancelledError
            with pytest.raises(expected, match="child"):
                await asyncio.wait_for(adapter._await_receipt_settlement(child()), 1)
    else:
        queued_lane = scenario == "queued"
        adapter, channel, source, runner = native
        loop = asyncio.get_running_loop()
        old_executor = loop._default_executor
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        loop.set_default_executor(executor)
        release = threading.Event()
        blocked = threading.Event()
        entered = asyncio.Event()
        child_waiters = []
        def occupy():
            blocked.set()
            assert release.wait(10)
        actual_to_thread = asyncio.to_thread
        blocker = None
        async def observed_to_thread(func, *args, **kwargs):
            nonlocal blocker
            if func is ledger.mark_delivered:
                assert channel.visible  # the native ACK has already been acquired
                blocker = executor.submit(occupy)
                while not blocked.is_set():
                    await asyncio.sleep(0)
                child_waiters.append(asyncio.current_task())
                entered.set()
            return await actual_to_thread(func, *args, **kwargs)
        monkeypatch.setattr(asyncio, "to_thread", observed_to_thread)
        event = MessageEvent(text="request", source=source, message_id="first-inbound")
        if queued_lane:
            operation = runner._send_queued_final_text(
                adapter, source, NEXT, {}, None, "receipt-session", "first-inbound")
        else:
            operation = adapter.send_final_ledgered(event, "receipt-session", NEXT, {}, reply_to=None)
        task = asyncio.create_task(operation)
        try:
            await asyncio.wait_for(entered.wait(), 5)
            for i in range(cancel_count):
                previous_waiter = task._fut_waiter
                task.cancel("complete-cancel-" + str(i))
                for _ in range(100):
                    await asyncio.sleep(0)
                    if task.done() or (task._fut_waiter is not None
                                       and task._fut_waiter is not previous_waiter
                                       and not task._fut_waiter.done()):
                        break
                assert not task.done(), "acquired native receipt must settle before caller exits"
                assert task._fut_waiter is not previous_waiter
            release.set()
            if cancel_count:
                with pytest.raises(asyncio.CancelledError, match="complete-cancel-0"):
                    await asyncio.wait_for(task, 5)
            else:
                await asyncio.wait_for(task, 5)
            assert rows()[0][1:] == ("delivered", NEXT, "42")
            assert all(waiter.done() for waiter in child_waiters)
            assert len(channel.visible) == 1
        finally:
            release.set()
            await asyncio.gather(task, return_exceptions=True)
            loop.set_default_executor(old_executor or concurrent.futures.ThreadPoolExecutor())
            executor.shutdown(wait=True)
