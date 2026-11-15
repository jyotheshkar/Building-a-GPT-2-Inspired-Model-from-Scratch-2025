"""Actual Tk controller regressions; ML execution is covered by core smoke.

These tests use a deliberately small fake compute backend to isolate thread,
cancellation and window-close behavior. They skip if no desktop/Xvfb exists.
"""
import threading
import time
import tkinter as tk
from types import SimpleNamespace
import unittest

from gpt_scratch_lab.gui import GPTScratchApp


class ControlledBackend:
    def __init__(self, fail=False, ignore_cancel=False):
        self.started = threading.Event()
        self.release = threading.Event()
        self.fail = fail
        self.ignore_cancel = ignore_cancel
        self.worker_thread = None

    def demo(self):
        return "a small local training fixture.\n" * 50

    def train(self, corpus, settings, cancel, progress):
        self.worker_thread = threading.get_ident()
        self.started.set()
        if self.fail:
            raise ValueError("The selected text is too short for this context.")
        progress({"step":1,"total_steps":settings["steps"],"train_loss":2.2,
                  "val_loss":2.3,"elapsed_seconds":0.01})
        deadline = time.monotonic()+5
        while not self.release.is_set():
            if cancel.is_set() and not self.ignore_cancel:
                break
            if time.monotonic()>deadline:
                raise RuntimeError("Fixture worker was not released")
            self.release.wait(0.005)
        return SimpleNamespace(cancelled=cancel.is_set(),steps_completed=1,history=[])

    def describe(self, result):
        return "One deliberately synthetic GUI test update."

    def vocabulary(self, result):
        return tuple(sorted(set(self.demo())))


class DesktopControllerTests(unittest.TestCase):
    def setUp(self):
        try:
            self.root = tk.Tk()
            self.root.withdraw()
        except tk.TclError as exc:
            self.skipTest(f"Tk display unavailable: {exc}")
        self.app = None
        self.backends = []

    def tearDown(self):
        for backend in self.backends:
            backend.release.set()
        if self.app is not None and not self.app.closing:
            self.app.close()
        else:
            try:
                self.root.destroy()
            except tk.TclError:
                pass

    def app_with(self, backend):
        self.backends.append(backend)
        self.app = GPTScratchApp(self.root,backend=backend)
        return self.app

    def pump_until(self, predicate, timeout=3):
        deadline = time.monotonic()+timeout
        while not predicate():
            if time.monotonic()>deadline:
                self.fail("Desktop event queue did not reach the expected state")
            self.root.update()
            time.sleep(0.005)
        self.root.update()

    def test_cancel_training_restores_controls_and_keeps_partial_result(self):
        backend = ControlledBackend()
        app = self.app_with(backend)
        main_thread = threading.get_ident()
        app._start_training()
        self.pump_until(backend.started.is_set)
        self.assertNotEqual(backend.worker_thread,main_thread)
        self.assertTrue(app.busy)
        self.assertEqual(str(app.load_checkpoint_button.cget("state")),"disabled")
        self.assertEqual(str(app.train_button.cget("state")),"disabled")
        self.assertEqual(str(app.cancel_button.cget("state")),"normal")
        app._stop()
        self.pump_until(lambda:not app.busy)
        self.assertTrue(app.result.cancelled)
        self.assertEqual(str(app.generate_button.cget("state")),"normal")
        self.assertEqual(str(app.save_checkpoint_button.cget("state")),"normal")
        self.assertEqual(str(app.cancel_button.cget("state")),"disabled")

    def test_worker_error_retains_previous_model_and_reports_on_main_queue(self):
        backend = ControlledBackend(fail=True)
        app = self.app_with(backend)
        previous_model = object()
        app.result = previous_model
        errors=[]
        error_threads=[]
        def capture(error):
            errors.append(str(error))
            error_threads.append(threading.get_ident())
        app._error=capture
        app._start_training()
        self.pump_until(lambda:not app.busy)
        self.assertIs(app.result,previous_model)
        self.assertIn("too short",errors[0])
        self.assertEqual(error_threads,[threading.get_ident()])
        self.assertEqual(str(app.train_button.cget("state")),"normal")

    def test_close_cancels_without_waiting_for_active_compute(self):
        backend = ControlledBackend(ignore_cancel=True)
        app = self.app_with(backend)
        app._start_training()
        self.pump_until(backend.started.is_set)
        worker = app._worker
        before=time.monotonic()
        app.close()
        self.assertLess(time.monotonic()-before,0.5)
        self.assertTrue(app._cancel.is_set())
        self.assertTrue(worker.is_alive())
        self.assertFalse(app._close_complete)
        backend.release.set()
        worker.join(timeout=1)
        self.assertFalse(worker.is_alive())
        self.pump_until(lambda:app._close_complete)

    def test_invalid_checkpoint_history_preserves_model_and_queue_pump(self):
        app=self.app_with(ControlledBackend())
        previous_model=object()
        app.result=previous_model
        errors=[]
        app._error=lambda error:errors.append(str(error))
        malformed=SimpleNamespace(history=[{"step":None,"train_loss":None}],steps_completed=2)
        previous_poll=app._poll_id
        app.busy=True
        app._busy_kind="load"
        app._refresh_buttons()
        app._events.put(("done",("load",malformed,None)))
        self.pump_until(lambda:bool(errors))
        self.assertIs(app.result,previous_model)
        self.assertFalse(app.busy)
        self.assertIn("invalid step",errors[0])
        self.assertNotEqual(app._poll_id,previous_poll)
        self.assertIn(app._poll_id,self.root.tk.call("after","info"))


if __name__=="__main__":
    unittest.main()
