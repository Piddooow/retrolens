"""Model inference in a background thread: the newest frame wins, and results stay with their frame."""

import threading
import time
from typing import Any, Callable, Optional, Tuple


class InferenceWorker:
    """Runs fungsi(job) on one background thread, one job at a time, in submission order.

    A job that is still waiting when a newer one arrives is dropped (latest frame wins, queue of one).
    A finished result always comes back together with its own job, so a mask or landmarks are never
    used with another frame. Jobs are numbered in the order they are sent.
    """

    def __init__(self, fungsi: Callable[[Any], Any]):
        self._fungsi = fungsi
        self._kunci = threading.Condition()
        self._antre = None  # type: Optional[Tuple[int, Any]]
        self._selesai = None  # type: Optional[Tuple[int, Any, Any, Optional[BaseException]]]
        self._dikirim = 0
        self._sedang = 0  # number of the job being inferred right now, 0 when idle
        self._berhenti = False
        self._thread = threading.Thread(target=self._jalan, name="snap-inference", daemon=True)
        self._thread.start()

    def kirim(self, job: Any) -> int:
        """Queue a job and return its number; a job that has not started yet is replaced."""
        with self._kunci:
            self._dikirim += 1
            self._antre = (self._dikirim, job)
            self._kunci.notify_all()
            return self._dikirim

    def ambil(self, setelah: int, timeout: float) -> Optional[Tuple[int, Any, Any]]:
        """Newest finished (number, job, result) with a number above setelah, or None.

        Waits up to timeout only while an older job than the last one sent is still on its way, so the
        caller works on frame N - 1 while frame N is being inferred. Re-raises an error from fungsi.
        """
        batas = time.monotonic() + timeout
        with self._kunci:
            while not self._berhenti:
                if self._selesai is not None and self._selesai[0] > setelah:
                    break
                # Only an older job than the newest one can be worth waiting for; a queued older job is
                # always replaced, so that leaves the one being inferred.
                menunggu = setelah < self._sedang < self._dikirim
                sisa = batas - time.monotonic()
                if not menunggu or sisa <= 0.0:
                    return None
                self._kunci.wait(sisa)
            else:
                return None
            nomor, job, hasil, error = self._selesai
            self._selesai = None
        if error is not None:
            raise RuntimeError("Model inference failed: %s" % error) from error
        return nomor, job, hasil

    def _jalan(self) -> None:
        while True:
            with self._kunci:
                while self._antre is None and not self._berhenti:
                    self._kunci.wait()
                if self._berhenti:
                    return
                (nomor, job), self._antre = self._antre, None
                self._sedang = nomor
            try:
                hasil, error = self._fungsi(job), None  # type: Any, Optional[BaseException]
            except Exception as e:
                hasil, error = None, e
            with self._kunci:
                # An older result nobody took yet is replaced by this newer one.
                self._selesai = (nomor, job, hasil, error)
                self._sedang = 0
                self._kunci.notify_all()

    def tutup(self, timeout: float = 2.0) -> None:
        """Stop the thread after the job it is running; returns once it has ended or after timeout."""
        with self._kunci:
            self._berhenti = True
            self._kunci.notify_all()
        self._thread.join(timeout)
