import asyncio
import unittest

from analysis_admission import AnalysisAdmission, AdmissionFull, AdmissionTimeout, finish_thread_on_cancel


class AdmissionTests(unittest.IsolatedAsyncioTestCase):
    async def test_capacity_timeout_and_recovery(self):
        admission = AnalysisAdmission(1, 1, .02)
        async with admission.reserve(), admission.worker():
            async with admission.reserve():
                with self.assertRaises(AdmissionFull):
                    async with admission.reserve():
                        self.fail('over capacity')
                with self.assertRaises(AdmissionTimeout):
                    async with admission.worker():
                        self.fail('two active workers')
            self.assertEqual(admission.active, 1)
        async with admission.reserve(), admission.worker():
            self.assertEqual(admission.active, 1)
        self.assertEqual(admission.admitted, 0)
        self.assertEqual(admission.waiting, 0)
        self.assertEqual(admission.timed_out, 1)
        self.assertEqual(admission.rejected, 1)

    async def test_cancellation_keeps_slot_until_work_finishes(self):
        admission = AnalysisAdmission(1, 0, 1)
        entered = asyncio.Event()
        finish = asyncio.Event()

        async def work():
            entered.set()
            await finish.wait()

        async def request():
            async with admission.reserve(), admission.worker():
                await finish_thread_on_cancel(work())

        task = asyncio.create_task(request())
        await entered.wait()
        task.cancel()
        await asyncio.sleep(0)
        self.assertEqual(admission.active, 1)
        with self.assertRaises(AdmissionFull):
            async with admission.reserve():
                self.fail('released too early')
        task.cancel()
        finish.set()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertEqual(admission.active, 0)
        self.assertEqual(admission.admitted, 0)

    async def test_cancelled_waiter_does_not_leak(self):
        admission = AnalysisAdmission(1, 1, 10)
        async def waiter():
            async with admission.reserve(), admission.worker():
                pass
        async with admission.reserve(), admission.worker():
            task = asyncio.create_task(waiter())
            await asyncio.sleep(0)
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
            self.assertEqual(admission.waiting, 0)
            self.assertEqual(admission.admitted, 1)


if __name__ == '__main__':
    unittest.main()
