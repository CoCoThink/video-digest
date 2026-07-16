import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import httpx

from server import main, store


class FakeScheduler:
    def __init__(self):
        self.items = []

    def snapshot(self):
        return {"queue_depth": len(self.items)}

    def enqueue(self, task_id):
        if task_id in self.items:
            return True
        self.items.append(task_id)
        return True

    def position(self, task_id):
        try:
            return self.items.index(task_id) + 1
        except ValueError:
            return None


class ChunkedUploadTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.old_data_dir = main.settings.data_dir
        self.old_max_size = main.settings.max_size_bytes
        self.old_chunk_size = main.settings.upload_chunk_size_bytes
        self.old_capacity = main.settings.queue_capacity
        main.settings.data_dir = Path(self.tmp.name)
        main.settings.max_size_bytes = 20
        main.settings.upload_chunk_size_bytes = 4
        main.settings.queue_capacity = 10
        store.ensure_dirs(main.settings.data_dir)
        main.scheduler = FakeScheduler()
        main.upload_locks.clear()
        self.client = httpx.AsyncClient(
            transport=httpx.ASGITransport(app=main.app), base_url="http://test")

    async def asyncTearDown(self):
        await self.client.aclose()
        main.settings.data_dir = self.old_data_dir
        main.settings.max_size_bytes = self.old_max_size
        main.settings.upload_chunk_size_bytes = self.old_chunk_size
        main.settings.queue_capacity = self.old_capacity
        main.upload_locks.clear()
        self.tmp.cleanup()

    async def init_upload(self, size=10):
        response = await self.client.post(
            "/api/uploads", json={"filename": "sample.mka", "size": size})
        self.assertEqual(response.status_code, 201)
        return response.json()["upload_id"]

    async def put(self, upload_id, offset, content):
        return await self.client.put(
            f"/api/uploads/{upload_id}", content=content,
            headers={"Upload-Offset": str(offset),
                     "Content-Type": "application/octet-stream"})

    async def test_chunks_are_joined_and_complete_is_idempotent(self):
        upload_id = await self.init_upload()
        first = await self.put(upload_id, 0, b"abcd")
        second = await self.put(upload_id, 4, b"efgh")
        third = await self.put(upload_id, 8, b"ij")
        self.assertEqual(first.json()["received_size"], 4)
        self.assertEqual(second.json()["received_size"], 8)
        self.assertEqual(third.json()["received_size"], 10)

        status = await self.client.get(f"/api/uploads/{upload_id}")
        self.assertEqual(status.json()["received_size"], 10)

        async def run_inline(func, *args):
            return func(*args)

        with (
            patch.object(main, "probe_audio", return_value=12.5),
            patch("fastapi.concurrency.run_in_threadpool", side_effect=run_inline),
        ):
            completed = await self.client.post(f"/api/uploads/{upload_id}/complete")
            repeated = await self.client.post(f"/api/uploads/{upload_id}/complete")

        self.assertEqual(completed.status_code, 201)
        self.assertEqual(repeated.status_code, 201)
        self.assertEqual(completed.json()["task_id"], upload_id)
        self.assertEqual(main.scheduler.items, [upload_id])
        tdir = store.task_dir(main.settings.data_dir, upload_id)
        self.assertEqual((tdir / "upload.bin").read_bytes(), b"abcdefghij")
        self.assertFalse((tdir / store.UPLOAD_META).exists())
        self.assertEqual(store.read_status(tdir)["audio_size_bytes"], 10)

    async def test_bad_offset_and_oversized_chunk_do_not_change_file(self):
        upload_id = await self.init_upload()
        self.assertEqual((await self.put(upload_id, 0, b"abcd")).status_code, 200)

        wrong = await self.put(upload_id, 0, b"xx")
        oversized = await self.put(upload_id, 4, b"12345")

        self.assertEqual(wrong.status_code, 409)
        self.assertEqual(wrong.json()["error"]["code"], "UPLOAD_OFFSET_MISMATCH")
        self.assertEqual(oversized.status_code, 413)
        tdir = store.task_dir(main.settings.data_dir, upload_id)
        self.assertEqual((tdir / "upload.bin").read_bytes(), b"abcd")

    async def test_declared_total_is_enforced_and_partial_write_is_rolled_back(self):
        upload_id = await self.init_upload(size=3)
        response = await self.put(upload_id, 0, b"abcd")

        self.assertEqual(response.status_code, 413)
        self.assertEqual(response.json()["error"]["code"], "SIZE_EXCEEDED")
        tdir = store.task_dir(main.settings.data_dir, upload_id)
        self.assertEqual((tdir / "upload.bin").stat().st_size, 0)

    async def test_incomplete_upload_can_be_cancelled(self):
        upload_id = await self.init_upload()
        await self.put(upload_id, 0, b"ab")

        incomplete = await self.client.post(f"/api/uploads/{upload_id}/complete")
        deleted = await self.client.delete(f"/api/uploads/{upload_id}")

        self.assertEqual(incomplete.status_code, 409)
        self.assertEqual(incomplete.json()["error"]["code"], "UPLOAD_INCOMPLETE")
        self.assertEqual(deleted.status_code, 200)
        self.assertFalse(store.task_dir(main.settings.data_dir, upload_id).exists())

    async def test_stale_upload_cleanup(self):
        upload_id = await self.init_upload()
        tdir = store.task_dir(main.settings.data_dir, upload_id)
        meta = tdir / store.UPLOAD_META
        os.utime(meta, (1, 1))

        self.assertEqual(store.cleanup_stale_uploads(main.settings.data_dir, 1), 1)
        self.assertFalse(tdir.exists())

    async def test_upload_sessions_share_queue_capacity(self):
        main.settings.queue_capacity = 1
        await self.init_upload()

        response = await self.client.post(
            "/api/uploads", json={"filename": "second.mka", "size": 10})

        self.assertEqual(response.status_code, 429)
        self.assertEqual(response.json()["error"]["code"], "QUEUE_FULL")

    async def test_legacy_whole_file_upload_endpoint_is_removed(self):
        methods = set()
        for route in main.app.routes:
            if getattr(route, "path", None) == "/api/tasks":
                methods.update(route.methods or ())

        self.assertIn("GET", methods)
        self.assertNotIn("POST", methods)


if __name__ == "__main__":
    unittest.main()
