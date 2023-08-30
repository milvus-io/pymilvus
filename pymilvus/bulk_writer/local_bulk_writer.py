# Copyright (C) 2019-2023 Zilliz. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License. You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License
# is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
# or implied. See the License for the specific language governing permissions and limitations under
# the License.

import logging
import threading
import time
import uuid
from pathlib import Path
from threading import Thread
from typing import Callable, Optional

from pymilvus.orm.schema import CollectionSchema

from .bulk_writer import BulkWriter
from .constants import (
    MB,
    BulkFileType,
)

logger = logging.getLogger("local_bulk_writer")
logger.setLevel(logging.DEBUG)


class LocalBulkWriter(BulkWriter):
    def __init__(
        self,
        schema: CollectionSchema,
        local_path: str,
        segment_size: int = 512 * MB,
        file_type: BulkFileType = BulkFileType.NPY,
    ):
        super().__init__(schema, segment_size, file_type)
        self._local_path = local_path
        self._make_dir()
        self._flush_count = 0
        self._working_thread = {}

    def __del__(self):
        if len(self._working_thread) > 0:
            for k, th in self._working_thread.items():
                logger.info(f"Wait thread '{k}' to finish")
                th.join()

    def _make_dir(self):
        Path(self._local_path).mkdir(exist_ok=True)
        uidir = Path(self._local_path).joinpath(str(uuid.uuid4()))
        Path(uidir).mkdir(exist_ok=True)
        self._local_path = uidir
        logger.info(f"Local buffer writer initialized, target path: {uidir}")

    def append(self, row: dict, **kwargs):
        super().append(row, **kwargs)

        if super().buffer_size > super().segment_size:
            self.commit(_async=True)

    def commit(self, **kwargs):
        while len(self._working_thread) > 0:
            logger.info("Previous flush action is not finished, waiting...")
            time.sleep(0.5)

        logger.info(
            f"Prepare to flush buffer, row_count: {super().buffer_row_count}, size: {super().buffer_size}"
        )
        _async = kwargs.get("_async", False)
        call_back = kwargs.get("call_back", None)
        x = Thread(target=self._flush, args=(call_back,))
        x.start()
        if not _async:
            x.join()
        super().commit()  # reset the buffer size

    def _flush(self, call_back: Optional[Callable] = None):
        self._working_thread[threading.current_thread().name] = threading.current_thread()
        self._flush_count = self._flush_count + 1
        target_path = Path.joinpath(self._local_path, str(self._flush_count))

        old_buffer = super()._new_buffer()
        file_list = old_buffer.persist(str(target_path))
        if call_back:
            call_back(file_list)
        del self._working_thread[threading.current_thread().name]

    @property
    def data_path(self):
        return self._local_path
