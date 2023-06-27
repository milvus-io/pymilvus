# Copyright (C) 2019-2021 Zilliz. All rights reserved.
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

from typing import Any

from .mutation import MutationResult
from .search import SearchResult


# TODO(dragondriver): how could we inherit the docstring elegantly?
class BaseFuture:
    def __init__(self, future: Any) -> None:
        self._f = future

    def result(self) -> Any:
        """Return the result from future object.

        It's a synchronous interface. It will wait executing until
        server respond or timeout occur(if specified).
        """
        return self.on_response(self._f.result())

    def on_response(self, res: Any):
        return res

    def cancel(self):
        """Cancel the request."""
        return self._f.cancel()

    def done(self):
        """Wait for request done."""
        return self._f.done()


class SearchFuture(BaseFuture):
    def on_response(self, res: Any):
        return SearchResult(res)


class MutationFuture(BaseFuture):
    def on_response(self, res: Any):
        return MutationResult(res)
