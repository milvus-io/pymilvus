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

from typing import List
import grpc

from ..grpc_gen import milvus_pb2_grpc
from ..abstract_grpc_handler import SecureMixin
from ..settings import DefaultConfig


class AsyncGrpcHandler(milvus_pb2_grpc.MilvusServiceStub, SecureMixin):
    channel: grpc.aio.Channel

    def __init__(self, address: str, timeout: float, **kwargs):
        self.opts = DefaultConfig.CONNECTION_OPTS

        secure_opts = kwargs.get("secure")

        _extra_opts = self.get_extra_opts(secure_opts)
        _creds = self.get_credentials(secure_opts)
        _interceptors = self.get_interceptors(secure_opts)

        if isinstance(_extra_opts, List) and len(_extra_opts)> 0:
            self.opts.extend(_extra_opts)

        params = {
            "target": address,
            "options": self.opts,
        }
        if len(_interceptors) > 0:
            params["interceptors"] = _interceptors

        if _creds is not None:
            params["credentials"] = _creds
            self.channel = grpc.aio.secure_channel(**params)
        else:
            self.channel = grpc.aio.insecure_channel(**params)

        # set up MilvusServiceStub
        super().__init__(self.channel)
