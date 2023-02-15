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

from typing import Dict, List

import grpc


class SecureMixin:
    # TODO get interceptors from secure options
    def get_interceptors(self, secure_opts) -> List:
        return  []

    # TODO get ssl credentials
    def get_credentials(self, secure_opts: Dict) ->grpc.ChannelCredentials:
        if isinstance(secure_opts, Dict) and len(secure_opts) > 0:
            return grpc.ssl_channel_credentials()

    def get_extra_opts(self, secure_opts: Dict) -> List[tuple]:
        # TODO try to get extra_opts from secure_opts
        if isinstance(secure_opts, Dict) and len(secure_opts) > 0:
            return ('grpc.ssl_target_name_override', 'server_name')
