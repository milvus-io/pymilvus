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

import traceback
import warnings


def _is_internal_import():
    """Check if the import is from within pymilvus package itself."""
    # Get the call stack
    stack = traceback.extract_stack()
    # Check if the import originates from pymilvus package
    for frame in stack:
        filename = frame.filename
        # Skip the current file and standard library files
        if "pymilvus/orm/__init__.py" in filename:
            continue
        # Check if the import is from pymilvus package
        if "pymilvus" in filename and "site-packages" not in filename:
            # Internal import from pymilvus package
            return True
        # Check if it's from tests (which we consider internal for testing purposes)
        if "/tests/" in filename or "\\tests\\" in filename:
            return True
    return False


# Only show deprecation warning for external imports
if not _is_internal_import():
    warnings.warn(
        "Importing from 'pymilvus.orm' is deprecated and will be removed in a future version. "
        "Please import directly from 'pymilvus' instead. "
        "Example: from pymilvus import Collection, connections",
        DeprecationWarning,
        stacklevel=2,
    )
