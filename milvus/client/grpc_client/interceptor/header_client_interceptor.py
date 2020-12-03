import collections

import grpc
from . import generic_interceptor


class _ClientCallDetails(
        collections.namedtuple(
            '_ClientCallDetails',
            ('method', 'timeout', 'metadata', 'credentials')),
        grpc.ClientCallDetails):
    pass


def header_adder_interceptor(metadata):

    def intercept_call(client_call_details, request_iterator, request_streaming,
                       response_streaming):
        _metadata = []
        if client_call_details.metadata is not None:
            _metadata = list(client_call_details.metadata)
        _metadata.extend(metadata)
        client_call_details = _ClientCallDetails(
            client_call_details.method, client_call_details.timeout, _metadata,
            client_call_details.credentials)
        return client_call_details, request_iterator, None

    return generic_interceptor.create(intercept_call)
