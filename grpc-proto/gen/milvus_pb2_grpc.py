# Generated by the gRPC Python protocol compiler plugin. DO NOT EDIT!
import grpc

import common_pb2 as common__pb2
import milvus_pb2 as milvus__pb2


class MilvusServiceStub(object):
  # missing associated documentation comment in .proto file
  pass

  def __init__(self, channel):
    """Constructor.

    Args:
      channel: A grpc.Channel.
    """
    self.CreateCollection = channel.unary_unary(
        '/milvus.proto.milvus.MilvusService/CreateCollection',
        request_serializer=milvus__pb2.CreateCollectionRequest.SerializeToString,
        response_deserializer=common__pb2.Status.FromString,
        )
    self.DropCollection = channel.unary_unary(
        '/milvus.proto.milvus.MilvusService/DropCollection',
        request_serializer=milvus__pb2.DropCollectionRequest.SerializeToString,
        response_deserializer=common__pb2.Status.FromString,
        )
    self.HasCollection = channel.unary_unary(
        '/milvus.proto.milvus.MilvusService/HasCollection',
        request_serializer=milvus__pb2.HasCollectionRequest.SerializeToString,
        response_deserializer=milvus__pb2.BoolResponse.FromString,
        )
    self.LoadCollection = channel.unary_unary(
        '/milvus.proto.milvus.MilvusService/LoadCollection',
        request_serializer=milvus__pb2.LoadCollectionRequest.SerializeToString,
        response_deserializer=common__pb2.Status.FromString,
        )
    self.ReleaseCollection = channel.unary_unary(
        '/milvus.proto.milvus.MilvusService/ReleaseCollection',
        request_serializer=milvus__pb2.ReleaseCollectionRequest.SerializeToString,
        response_deserializer=common__pb2.Status.FromString,
        )
    self.DescribeCollection = channel.unary_unary(
        '/milvus.proto.milvus.MilvusService/DescribeCollection',
        request_serializer=milvus__pb2.DescribeCollectionRequest.SerializeToString,
        response_deserializer=milvus__pb2.DescribeCollectionResponse.FromString,
        )
    self.GetCollectionStatistics = channel.unary_unary(
        '/milvus.proto.milvus.MilvusService/GetCollectionStatistics',
        request_serializer=milvus__pb2.GetCollectionStatisticsRequest.SerializeToString,
        response_deserializer=milvus__pb2.GetCollectionStatisticsResponse.FromString,
        )
    self.ShowCollections = channel.unary_unary(
        '/milvus.proto.milvus.MilvusService/ShowCollections',
        request_serializer=milvus__pb2.ShowCollectionsRequest.SerializeToString,
        response_deserializer=milvus__pb2.ShowCollectionsResponse.FromString,
        )
    self.CreatePartition = channel.unary_unary(
        '/milvus.proto.milvus.MilvusService/CreatePartition',
        request_serializer=milvus__pb2.CreatePartitionRequest.SerializeToString,
        response_deserializer=common__pb2.Status.FromString,
        )
    self.DropPartition = channel.unary_unary(
        '/milvus.proto.milvus.MilvusService/DropPartition',
        request_serializer=milvus__pb2.DropPartitionRequest.SerializeToString,
        response_deserializer=common__pb2.Status.FromString,
        )
    self.HasPartition = channel.unary_unary(
        '/milvus.proto.milvus.MilvusService/HasPartition',
        request_serializer=milvus__pb2.HasPartitionRequest.SerializeToString,
        response_deserializer=milvus__pb2.BoolResponse.FromString,
        )
    self.LoadPartitions = channel.unary_unary(
        '/milvus.proto.milvus.MilvusService/LoadPartitions',
        request_serializer=milvus__pb2.LoadPartitionsRequest.SerializeToString,
        response_deserializer=common__pb2.Status.FromString,
        )
    self.ReleasePartitions = channel.unary_unary(
        '/milvus.proto.milvus.MilvusService/ReleasePartitions',
        request_serializer=milvus__pb2.ReleasePartitionsRequest.SerializeToString,
        response_deserializer=common__pb2.Status.FromString,
        )
    self.GetPartitionStatistics = channel.unary_unary(
        '/milvus.proto.milvus.MilvusService/GetPartitionStatistics',
        request_serializer=milvus__pb2.GetPartitionStatisticsRequest.SerializeToString,
        response_deserializer=milvus__pb2.GetPartitionStatisticsResponse.FromString,
        )
    self.ShowPartitions = channel.unary_unary(
        '/milvus.proto.milvus.MilvusService/ShowPartitions',
        request_serializer=milvus__pb2.ShowPartitionsRequest.SerializeToString,
        response_deserializer=milvus__pb2.ShowPartitionsResponse.FromString,
        )
    self.CreateIndex = channel.unary_unary(
        '/milvus.proto.milvus.MilvusService/CreateIndex',
        request_serializer=milvus__pb2.CreateIndexRequest.SerializeToString,
        response_deserializer=common__pb2.Status.FromString,
        )
    self.DescribeIndex = channel.unary_unary(
        '/milvus.proto.milvus.MilvusService/DescribeIndex',
        request_serializer=milvus__pb2.DescribeIndexRequest.SerializeToString,
        response_deserializer=milvus__pb2.DescribeIndexResponse.FromString,
        )
    self.GetIndexState = channel.unary_unary(
        '/milvus.proto.milvus.MilvusService/GetIndexState',
        request_serializer=milvus__pb2.GetIndexStateRequest.SerializeToString,
        response_deserializer=milvus__pb2.GetIndexStateResponse.FromString,
        )
    self.DropIndex = channel.unary_unary(
        '/milvus.proto.milvus.MilvusService/DropIndex',
        request_serializer=milvus__pb2.DropIndexRequest.SerializeToString,
        response_deserializer=common__pb2.Status.FromString,
        )
    self.Insert = channel.unary_unary(
        '/milvus.proto.milvus.MilvusService/Insert',
        request_serializer=milvus__pb2.InsertRequest.SerializeToString,
        response_deserializer=milvus__pb2.InsertResponse.FromString,
        )
    self.Search = channel.unary_unary(
        '/milvus.proto.milvus.MilvusService/Search',
        request_serializer=milvus__pb2.SearchRequest.SerializeToString,
        response_deserializer=milvus__pb2.SearchResults.FromString,
        )
    self.Flush = channel.unary_unary(
        '/milvus.proto.milvus.MilvusService/Flush',
        request_serializer=milvus__pb2.FlushRequest.SerializeToString,
        response_deserializer=common__pb2.Status.FromString,
        )
    self.GetPersistentSegmentInfo = channel.unary_unary(
        '/milvus.proto.milvus.MilvusService/GetPersistentSegmentInfo',
        request_serializer=milvus__pb2.GetPersistentSegmentInfoRequest.SerializeToString,
        response_deserializer=milvus__pb2.GetPersistentSegmentInfoResponse.FromString,
        )
    self.GetQuerySegmentInfo = channel.unary_unary(
        '/milvus.proto.milvus.MilvusService/GetQuerySegmentInfo',
        request_serializer=milvus__pb2.GetQuerySegmentInfoRequest.SerializeToString,
        response_deserializer=milvus__pb2.GetQuerySegmentInfoResponse.FromString,
        )
    self.RegisterLink = channel.unary_unary(
        '/milvus.proto.milvus.MilvusService/RegisterLink',
        request_serializer=milvus__pb2.RegisterLinkRequest.SerializeToString,
        response_deserializer=milvus__pb2.RegisterLinkResponse.FromString,
        )


class MilvusServiceServicer(object):
  # missing associated documentation comment in .proto file
  pass

  def CreateCollection(self, request, context):
    # missing associated documentation comment in .proto file
    pass
    context.set_code(grpc.StatusCode.UNIMPLEMENTED)
    context.set_details('Method not implemented!')
    raise NotImplementedError('Method not implemented!')

  def DropCollection(self, request, context):
    # missing associated documentation comment in .proto file
    pass
    context.set_code(grpc.StatusCode.UNIMPLEMENTED)
    context.set_details('Method not implemented!')
    raise NotImplementedError('Method not implemented!')

  def HasCollection(self, request, context):
    # missing associated documentation comment in .proto file
    pass
    context.set_code(grpc.StatusCode.UNIMPLEMENTED)
    context.set_details('Method not implemented!')
    raise NotImplementedError('Method not implemented!')

  def LoadCollection(self, request, context):
    # missing associated documentation comment in .proto file
    pass
    context.set_code(grpc.StatusCode.UNIMPLEMENTED)
    context.set_details('Method not implemented!')
    raise NotImplementedError('Method not implemented!')

  def ReleaseCollection(self, request, context):
    # missing associated documentation comment in .proto file
    pass
    context.set_code(grpc.StatusCode.UNIMPLEMENTED)
    context.set_details('Method not implemented!')
    raise NotImplementedError('Method not implemented!')

  def DescribeCollection(self, request, context):
    # missing associated documentation comment in .proto file
    pass
    context.set_code(grpc.StatusCode.UNIMPLEMENTED)
    context.set_details('Method not implemented!')
    raise NotImplementedError('Method not implemented!')

  def GetCollectionStatistics(self, request, context):
    # missing associated documentation comment in .proto file
    pass
    context.set_code(grpc.StatusCode.UNIMPLEMENTED)
    context.set_details('Method not implemented!')
    raise NotImplementedError('Method not implemented!')

  def ShowCollections(self, request, context):
    # missing associated documentation comment in .proto file
    pass
    context.set_code(grpc.StatusCode.UNIMPLEMENTED)
    context.set_details('Method not implemented!')
    raise NotImplementedError('Method not implemented!')

  def CreatePartition(self, request, context):
    # missing associated documentation comment in .proto file
    pass
    context.set_code(grpc.StatusCode.UNIMPLEMENTED)
    context.set_details('Method not implemented!')
    raise NotImplementedError('Method not implemented!')

  def DropPartition(self, request, context):
    # missing associated documentation comment in .proto file
    pass
    context.set_code(grpc.StatusCode.UNIMPLEMENTED)
    context.set_details('Method not implemented!')
    raise NotImplementedError('Method not implemented!')

  def HasPartition(self, request, context):
    # missing associated documentation comment in .proto file
    pass
    context.set_code(grpc.StatusCode.UNIMPLEMENTED)
    context.set_details('Method not implemented!')
    raise NotImplementedError('Method not implemented!')

  def LoadPartitions(self, request, context):
    # missing associated documentation comment in .proto file
    pass
    context.set_code(grpc.StatusCode.UNIMPLEMENTED)
    context.set_details('Method not implemented!')
    raise NotImplementedError('Method not implemented!')

  def ReleasePartitions(self, request, context):
    # missing associated documentation comment in .proto file
    pass
    context.set_code(grpc.StatusCode.UNIMPLEMENTED)
    context.set_details('Method not implemented!')
    raise NotImplementedError('Method not implemented!')

  def GetPartitionStatistics(self, request, context):
    # missing associated documentation comment in .proto file
    pass
    context.set_code(grpc.StatusCode.UNIMPLEMENTED)
    context.set_details('Method not implemented!')
    raise NotImplementedError('Method not implemented!')

  def ShowPartitions(self, request, context):
    # missing associated documentation comment in .proto file
    pass
    context.set_code(grpc.StatusCode.UNIMPLEMENTED)
    context.set_details('Method not implemented!')
    raise NotImplementedError('Method not implemented!')

  def CreateIndex(self, request, context):
    # missing associated documentation comment in .proto file
    pass
    context.set_code(grpc.StatusCode.UNIMPLEMENTED)
    context.set_details('Method not implemented!')
    raise NotImplementedError('Method not implemented!')

  def DescribeIndex(self, request, context):
    # missing associated documentation comment in .proto file
    pass
    context.set_code(grpc.StatusCode.UNIMPLEMENTED)
    context.set_details('Method not implemented!')
    raise NotImplementedError('Method not implemented!')

  def GetIndexState(self, request, context):
    # missing associated documentation comment in .proto file
    pass
    context.set_code(grpc.StatusCode.UNIMPLEMENTED)
    context.set_details('Method not implemented!')
    raise NotImplementedError('Method not implemented!')

  def DropIndex(self, request, context):
    # missing associated documentation comment in .proto file
    pass
    context.set_code(grpc.StatusCode.UNIMPLEMENTED)
    context.set_details('Method not implemented!')
    raise NotImplementedError('Method not implemented!')

  def Insert(self, request, context):
    # missing associated documentation comment in .proto file
    pass
    context.set_code(grpc.StatusCode.UNIMPLEMENTED)
    context.set_details('Method not implemented!')
    raise NotImplementedError('Method not implemented!')

  def Search(self, request, context):
    # missing associated documentation comment in .proto file
    pass
    context.set_code(grpc.StatusCode.UNIMPLEMENTED)
    context.set_details('Method not implemented!')
    raise NotImplementedError('Method not implemented!')

  def Flush(self, request, context):
    # missing associated documentation comment in .proto file
    pass
    context.set_code(grpc.StatusCode.UNIMPLEMENTED)
    context.set_details('Method not implemented!')
    raise NotImplementedError('Method not implemented!')

  def GetPersistentSegmentInfo(self, request, context):
    # missing associated documentation comment in .proto file
    pass
    context.set_code(grpc.StatusCode.UNIMPLEMENTED)
    context.set_details('Method not implemented!')
    raise NotImplementedError('Method not implemented!')

  def GetQuerySegmentInfo(self, request, context):
    # missing associated documentation comment in .proto file
    pass
    context.set_code(grpc.StatusCode.UNIMPLEMENTED)
    context.set_details('Method not implemented!')
    raise NotImplementedError('Method not implemented!')

  def RegisterLink(self, request, context):
    """TODO: remove
    """
    context.set_code(grpc.StatusCode.UNIMPLEMENTED)
    context.set_details('Method not implemented!')
    raise NotImplementedError('Method not implemented!')


def add_MilvusServiceServicer_to_server(servicer, server):
  rpc_method_handlers = {
      'CreateCollection': grpc.unary_unary_rpc_method_handler(
          servicer.CreateCollection,
          request_deserializer=milvus__pb2.CreateCollectionRequest.FromString,
          response_serializer=common__pb2.Status.SerializeToString,
      ),
      'DropCollection': grpc.unary_unary_rpc_method_handler(
          servicer.DropCollection,
          request_deserializer=milvus__pb2.DropCollectionRequest.FromString,
          response_serializer=common__pb2.Status.SerializeToString,
      ),
      'HasCollection': grpc.unary_unary_rpc_method_handler(
          servicer.HasCollection,
          request_deserializer=milvus__pb2.HasCollectionRequest.FromString,
          response_serializer=milvus__pb2.BoolResponse.SerializeToString,
      ),
      'LoadCollection': grpc.unary_unary_rpc_method_handler(
          servicer.LoadCollection,
          request_deserializer=milvus__pb2.LoadCollectionRequest.FromString,
          response_serializer=common__pb2.Status.SerializeToString,
      ),
      'ReleaseCollection': grpc.unary_unary_rpc_method_handler(
          servicer.ReleaseCollection,
          request_deserializer=milvus__pb2.ReleaseCollectionRequest.FromString,
          response_serializer=common__pb2.Status.SerializeToString,
      ),
      'DescribeCollection': grpc.unary_unary_rpc_method_handler(
          servicer.DescribeCollection,
          request_deserializer=milvus__pb2.DescribeCollectionRequest.FromString,
          response_serializer=milvus__pb2.DescribeCollectionResponse.SerializeToString,
      ),
      'GetCollectionStatistics': grpc.unary_unary_rpc_method_handler(
          servicer.GetCollectionStatistics,
          request_deserializer=milvus__pb2.GetCollectionStatisticsRequest.FromString,
          response_serializer=milvus__pb2.GetCollectionStatisticsResponse.SerializeToString,
      ),
      'ShowCollections': grpc.unary_unary_rpc_method_handler(
          servicer.ShowCollections,
          request_deserializer=milvus__pb2.ShowCollectionsRequest.FromString,
          response_serializer=milvus__pb2.ShowCollectionsResponse.SerializeToString,
      ),
      'CreatePartition': grpc.unary_unary_rpc_method_handler(
          servicer.CreatePartition,
          request_deserializer=milvus__pb2.CreatePartitionRequest.FromString,
          response_serializer=common__pb2.Status.SerializeToString,
      ),
      'DropPartition': grpc.unary_unary_rpc_method_handler(
          servicer.DropPartition,
          request_deserializer=milvus__pb2.DropPartitionRequest.FromString,
          response_serializer=common__pb2.Status.SerializeToString,
      ),
      'HasPartition': grpc.unary_unary_rpc_method_handler(
          servicer.HasPartition,
          request_deserializer=milvus__pb2.HasPartitionRequest.FromString,
          response_serializer=milvus__pb2.BoolResponse.SerializeToString,
      ),
      'LoadPartitions': grpc.unary_unary_rpc_method_handler(
          servicer.LoadPartitions,
          request_deserializer=milvus__pb2.LoadPartitionsRequest.FromString,
          response_serializer=common__pb2.Status.SerializeToString,
      ),
      'ReleasePartitions': grpc.unary_unary_rpc_method_handler(
          servicer.ReleasePartitions,
          request_deserializer=milvus__pb2.ReleasePartitionsRequest.FromString,
          response_serializer=common__pb2.Status.SerializeToString,
      ),
      'GetPartitionStatistics': grpc.unary_unary_rpc_method_handler(
          servicer.GetPartitionStatistics,
          request_deserializer=milvus__pb2.GetPartitionStatisticsRequest.FromString,
          response_serializer=milvus__pb2.GetPartitionStatisticsResponse.SerializeToString,
      ),
      'ShowPartitions': grpc.unary_unary_rpc_method_handler(
          servicer.ShowPartitions,
          request_deserializer=milvus__pb2.ShowPartitionsRequest.FromString,
          response_serializer=milvus__pb2.ShowPartitionsResponse.SerializeToString,
      ),
      'CreateIndex': grpc.unary_unary_rpc_method_handler(
          servicer.CreateIndex,
          request_deserializer=milvus__pb2.CreateIndexRequest.FromString,
          response_serializer=common__pb2.Status.SerializeToString,
      ),
      'DescribeIndex': grpc.unary_unary_rpc_method_handler(
          servicer.DescribeIndex,
          request_deserializer=milvus__pb2.DescribeIndexRequest.FromString,
          response_serializer=milvus__pb2.DescribeIndexResponse.SerializeToString,
      ),
      'GetIndexState': grpc.unary_unary_rpc_method_handler(
          servicer.GetIndexState,
          request_deserializer=milvus__pb2.GetIndexStateRequest.FromString,
          response_serializer=milvus__pb2.GetIndexStateResponse.SerializeToString,
      ),
      'DropIndex': grpc.unary_unary_rpc_method_handler(
          servicer.DropIndex,
          request_deserializer=milvus__pb2.DropIndexRequest.FromString,
          response_serializer=common__pb2.Status.SerializeToString,
      ),
      'Insert': grpc.unary_unary_rpc_method_handler(
          servicer.Insert,
          request_deserializer=milvus__pb2.InsertRequest.FromString,
          response_serializer=milvus__pb2.InsertResponse.SerializeToString,
      ),
      'Search': grpc.unary_unary_rpc_method_handler(
          servicer.Search,
          request_deserializer=milvus__pb2.SearchRequest.FromString,
          response_serializer=milvus__pb2.SearchResults.SerializeToString,
      ),
      'Flush': grpc.unary_unary_rpc_method_handler(
          servicer.Flush,
          request_deserializer=milvus__pb2.FlushRequest.FromString,
          response_serializer=common__pb2.Status.SerializeToString,
      ),
      'GetPersistentSegmentInfo': grpc.unary_unary_rpc_method_handler(
          servicer.GetPersistentSegmentInfo,
          request_deserializer=milvus__pb2.GetPersistentSegmentInfoRequest.FromString,
          response_serializer=milvus__pb2.GetPersistentSegmentInfoResponse.SerializeToString,
      ),
      'GetQuerySegmentInfo': grpc.unary_unary_rpc_method_handler(
          servicer.GetQuerySegmentInfo,
          request_deserializer=milvus__pb2.GetQuerySegmentInfoRequest.FromString,
          response_serializer=milvus__pb2.GetQuerySegmentInfoResponse.SerializeToString,
      ),
      'RegisterLink': grpc.unary_unary_rpc_method_handler(
          servicer.RegisterLink,
          request_deserializer=milvus__pb2.RegisterLinkRequest.FromString,
          response_serializer=milvus__pb2.RegisterLinkResponse.SerializeToString,
      ),
  }
  generic_handler = grpc.method_handlers_generic_handler(
      'milvus.proto.milvus.MilvusService', rpc_method_handlers)
  server.add_generic_rpc_handlers((generic_handler,))


class ProxyServiceStub(object):
  # missing associated documentation comment in .proto file
  pass

  def __init__(self, channel):
    """Constructor.

    Args:
      channel: A grpc.Channel.
    """
    self.RegisterLink = channel.unary_unary(
        '/milvus.proto.milvus.ProxyService/RegisterLink',
        request_serializer=milvus__pb2.RegisterLinkRequest.SerializeToString,
        response_deserializer=milvus__pb2.RegisterLinkResponse.FromString,
        )


class ProxyServiceServicer(object):
  # missing associated documentation comment in .proto file
  pass

  def RegisterLink(self, request, context):
    # missing associated documentation comment in .proto file
    pass
    context.set_code(grpc.StatusCode.UNIMPLEMENTED)
    context.set_details('Method not implemented!')
    raise NotImplementedError('Method not implemented!')


def add_ProxyServiceServicer_to_server(servicer, server):
  rpc_method_handlers = {
      'RegisterLink': grpc.unary_unary_rpc_method_handler(
          servicer.RegisterLink,
          request_deserializer=milvus__pb2.RegisterLinkRequest.FromString,
          response_serializer=milvus__pb2.RegisterLinkResponse.SerializeToString,
      ),
  }
  generic_handler = grpc.method_handlers_generic_handler(
      'milvus.proto.milvus.ProxyService', rpc_method_handlers)
  server.add_generic_rpc_handlers((generic_handler,))
