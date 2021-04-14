| Methods                                        | Descriptions                     | 参数                                                         | 返回值                  |
| ---------------------------------------------- | :------------------------------- | ------------------------------------------------------------ | ----------------------- |
| Connections.configure(**kwargs)                | Configure connections.           | milvus 客户端连接相关配置，包括 ip，port 等；                | None或Raise Exception   |
| Connections.add_connection(alias, conn)        | Add a connection using alias.    | alias：待添加的 milvus 客户端连接 conn 的别名，conn：milvus 客户端连接； | None 或 Raise Exception |
| Connections.remove_connection(alias)           | Remove a connection by alias.    | alias：待删除的 milvus 客户端连接别名；                      | None 或 Raise Exception |
| Connections.create_connection(alias, **kwargs) | Create a connection named alias. | alias：待创建的 milvus 客户端连接别名，kwargs：客户端连接配置，包括 ip，port 等； | None 或 Raise Exception |
| Connections.get_connection(alias)              | Get a connection by alias.       | alias：待使用的 milvus 客户端连接别名；                      | milvus 客户端连接       |
