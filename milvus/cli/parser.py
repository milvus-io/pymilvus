#  Tencent is pleased to support the open source community by making GNES available.
#
#  Copyright (C) 2019 THL A29 Limited, a Tencent company. All rights reserved.
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#  http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.


import argparse
from . import api


def set_base_parser():
    # create the top-level parser
    parser = argparse.ArgumentParser(
        description='Milvus, An open source similarity search engine for massive feature vectors.\n'
                    'Visit %s for tutorials and documentations.' % (('https://github.com/milvus-io/milvus')),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--verbose', action='store_true', default=False,
                        help='turn on detailed logging for debug')
    return parser


def set_composer_parser(parser=None):
    if not parser:
        parser = set_base_parser()
    parser.add_argument('--port',
                        type=int,
                        default=8800,
                        help='host port of the grpc service')
    parser.add_argument('--name',
                        type=str,
                        default='GNES app',
                        help='name of the instance')
    parser.add_argument('--html_path', type=argparse.FileType('w', encoding='utf8'),
                        help='output path of the HTML file, will contain all possible generations')
    parser.add_argument('--shell_path', type=argparse.FileType('w', encoding='utf8'),
                        help='output path of the shell-based starting script')
    parser.add_argument('--swarm_path', type=argparse.FileType('w', encoding='utf8'),
                        help='output path of the docker-compose file for Docker Swarm')
    parser.add_argument('--k8s_path', type=argparse.FileType('w', encoding='utf8'),
                        help='output path of the docker-compose file for Docker Swarm')
    parser.add_argument('--graph_path', type=argparse.FileType('w', encoding='utf8'),
                        help='output path of the mermaid graph file')
    parser.add_argument('--shell_log_redirect', type=str,
                        help='the file path for redirecting shell output. '
                             'when not given, the output will be flushed to stdout')
    parser.add_argument('--mermaid_leftright', action='store_true', default=False,
                        help='showing the flow in left-to-right manner rather than top down')
    parser.add_argument('--docker_img', type=str,
                        default='gnes/gnes:latest-alpine',
                        help='the docker image used in Docker Swarm & Kubernetes')
    return parser


def set_index_parser(parser=None):
    if not parser:
        parser = set_base_parser()

    parser.add_argument('--host', type=str, default='127.0.0.1',
                        help='host address of the checked service')
    parser.add_argument('--port', type=int, required=True,
                        help='control port of the checked service')
    parser.add_argument('--timeout', type=int, default=1000,
                        help='timeout (ms) of one check, -1 for waiting forever')
    parser.add_argument('--retries', type=int, default=3,
                        help='max number of tried health checks before exit 1')
    return parser


def _set_grpc_parser(parser=None):
    if not parser:
        parser = set_base_parser()
    parser.add_argument('--host', type=str, default='127.0.0.1', help='host address of the grpc service')
    parser.add_argument('--port', type=int, default=19530, help='host port of the grpc service')
    return parser

################################################################################
# milvus table [sub-command]

def set_table_create_parser(parser=None):
    if not parser:
        parser = set_base_parser()
    _set_grpc_parser(parser)

    parser.add_argument('--table_name', type=str, default='', help='table name')
    parser.add_argument('--dim', type=int, default=128, help='vector dimension')
    parser.add_argument('--metric_type', type=str, default='L2', help='L2 or IP')

    parser.set_defaults(function=api._create_table)
    return parser

def set_table_exist_parser(parser=None):
    if not parser:
        parser = set_base_parser()
    _set_grpc_parser(parser)

    parser.add_argument('--table_name', type=str, default='', help='table name')
    return parser

def set_table_delete_parser(parser=None):
    if not parser:
        parser = set_base_parser()
    _set_grpc_parser(parser)

    parser.add_argument('--table_name', type=str, default='', help='table name')
    return parser

################################################################################
# milvus index [sub-command]

def set_index_create_parser(parser=None):
    if not parser:
        parser = set_base_parser()
    _set_grpc_parser(parser)

    parser.add_argument('--table_name', type=str, default='', help='table name')
    parser.add_argument('--index_type', type=str, default='', help='index type')
    return parser

def set_index_describe_parser(parser=None):
    if not parser:
        parser = set_base_parser()
    _set_grpc_parser(parser)
    parser.add_argument('--table_name', type=str, default='', help='table name')
    return parser

def set_index_drop_parser(parser=None):
    if not parser:
        parser = set_base_parser()
    _set_grpc_parser(parser)
    parser.add_argument('--table_name', type=str, default='', help='table name')
    return parser

def get_main_parser():
    # create the top-level parser
    parser = set_base_parser()
    adf = argparse.ArgumentDefaultsHelpFormatter
    sp = parser.add_subparsers(dest='cli', title='MILVUS sub-commands',
                               description='use "milvus [sub-command] --help" '
                                           'to get detailed information about each sub-command')

    # table operations
    pp1 = sp.add_parser('table', help='table related operations')
    spp = pp1.add_subparsers(dest='table', title='milvus table sub-commands',
                            description='use "milvus table [sub-command] --help" '
                                        'to get detailed information about each table sub-command')
    spp.required = True
    set_table_create_parser(spp.add_parser('create', help='create table', formatter_class=adf))
    set_table_exist_parser(spp.add_parser('exist', help='check table existence', formatter_class=adf))
    set_table_delete_parser(spp.add_parser('delete', help='delete table', formatter_class=adf))

    # index operations
    # pp = sp.add_parser('index', help='index related operations')
    # spp = pp.add_subparsers(dest='index', title='milvus index sub-commands',
    #                         description='use "milvus index [sub-command] --help" '
    #                                     'to get detailed information about each index sub-command')
    # spp.required = True
    # set_index_create_parser(spp.add_parser('create', help='create index', formatter_class=adf))
    # set_index_describe_parser(spp.add_parser('describe', help='describe index', formatter_class=adf))
    # set_index_drop_parser(spp.add_parser('drop', help='drop index', formatter_class=adf))

    return parser

