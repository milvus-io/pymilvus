# Change logs

## pymilvus 1.1.1(TBD)

### New Feature

### Improvement

## pymilvus 1.1.0(2021-04-29)

### New Feature
- \#295 - Complete all tests
- \#456 - Remove Numpy in require packages
- \#484 - Add `partition_tag` parameter to `get_entity_by_id` and `delete_entity_by_id`, Support `release_collection`

### Improvement
- \#442 - Add tutorial documentation
- \#444 - Add index-param documentation
- \#447 - Add query-results documentation
- \#450 - Update API reference documentation
- \#457 - Update require package `ujson` version
- \#491 - Update support versions
- \#497 - Force using http/https scheme with http-handler
- \#498 - Bump jinja2 pygments and py 
- \#500 - Prepare for 1.1.0


## pymilvus 1.0.1(2021-03-08)

### New Feature
- \#436 - Update PyMilvus v1.0.x documentation frame

### Task
- \#432 - Remove hybrid APIs


## pymilvus 1.0.0(2021-03-05)

### Improvement
- \#424 - Prepare for 1.x


## pymilvus 0.4.0(2021-01-21)

### Improvement
- \#345 - Do not support python 3.5.x any more


## pymilvus 0.2.15(2021-01-06)

### New Feature
- \#392 - Upgrade method "load_collection" for milvus v0.10.5


## pymilvus 0.2.14(2020-07-20)

### Bug
- \#237 - Fix wrong result on 'has_partition' with http handler


## pymilvus 0.2.13(2020-06-13)

### New Feature
- \#220 - Add sdk statistics badges
- \#224 - Add option 'grpc.keepalive_time_ms' in grpc channel

### Bug
- \#218 - Fix retults by 'get_entity_by_id' with HTTP
- \#222 - Fix protobuf import failed


## pymilvus 0.2.12(2020-05-29)

### Bug
- \215 - Fix 'ping() got an unexpected keyword argument 'max_retry' when using http handler'

### Improvement
- \#216 Optimize checking matched version behavior

### New Feature
- \#213 - Add new API called 'reload_segments'


## pymilvus 0.2.11(2020-05-15)

### Bug
- \#190 - Fix rpc return error when error occur
- \#200 - Fix TypeError raised in inserting vectors with async set true
- \#201 - Fix get collection info failed when info is null
- \#202 - Fix async 'NoneType' object has no attribute 'result'
- \#207 - Fix excepted exceptions if async invoke with timeout

### Improvement
- \#175 - Remove connect API
- \#203 - Add method 'is_done()' in Future class
- \#205 - Filter search results which id is -1
- \#206 - Update APIs names

### New Feature
- \#174 - Support connection pool
- \#177 - Support async API
- \#199 - Add API 'has_partition'


## pymilvus 0.2.10(2020-04-15)

### Improvement
- \#182 - Optimize usage of gRPC future

### New Feature
- \#178 - New index annoy


## pymilvus 0.2.9(2020-03-29)

### Bug
- \#168 - Fix incorrect demo in readme
- \#172 - Allow empty list in flush passing parameter

### Improvement
- \#175 - Remove connect APIs

### New Feature
- \#170 - Add index RNSG into HTTP handler
- \#171 - Add two new binary metric: SUBSTRUCTURE and SUPERSTRUCTURE
- \#174 - Support connection pool
- \#177 - Support async APIs


## pymilvus 0.2.8(2020-03-11)

### Bug
- \#140 - Convert distance in http result to float value
- \#141 - Add index IVF_SQ8H into http handler
- \#146 - Preserve existing loggers when imported to other applications
- \#162 - Remove version_notes from API doc

### Improvement
- \#143 - Remove query range when searching
- \#149 - Change json parser to `ujson`
- \#151 - Add new interface `compact`
- \#154 - Add new interface `get_vector_by_id`
- \#156 - Remove data range
- \#157 - Remove partition name
- \#160 - Add CRUD API to readme
- \#164 - Add vectors CRUD example


## pymilvus 0.2.7(2020-01-16)

### Bug
- \#136 - Fix incorrect description in README.md

### New Feature
- \#138 - Add binary vectors support


## pymilvus 0.2.6(2019-12-07)

### Bug
- \#127 - Fix crash when printing search result
- \#133 - Change error message when array is illegal

### Improvement
- \#124 - Make method `_set_hook` public
- \#125 - Replace all old method invoke with alternative new method name
- \#132 - Change index type `MIX_NSG` to `NSG`

### New Feature
- \#112 - Support operation `with`
- \#123 - Add a new index named `IVF_PQ`

### Task
- \#117 - Implement new api for partition


## pymilvus 0.2.5(2019-11-13)

### Bug
- \#114 - Fix method `has_table` bug for return a tuple

### Improvement
- \#108 - Update version table in README.md
- \#110 - Remove attribute `server_address` in class GrpcMilvus
- \#111 - Make method `set_channel` protected
- \#115 - Format index type name
- \#118 - Add hook in search method
- \#119 - Set timeout -1 for default to allow invoke synchronously


## pymilvus 0.2.4(2019-11-04)

### Bug
- \#102 - Make methods `delete_by_range` private

### Improvement
- \#103 - Remove .codecov.yml and .travis.yml
- \#105 - Update READ.md to update version table and version note
- \#108 - Update version table in README.md adding milvus v0.5.1

### New Feature
- \#100 - Add new index type PQ
- \#101 - Give client methods new alterative names.


## pymilvus 0.2.3(2019-10-21)

### Bug
- MS-452 - Fix build index timeout bug
- MS-469 - Add index_file_size min value check
- MS-512 - Fix ids error
- MS-521 - Fix ids check error
- MS-522 - Fix index_param check error
- \#83 - Fix celery server create table error
- \#88 - Fix bug connecting failed in celery cluster
- \#90 - Remove TODO file
- \#93 - Fix describe_table not sync'ed with latest proto bug

### Improvement
- \#97 - Modify CHANGLOG.md to adjust standard

### Feature
- \#94 - Remove stream call in grpc


## pymilvus 0.2.2(2019-09-12)

### Bug
- \#12 - IndexType change & fix param of search_vector_in_files ranges
- \#23 - Fix not-connect raise wrong exception bug
- \#26 - Format of server_version's return value fixed
- \#27 - Correct connect and disconnect logic    
- \#32 - Top_k in search_vectors set ranges
- \#41 - Optimize some dataclass and add utils.py module, Fix type of tile_ids in search_in_files api transfer bug, Fix Prepare after Prepare will raise ParamError bug
- \#43 - Fix prepare after prepare will raise exceptions bug
- MS-112 - Fix type of top_k not checked error
- MS-118 - IndexType param checked
- MS-123 - Create table param re-checked
- MS-132 - Fix connected return value wrong error
- MS-134 - Removing not using code and comment
- MS-165 ~ MS-170 - Fixed default ip and port wrong bug
- MS-174 - Fix table_name=None buf create table successfully bug
- MS-182 - Fix query ranges param not control bug
- MS-185 - Fix connection no timeout bug 
- MS-243 - Fixed python client cost too much time bug
- MS-424 - IndexParam str more friendly
- MS-433 - Fix wrong passing param bug
- MS-438 - Fix add vectors with ids bug
- MS-439 - Fix search result unpack bug on windows
- MS-444 - Fix timeout bug
  
### Improvement
- \#10 - Update examples
- \#14 - Update example
- \#28 - Update README with newest sdk
- \#33 - Update thrift has_table API
- \#38 - Update thrift score to distance
- \#45, \#47 -  Update AdvancedExample
- \#51 - Brute force thread_safe for sdk
- \#71 - Update thrift api, search vectors results are binaries
- \#73 - Change table schema print format
- \#74 - Add a new attribute server_address
- \#77 - Print out TopKResult more friendly

### Feature
- \#3 - Transport protocol configurable by settings, add_vector support non-binary array inputs
- \#6 - Status quick-check-success, Log message more understandable, Status code related to Thrift Exception, Operations before connect will raise NotConnectError, Adding UNKNOWN Status
- \#8 - Add new api: search_vectors_by_file, fix some bugs
- \#17 - Implement has_table interface, fix spelling error, reformat as PEP8
- \#19 - Hide Prepare object and support old version
- \#21 - Support search by range, fix server_status return None bug
- \#76 - Replace thrift with grpc
- \#79 - Add new indextype MIN_NSG, fix indextype name bug
- \#81 - Remove all thrift, update interface and interface param
- \#82 - Add timeout option, add default param in tableschema and indexparam
    
### Task
- \#1 - Build Repository
- \#2 - Add CHANGELOG.md and LICENSE, update setup.py
- \#70 - Update README and examples
