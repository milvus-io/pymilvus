# pymilvus 0.2.9(TBD)

## Bug
- \#168 - fix incorrect demo in readme

## New Feature
- \#170 - add index RNSG into HTTP handler
- \#171 - add two new binary metric: SUBSTRUCTURE and SUPERSTRUCTURE

# pymilvus 0.2.8(2020-03-11)

## Bug
- \#140 - convert distance in http result to float value
- \#141 - add index IVF_SQ8H into http handler
- \#146 - preserve existing loggers when imported to other applications
- \#162 - remove version_notes from API doc

## Improvement
- \#143 - remove query range when searching
- \#149 - change json parser to `ujson`
- \#151 - add new interface `compact`
- \#154 - add new interface `get_vector_by_id`
- \#156 - remove data range
- \#157 - remove partition name
- \#160 - add CRUD API to readme
- \#164 - add vectors CRUD example


# pymilvus 0.2.7(2020-01-16)

## Bug
- \#136 - fix incorrect description in README.md

## New Feature
- \#138 - add binary vectors support


# pymilvus 0.2.6(2019-12-07)

## Bug
- \#127 - fix crash when printing search result
- \#133 - change error message when array is illegal

## Improvement
- \#124 - make method `_set_hook` public
- \#125 - replace all old method invoke with alternative new method name
- \#132 - change index type `MIX_NSG` to `NSG`

## New Feature
- \#112 - support operation `with`
- \#123 - add a new index named `IVF_PQ`

## Task
- \#117 - implement new api for partition


# pymilvus 0.2.5(2019-11-13)

## Bug
- \#114 - fix method `has_table` bug for return a tuple

## Improvement
- \#108 - Update version table in README.md
- \#110 - remove attribute `server_address` in class GrpcMilvus
- \#111 - make method `set_channel` protected
- \#115 - Format index type name
- \#118 - add hook in search method
- \#119 - set timeout -1 for default to allow invoke synchronously


# pymilvus 0.2.4(2019-11-04)

## Bug
- \#102 - make methods `delete_by_range` private

## Improvement
- \#103 - remove .codecov.yml and .travis.yml
- \#105 - update READ.md to update version table and version note
- \#108 - Update version table in README.md adding milvus v0.5.1

## New Feature
- \#100 - add new index type PQ
- \#101 - Give client methods new alterative names.


# pymilvus 0.2.3(2019-10-21)

## Bug
- MS-452 - fix build index timeout bug
- MS-469 - add index_file_size min value check
- MS-512 - fix ids error
- MS-521 - fix ids check error
- MS-522 - fix index_param check error
- \#83 - fix celery server create table error
- \#88 - fix bug connecting failed in celery cluster
- \#90 - remove TODO file
- \#93 - fix describe_table not sync'ed with latest proto bug

## Improvement
- \#97 - modify CHANGLOG.md to adjust standard

## Feature
- \#94 - remove stream call in grpc


# pymilvus 0.2.2(2019-09-12)

## Bug
- \#12 - IndexType change & fix param of search_vector_in_files ranges
- \#23 - fix not-connect raise wrong exception bug
- \#26 - format of server_version's return value fixed
- \#27 - correct connect and disconnect logic    
- \#32 - top_k in search_vectors set ranges
- \#41 - Optimize some dataclass and add utils.py module, Fix type of tile_ids in search_in_files api transfer bug, Fix Prepare after Prepare will raise ParamError bug
- \#43 - fix prepare after prepare will raise exceptions bug
- MS-112 - fix type of top_k not checked error
- MS-118 - IndexType param checked
- MS-123 - create table param re-checked
- MS-132 - fix connected return value wrong error
- MS-134 - removing not using code and comment
- MS-165 ~ MS-170 - fixed default ip and port wrong bug
- MS-174 - fix table_name=None buf create table successfully bug
- MS-182 - fix query ranges param not control bug
- MS-185 - fix connection no timeout bug 
- MS-243 - fixed python client cost too much time bug
- MS-424 - IndexParam str more friendly
- MS-433 - fix wrong passing param bug
- MS-438 - fix add vectors with ids bug
- MS-439 - fix search result unpack bug on windows
- MS-444 - fix timeout bug
  
## Improvement
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
- \#77 - print out TopKResult more friendly

## Feature
- \#3 - transport protocol configurable by settings, add_vector support non-binary array inputs
- \#6 - Status quick-check-success, Log message more understandable, Status code related to Thrift Exception, Operations before connect will raise NotConnectError, Adding UNKNOWN Status
- \#8 - Add new api: search_vectors_by_file, fix some bugs
- \#17 - Implement has_table interface, fix spelling error, reformat as PEP8
- \#19 - Hide Prepare object and support old version
- \#21 - support search by range, fix server_status return None bug
- \#76 - replace thrift with grpc
- \#79 - add new indextype MIN_NSG, fix indextype name bug
- \#81 - remove all thrift, update interface and interface param
- \#82 - add timeout option, add default param in tableschema and indexparam
    
## Task
- \#1 - Build Repository
- \#2 - Add CHANGELOG.md and LICENSE, update setup.py
- \#70 - update README and examples
