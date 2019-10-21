# pymilvus 0.5.0(2019-10-21)

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
- \#97 - modify CHANGLOG.md to adjust standdard

## New Feature
- \#94 - remove stream call in grpc


# pymilvus 0.4.0(2019-09-12)

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

## New Feature
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
