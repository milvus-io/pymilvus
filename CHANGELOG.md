### Bug
---

- \#12 ISSUE
    - IndexType change
    - fix param of search_vector_in_files ranges
- \#23 ISSUE: fix not-connect raise wrong exception bug
- \#25 ISSUE: **[M-112]** fix type of top_k not checked error
- \#26 ISSUE: format of server_version's return value fixed
- \#27 ISSUE: correct connect and disconnect logic  
- \#30 ISSUE: **[M-118]** IndexType param checked  
- \#31 ISSUE: **[M-123]** create table param re-checked
- \#32 ISSUE: top_k in search_vectors set ranges
- \#35 ISSUE: **[M-132]** fix connected return value wrong error
- \#39 ISSUE: **[M-134]** removing not using code and comment
- \#41 ISSUE
    - Optimize some dataclass and add utils.py module
    - Fix type of tile_ids in search_in_files api transfer bug
    - Fix Prepare after Prepare will raise ParamError bug
- \#43 ISSUE: fix prepare after prepare will raise exceptions bug
- \#50 ISSUE: **[M-174]** fix table_name=None buf create table successfully bug
- \#53 ISSUE: **[MS-165 ~ MS-170]** fixed default ip and port wrong bug
- \#59 ISSUE: **[MS-182]** fix query ranges param not control bug
- \#60 ISSUE: **[MS-185]** fix connection no timeout bug 
### Improvement
---
- \#10 ISSUE: Update examples
- \#14 ISSUE: Update example
- \#28 ISSUE: Update README with newest sdk
- \#33 ISSUE: Update thrift has_table API
- \#38 ISSUE: Update thrift score to distance
- \#45, \#47 ISSUE:  Update AdvancedExample
- \#51 ISSUE: Brute force thread_safe for sdk
### New Feature
---
- \#3
    - transport protocol configurable by settings
    - add_vector support non-binary array inputs

- \#6 ISSUE   
    - Status quick-check-success
    - Log message more understandable
    - Status code related to Thrift Exception
    - Operations before connect will raise NotConnectError, Adding UNKNOWN Status

- \#8 ISSUE
    - Add new api: search_vectors_by_file
    - fix some bugs

- \#17 ISSUE
    - Implement has_table interface
    - fix spelling error, reformat as PEP8

- \#19 ISSUE
    - Hide Prepare object and support old version
    
- \#21 ISSUE
    - support search by range
    - fix server_status return None bug
### Task
---
- \#1 Build Repository

- \#2 Add CHANGELOG.md and LICENSE, update setup.py
