### Bug
- \#12 ISSUE
    - IndexType change
    - fix param of search_vector_in_files ranges
- \#23 ISSUE: fix not-connect raise wrong exception bug
- \#25 ISSUE: *[M-112]* fix type of top_k not checked error
- \#26 ISSUE: format of server_version's return value fixed
- \#27 ISSUE: correct connect and disconnect logic    
### Improvement

- \#10 Update examples
- \#14 Update example
- \#28 Update README with newest sdk

### New Feature

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

- \#1 Build Repository

- \#2 Add CHANGELOG.md and LICENSE, update setup.py
