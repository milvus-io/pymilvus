pymilvus index params
=====================

IVF_FLAT
------------

**Create index param**

* nlist

  * number of inverted file cell list
  * Range: [1, 999999]
  * recommended value: 16384

**Search param**

* nprobe

  * number of inverted file cell to probe
  * Range: [1,999999]
  * recommended value: 32


IVF_PQ
------------

**Create index param**

* m

  * m is decided by dim and have a couple of results. each result represent a kind of compress ratio.

* nlist

  * number of inverted file cell list
  * Range: [1, 999999]
  * recommended value: 16384

**Search param**

* nprobe

  * number of inverted file cell to probe
  * Range: [1,999999]
  * recommended value: 32


IVF_SQ8/IVF_SQ8H
------------

**Create index param**

* m

  * m is decided by dim and have a couple of results. each result represent a kind of compress ratio.

* nlist

  * number of inverted file cell list
  * Range: [1, 999999]
  * recommended value: 16384

**Search param**

* nprobe

  * number of inverted file cell to probe
  * Range: [1,999999]
  * recommended value: 32
