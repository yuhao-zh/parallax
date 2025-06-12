# Parallax
* Paged Attention + Continuous Batching for MAC;
* Pipeline Parallel sharding;
* Currently we suffer from the unacceptable I/O for whole kv cache pool .eval(),
* thus, we are currently working on Paged Flash Attention Kernel.
